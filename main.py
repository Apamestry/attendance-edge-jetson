#!/usr/bin/env python3
"""
Face Recognition Attendance System (Jetson Nano) - interactive CLI
- Blink-based anti-spoofing (liveness expires, so a photo can't reuse an old blink)
- Per-person photo folders, cached encodings (fast startup)
- On-screen Exit / Capture / Done buttons + Q / ESC keys + window close
Recognition pipeline, camera, members and attendance log live in edge_core.py
(shared with the web API in server.py).
"""

import os
import sys
import csv
import time
import shutil
from datetime import datetime

import cv2
import numpy as np
import face_recognition

from edge_core import (
    C, ok, warn, err, info,
    PHOTOS_DIR, ATTENDANCE_FILE, TARGET_PHOTOS, POSES, STATE_COLORS,
    open_camera, read_frame, ensure_environment, get_members, delete_member as delete_member_files,
    detect_faces, face_state, Engine,
)


# =========================================================================
# TERMINAL UI
# =========================================================================
def banner(title, width=56):
    print(f"\n{C.CYAN}{'=' * width}")
    print(f"{C.BOLD}{title.center(width)}{C.RESET}{C.CYAN}")
    print(f"{'=' * width}{C.RESET}")


def pause():
    input(f"{C.DIM}Press Enter to continue...{C.RESET}")


# =========================================================================
# OPENCV WINDOW UI
# =========================================================================
BG_PANEL = (35, 30, 28)
WHITE = (255, 255, 255)
GREEN = (80, 200, 90)
YELLOW = (0, 215, 255)
RED = (60, 60, 230)
BLUE = (230, 160, 50)
GREY = (170, 170, 170)
HEADER_H = 44
FOOTER_H = 30
FONT = cv2.FONT_HERSHEY_DUPLEX


def overlay_rect(frame, p1, p2, color, alpha=0.65):
    h, w = frame.shape[:2]
    x1, y1 = max(0, p1[0]), max(0, p1[1])
    x2, y2 = min(w, p2[0]), min(h, p2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    colored = np.full_like(roi, color)
    cv2.addWeighted(colored, alpha, roi, 1 - alpha, 0, roi)


def put_text(frame, text, org, scale=0.6, color=WHITE, thickness=1):
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def text_width(text, scale=0.6, thickness=1):
    return cv2.getTextSize(text, FONT, scale, thickness)[0][0]


class Buttons:
    """Tiny clickable-button helper for OpenCV windows."""

    def __init__(self):
        self.rects = {}
        self.clicked = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for name, (x1, y1, x2, y2) in self.rects.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.clicked = name

    def pop(self):
        c, self.clicked = self.clicked, None
        return c

    def draw(self, frame, name, label, rect, color):
        self.rects[name] = rect
        x1, y1, x2, y2 = rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FILLED)
        cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 1)
        tw = text_width(label, 0.55)
        cv2.putText(frame, label, (x1 + (x2 - x1 - tw) // 2, y2 - (y2 - y1 - 12) // 2),
                    FONT, 0.55, WHITE, 1, cv2.LINE_AA)


def draw_header(frame, title, right_text=""):
    h, w = frame.shape[:2]
    overlay_rect(frame, (0, 0), (w, HEADER_H), BG_PANEL, 0.8)
    cv2.line(frame, (0, HEADER_H), (w, HEADER_H), BLUE, 2)
    put_text(frame, title, (14, 29), 0.7, WHITE, 1)
    if right_text:
        put_text(frame, right_text, (w - 110 - text_width(right_text, 0.5) - 10, 28), 0.5, GREY)


def draw_footer(frame, text):
    h, w = frame.shape[:2]
    overlay_rect(frame, (0, h - FOOTER_H), (w, h), BG_PANEL, 0.8)
    put_text(frame, text, (12, h - 10), 0.5, GREY)


def draw_toast(frame, text, color=GREEN):
    h, w = frame.shape[:2]
    y1 = HEADER_H + 10
    tw = text_width(text, 0.7, 1)
    x1 = max(10, (w - tw) // 2 - 16)
    overlay_rect(frame, (x1, y1), (min(w - 10, x1 + tw + 32), y1 + 42), color, 0.85)
    put_text(frame, text, (x1 + 16, y1 + 28), 0.7, WHITE)


def draw_face_box(frame, loc, color, thickness=2):
    top, right, bottom, left = loc
    length = max(14, (right - left) // 4)
    for x, y, dx, dy in ((left, top, 1, 1), (right, top, -1, 1),
                         (left, bottom, 1, -1), (right, bottom, -1, -1)):
        cv2.line(frame, (x, y), (x + dx * length, y), color, thickness + 1, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + dy * length), color, thickness + 1, cv2.LINE_AA)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 1)


def draw_label(frame, loc, text, color):
    top, right, bottom, left = loc
    h, w = frame.shape[:2]
    tw = text_width(text, 0.55)
    x1, x2 = left, max(right, left + tw + 14)
    y1 = min(bottom + 4, h - FOOTER_H - 28)
    cv2.rectangle(frame, (x1, y1), (x2, y1 + 26), color, cv2.FILLED)
    cv2.putText(frame, text, (x1 + 7, y1 + 18), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


_seen_visible = {}


def window_closed(win, frames_seen):
    """True only if the window was visible before and now is gone (X button clicked).
    Some OpenCV builds on Jetson always report 'not visible', so we never trust
    the check unless it has reported 'visible' at least once."""
    if frames_seen <= 2:
        _seen_visible.pop(win, None)
        return False
    try:
        visible = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE)
    except cv2.error:
        return False
    if visible >= 1:
        _seen_visible[win] = True
        return False
    return _seen_visible.get(win, False)


def flush_stdin():
    """Drop keys typed into the terminal while the camera window was open."""
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


# =========================================================================
# ATTENDANCE TRACKER
# =========================================================================
WIN_TRACK = "Attendance Tracker"


def start_attendance_system():
    banner("ATTENDANCE TRACKER")
    engine = Engine()
    try:
        engine.start_tracking()     # loads faces, opens camera, starts recognition thread
    except RuntimeError as e:
        err(str(e))
        return

    btns = Buttons()
    cv2.namedWindow(WIN_TRACK, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_TRACK, btns.on_mouse)

    seq, frame_count = 0, 0
    info("Running. Press Q or ESC (or click EXIT) to stop.")

    try:
        while engine.mode == "tracking":
            seq, frame = engine.camera.read(seq, timeout=1.0)
            if frame is None:
                continue
            frame = frame.copy()     # the camera frame is shared with the recognition thread
            frame_count += 1

            # ---- draw latest recognition results ----
            for r in engine.results:
                state, label = face_state(r, engine.log)
                color = STATE_COLORS[state]
                draw_face_box(frame, r["loc"], color)
                draw_label(frame, r["loc"], label, color)

            now = time.time()
            draw_header(frame, "ATTENDANCE TRACKER",
                        f"{engine.fps:4.1f} FPS | {datetime.now().strftime('%H:%M:%S')}")
            h, w = frame.shape[:2]
            btns.draw(frame, "exit", "EXIT", (w - 98, 7, w - 8, HEADER_H - 7), (50, 50, 200))
            marked_name, marked_at = engine.last_mark
            if marked_name and now < marked_at + 3.0:
                draw_toast(frame, f"Attendance marked: {marked_name}")
            draw_footer(frame, f"Present today: {engine.log.count_today()}   |   Q / ESC = exit")

            cv2.imshow(WIN_TRACK, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or btns.pop() == "exit" \
                    or window_closed(WIN_TRACK, frame_count):
                break
    except KeyboardInterrupt:
        print()
    finally:
        engine.stop_tracking()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        info("Tracker stopped.")


# =========================================================================
# REGISTRATION
# =========================================================================
WIN_REG = "Register New Member"


def add_new_member_webcam():
    banner("ADD NEW MEMBER")
    member_name = input("Full name (Enter to cancel): ").strip().title()
    if not member_name:
        info("Cancelled.")
        return

    safe = "".join(c for c in member_name if c.isalnum() or c == " ").strip().replace(" ", "_")
    if not safe:
        err("Invalid name.")
        return
    person_dir = os.path.join(PHOTOS_DIR, safe)
    existed = os.path.isdir(person_dir)
    os.makedirs(person_dir, exist_ok=True)

    cap = open_camera()
    if not cap.isOpened():
        err("Cannot access camera.")
        if not existed:
            shutil.rmtree(person_dir, ignore_errors=True)
        return

    btns = Buttons()
    cv2.namedWindow(WIN_REG, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_REG, btns.on_mouse)
    info(f"Registering {member_name}. SPACE/S = capture, Q/ESC = finish.")

    photo_count, frame_count = 0, 0
    msg, msg_color, msg_until = "", GREEN, 0.0
    last_capture = 0.0

    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                err("Lost camera feed.")
                break
            frame_count += 1
            view = frame.copy()
            h, w = view.shape[:2]

            pose = POSES[photo_count % len(POSES)]
            draw_header(view, f"REGISTERING: {member_name}", f"Photos: {photo_count}/{TARGET_PHOTOS}")
            btns.draw(view, "done", "DONE", (w - 98, 7, w - 8, HEADER_H - 7), (50, 50, 200))
            btns.draw(view, "capture", "CAPTURE", (w // 2 - 70, h - FOOTER_H - 46, w // 2 + 70, h - FOOTER_H - 8),
                      (60, 160, 60))
            put_text(view, f"Next: {pose}", (14, HEADER_H + 30), 0.7, YELLOW)
            if time.time() < msg_until:
                draw_toast(view, msg, msg_color)
            draw_footer(view, "SPACE / S = capture   |   Q / ESC = finish")

            cv2.imshow(WIN_REG, view)
            key = cv2.waitKey(1) & 0xFF
            click = btns.pop()

            if key in (ord("q"), ord("Q"), 27) or click == "done" or window_closed(WIN_REG, frame_count):
                break

            if (key in (32, ord("s"), ord("S")) or click == "capture") and time.time() - last_capture > 0.5:
                last_capture = time.time()
                faces = detect_faces(frame)
                if len(faces) == 0:
                    msg, msg_color = "No face detected - look at the camera", RED
                elif len(faces) > 1:
                    msg, msg_color = "Multiple faces - only one person please", RED
                else:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if not face_recognition.face_encodings(rgb, faces):
                        msg, msg_color = "Face unclear - try again", RED
                    else:
                        fname = f"{safe}_{int(time.time())}_{photo_count}.jpg"
                        # Save the un-annotated, full-quality frame
                        cv2.imwrite(os.path.join(person_dir, fname), frame)
                        photo_count += 1
                        msg, msg_color = f"Saved photo {photo_count}", GREEN
                        print(f"{C.GREEN}[ OK ]{C.RESET} Saved photo {photo_count} for '{member_name}'")
                msg_until = time.time() + 1.5
    except KeyboardInterrupt:
        print()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    if photo_count == 0 and not existed:
        shutil.rmtree(person_dir, ignore_errors=True)
        warn("No photos saved - registration cancelled.")
    else:
        ok(f"Finished. {photo_count} new photo(s) saved for {member_name}.")
        if photo_count < 3:
            warn("Tip: 3-5 photos from different angles give much better accuracy.")


# =========================================================================
# MEMBER & LOG MANAGEMENT
# =========================================================================
def view_members():
    banner("REGISTERED MEMBERS")
    members = get_members()
    if not members:
        print("  No members registered yet.")
    else:
        print(f"  {'#':<4}{'Name':<28}{'Photos':>6}")
        print(f"  {'-' * 38}")
        for i, (name, paths) in enumerate(members.items(), start=1):
            print(f"  {i:<4}{name:<28}{len(paths):>6}")
        print(f"\n  Total members: {len(members)}")


def delete_member():
    banner("DELETE A MEMBER")
    members = get_members()
    if not members:
        err("No members available to delete.")
        return
    names = list(members.keys())
    for i, name in enumerate(names, start=1):
        print(f"  [{i}] {name}  ({len(members[name])} photos)")
    print("  [0] Cancel")

    choice = input("\nNumber of the member to delete: ").strip()
    if not choice.isdigit() or int(choice) == 0 or int(choice) > len(names):
        info("Cancelled." if choice in ("", "0") else "Invalid selection.")
        return

    name = names[int(choice) - 1]
    if input(f"{C.YELLOW}Delete '{name}' and all their photos? (y/N): {C.RESET}").strip().lower() != "y":
        info("Cancelled.")
        return

    delete_member_files(name, members[name])
    ok(f"Deleted '{name}'. (Past attendance records are kept.)")


def view_attendance_logs(today_only=False):
    ensure_environment()
    today = datetime.now().strftime("%Y-%m-%d")
    banner("TODAY'S ATTENDANCE" if today_only else "ALL ATTENDANCE LOGS")
    with open(ATTENDANCE_FILE, "r") as f:
        rows = [r for r in csv.reader(f) if len(r) >= 3][1:]
    if today_only:
        rows = [r for r in rows if r[1] == today]
    if not rows:
        print("  No attendance recorded." if not today_only else "  Nobody marked present today yet.")
        return
    print(f"  {'Name':<24}{'Date':<14}{'Time'}")
    print(f"  {'-' * 46}")
    for name, date, tm in rows:
        print(f"  {name:<24}{date:<14}{tm}")
    print(f"\n  Total entries: {len(rows)}")


# =========================================================================
# MAIN MENU
# =========================================================================
def main_menu():
    ensure_environment()
    while True:
        banner("FACE RECOGNITION ATTENDANCE (JETSON)")
        print(f"  {C.BOLD}[1]{C.RESET} Start Attendance Tracker")
        print(f"  {C.BOLD}[2]{C.RESET} Add New Member (webcam)")
        print(f"  {C.BOLD}[3]{C.RESET} Delete a Member")
        print(f"  {C.BOLD}[4]{C.RESET} View Registered Members")
        print(f"  {C.BOLD}[5]{C.RESET} View Today's Attendance")
        print(f"  {C.BOLD}[6]{C.RESET} View All Attendance Logs")
        print(f"  {C.BOLD}[7]{C.RESET} Exit")

        flush_stdin()
        choice = input(f"\n{C.CYAN}Select an option (1-7): {C.RESET}").strip()
        try:
            if choice == "1":
                start_attendance_system()
            elif choice == "2":
                add_new_member_webcam()
            elif choice == "3":
                delete_member()
            elif choice == "4":
                view_members()
                pause()
            elif choice == "5":
                view_attendance_logs(today_only=True)
                pause()
            elif choice == "6":
                view_attendance_logs()
                pause()
            elif choice == "7":
                print("Goodbye!")
                sys.exit(0)
            else:
                err("Invalid option.")
        except KeyboardInterrupt:
            print("\nInterrupted - back to menu.")


if __name__ == "__main__":
    main_menu()