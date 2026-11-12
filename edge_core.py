#!/usr/bin/env python3
"""
Shared core for the Jetson Nano attendance edge agent.
Used by both the CLI (main.py) and the web API (server.py).

Speed-ups vs. the original single loop (all behaviour preserved):
- Camera is read in its own thread, so recognition always works on the newest frame
  (no stale USB buffer lag) and the preview stays smooth.
- Faces are *tracked* between detections: the expensive detector runs every
  `detect_every` cycles, the face encoder only on new faces or every
  `reverify_seconds`. The cheap landmark pass runs every cycle, which also makes
  blink detection far more reliable at low FPS.
- One landmark pass per face per cycle (the old code ran two), numpy instead of scipy
  (faster startup, less RAM), known encodings kept as one numpy matrix.

Python 3.6+ compatible (JetPack 4.x ships Python 3.6).
"""

import os
import sys
import csv
import json
import time
import base64
import pickle
import shutil
import socket
import threading
import warnings
import collections
from datetime import datetime

import cv2
import numpy as np
import face_recognition

warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

# =========================================================================
# CONFIGURATION
# =========================================================================
PHOTOS_DIR = "photos"
ATTENDANCE_FILE = "attendance.csv"
ENCODINGS_CACHE = "encodings_cache.pkl"
SETTINGS_FILE = "settings.json"

USE_CSI_CAMERA = False
CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 640, 480      # USB camera resolution

# Registration
TARGET_PHOTOS = 5
POSES = ["Look straight", "Turn slightly LEFT", "Turn slightly RIGHT",
         "Tilt slightly UP", "Tilt slightly DOWN"]

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
DEVICE_ID = os.environ.get("EDGE_DEVICE_ID", socket.gethostname())
VERSION = "1.1"

# Runtime-tunable settings (the web UI can change these; saved to settings.json)
SETTINGS = {
    "flip_mode": "horizontal",       # "none", "horizontal" (mirror), "vertical", "both" (= rotate 180)
    "detection_model": "cnn",        # "cnn" = GPU (CUDA), "hog" = CPU
    "face_match_tolerance": 0.45,    # lower = stricter (0.6 is the library default)
    "scale_factor": 0.4,             # detection runs on a shrunken frame (lower = faster, less accurate)
    "upsample": 1,                   # detector upsampling. 0 = ~4x faster, but faces must be closer/bigger
    "eye_ar_thresh": 0.22,           # anti-spoofing (blink detection)
    "eye_ar_consec_frames": 2,
    "liveness_seconds": 10,          # a blink counts as proof of life for this long
    "detect_every": 3,               # run the full detector every N cycles while faces are tracked
    "reverify_seconds": 1.0,         # re-run the face encoder on a tracked face this often
    "unknown_alert_seconds": 30,     # min gap between "unknown face" alerts (0 = off)
}

SETTING_RULES = {
    "flip_mode": ("choice", ["none", "horizontal", "vertical", "both"]),
    "detection_model": ("choice", ["cnn", "hog"]),
    "face_match_tolerance": ("float", 0.2, 0.8),
    "scale_factor": ("float", 0.1, 1.0),
    "upsample": ("int", 0, 2),
    "eye_ar_thresh": ("float", 0.1, 0.4),
    "eye_ar_consec_frames": ("int", 1, 10),
    "liveness_seconds": ("float", 1, 300),
    "detect_every": ("int", 1, 30),
    "reverify_seconds": ("float", 0, 30),
    "unknown_alert_seconds": ("float", 0, 3600),
}

# dlib / CUDA objects are not safe to use from several threads at once
DLIB_LOCK = threading.Lock()


def validate_settings(updates):
    """Returns a cleaned dict, raises ValueError on bad keys/values."""
    clean = {}
    for key, value in updates.items():
        rule = SETTING_RULES.get(key)
        if rule is None:
            raise ValueError("Unknown setting: %s" % key)
        if rule[0] == "choice":
            if value not in rule[1]:
                raise ValueError("%s must be one of %s" % (key, rule[1]))
        else:
            try:
                value = int(value) if rule[0] == "int" else float(value)
            except (TypeError, ValueError):
                raise ValueError("%s must be a number" % key)
            if not rule[1] <= value <= rule[2]:
                raise ValueError("%s must be between %s and %s" % (key, rule[1], rule[2]))
        clean[key] = value
    return clean


def update_settings(updates):
    SETTINGS.update(validate_settings(updates))
    with open(SETTINGS_FILE, "w") as f:
        json.dump(SETTINGS, f, indent=2)


def _load_settings():
    if not os.path.exists(SETTINGS_FILE):
        return
    try:
        with open(SETTINGS_FILE) as f:
            saved = json.load(f)
        SETTINGS.update(validate_settings({k: v for k, v in saved.items() if k in SETTING_RULES}))
    except Exception as e:
        print("[WARN] Ignoring %s: %s" % (SETTINGS_FILE, e))


_load_settings()

# =========================================================================
# TERMINAL OUTPUT
# =========================================================================
class C:
    on = sys.stdout.isatty()
    RESET = "\033[0m" if on else ""
    BOLD = "\033[1m" if on else ""
    RED = "\033[91m" if on else ""
    GREEN = "\033[92m" if on else ""
    YELLOW = "\033[93m" if on else ""
    CYAN = "\033[96m" if on else ""
    DIM = "\033[2m" if on else ""


def ok(msg):   print(f"{C.GREEN}[ OK ]{C.RESET} {msg}")
def warn(msg): print(f"{C.YELLOW}[WARN]{C.RESET} {msg}")
def err(msg):  print(f"{C.RED}[FAIL]{C.RESET} {msg}")
def info(msg): print(f"{C.CYAN}[INFO]{C.RESET} {msg}")


# =========================================================================
# CAMERA
# =========================================================================
def get_gstreamer_pipeline(sensor_id=0, capture_w=1280, capture_h=720,
                           display_w=640, display_h=360, framerate=30, flip_method=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_w}, height=(int){capture_h}, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_w}, height=(int){display_h}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1"
    )


def open_camera():
    if USE_CSI_CAMERA:
        return cv2.VideoCapture(get_gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always hand out the newest frame
    return cap


_FLIP_CODES = {"horizontal": 1, "vertical": 0, "both": -1}


def apply_flip(frame):
    code = _FLIP_CODES.get(SETTINGS["flip_mode"])
    return frame if code is None else cv2.flip(frame, code)


def read_frame(cap, retries=5):
    """Read one frame (already flipped). Returns None if the camera is really gone."""
    for _ in range(retries):
        ok_, frame = cap.read()
        if ok_ and frame is not None:
            return apply_flip(frame)
        time.sleep(0.05)
    return None


class Camera(object):
    """Reads frames in a background thread and keeps only the newest one.
    Frames handed out must be treated as read-only (copy before drawing)."""

    def __init__(self):
        self.cap = None
        self.frame = None
        self.seq = 0
        self.running = False
        self.failed = False
        self._cond = threading.Condition()
        self._thread = None

    def start(self):
        if self.running:
            return True
        cap = open_camera()
        if not cap.isOpened():
            cap.release()
            return False
        self.cap, self.frame, self.failed, self.running = cap, None, False, True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        fails = 0
        while self.running:
            ok_, frame = self.cap.read()
            if not ok_ or frame is None:
                fails += 1
                if fails >= 20:            # ~1 s without frames = camera gone
                    self.failed = True
                    break
                time.sleep(0.05)
                continue
            fails = 0
            frame = apply_flip(frame)
            with self._cond:
                self.frame = frame
                self.seq += 1
                self._cond.notify_all()
        self.running = False
        with self._cond:
            self._cond.notify_all()

    def read(self, last_seq=0, timeout=1.0):
        """Wait for a frame newer than last_seq. Returns (seq, frame) or (last_seq, None)."""
        with self._cond:
            self._cond.wait_for(lambda: self.seq != last_seq or not self.running, timeout)
            if self.seq != last_seq and self.frame is not None:
                return self.seq, self.frame
            return last_seq, None

    def latest(self):
        with self._cond:
            return self.seq, self.frame

    def stop(self):
        self.running = False
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
        self.cap, self._thread, self.frame = None, None, None


# =========================================================================
# FILES, MEMBERS & ENCODINGS
# =========================================================================
def is_image(name):
    return name.lower().endswith(IMAGE_EXTS)


def ensure_environment():
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Name", "Date", "Time"])


def get_members():
    """Returns {display_name: [photo paths]}.
    New layout: photos/<Name_Surname>/*.jpg   Legacy layout: photos/<Name>_<id>.jpg"""
    ensure_environment()
    members = {}
    for entry in sorted(os.listdir(PHOTOS_DIR)):
        path = os.path.join(PHOTOS_DIR, entry)
        if os.path.isdir(path):
            files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if is_image(f)]
            if files:
                members.setdefault(entry.replace("_", " "), []).extend(files)
        elif is_image(entry):
            base = os.path.splitext(entry)[0].strip()
            name = base.split("_")[0].split("-")[0].strip().title()
            members.setdefault(name, []).append(path)
    return members


def member_id(name):
    """Stable ID used by the API: the photo folder name (spaces -> underscores)."""
    return name.replace(" ", "_")


def safe_name(name):
    return "".join(c for c in name if c.isalnum() or c == " ").strip().replace(" ", "_")


def list_members():
    out = []
    for name, paths in get_members().items():
        try:
            registered = datetime.fromtimestamp(min(os.path.getmtime(p) for p in paths)).strftime("%Y-%m-%dT%H:%M:%S")
        except OSError:
            registered = None
        out.append({"id": member_id(name), "name": name, "photo_count": len(paths),
                    "registered_at": registered})
    return out


def find_member(mid):
    for name, paths in get_members().items():
        if member_id(name) == mid:
            return name, paths
    return None, []


def delete_member(name, paths):
    """Delete a member's photos (past attendance records are kept)."""
    for path in paths:
        try:
            os.remove(path)
        except OSError as e:
            err(f"Could not delete {path}: {e}")
    folder = os.path.join(PHOTOS_DIR, member_id(name))
    if os.path.isdir(folder) and not os.listdir(folder):
        os.rmdir(folder)


def encode_image_file(path):
    image = face_recognition.load_image_file(path)
    h, w = image.shape[:2]
    if max(h, w) > 800:  # huge photos are slow on the Nano and give no benefit
        s = 800 / max(h, w)
        image = cv2.resize(image, (int(w * s), int(h * s)))
    encs = face_recognition.face_encodings(image)
    return encs[0] if encs else None


def load_known_faces():
    members = get_members()
    cache = {}
    if os.path.exists(ENCODINGS_CACHE):
        try:
            with open(ENCODINGS_CACHE, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    new_cache, encodings, names = {}, [], []
    all_paths = [(n, p) for n, ps in members.items() for p in ps]
    for i, (name, path) in enumerate(all_paths, start=1):
        try:
            st = os.stat(path)
            sig = (st.st_mtime, st.st_size)
            cached = cache.get(path)
            if cached and cached[0] == sig:
                enc = cached[1]
            else:
                print(f"\r  Encoding photos... {i}/{len(all_paths)}", end="", flush=True)
                enc = encode_image_file(path)
                if enc is None:
                    warn(f"No face found in {path}")
            new_cache[path] = (sig, enc)
            if enc is not None:
                encodings.append(enc)
                names.append(name)
        except Exception as e:
            err(f"Could not load {path}: {e}")
    print("\r" + " " * 40 + "\r", end="")

    try:
        with open(ENCODINGS_CACHE, "wb") as f:
            pickle.dump(new_cache, f)
    except Exception:
        pass
    return encodings, names


def save_face_photo(frame, person_dir, safe, tag):
    """Validate that the BGR frame holds exactly one usable face, then save it.
    Returns None on success or an error message."""
    with DLIB_LOCK:
        faces = detect_faces(frame)
        if len(faces) == 0:
            return "No face detected - look at the camera"
        if len(faces) > 1:
            return "Multiple faces - only one person please"
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not face_recognition.face_encodings(rgb, faces):
            return "Face unclear - try again"
    os.makedirs(person_dir, exist_ok=True)
    cv2.imwrite(os.path.join(person_dir, f"{safe}_{int(time.time())}_{tag}.jpg"), frame)
    return None


# =========================================================================
# ATTENDANCE LOG
# =========================================================================
class AttendanceLog:
    """Keeps today's marked names in memory so the CSV isn't re-read every frame.
    CSV format is unchanged: Name, Date, Time. Thread-safe."""

    def __init__(self):
        ensure_environment()
        self.date = None
        self.marked = set()
        self._lock = threading.RLock()
        self._load()

    def _load(self):
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.marked = set()
        with open(ATTENDANCE_FILE, "r") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[1] == self.date:
                    self.marked.add(row[0])

    def _roll(self):
        if datetime.now().strftime("%Y-%m-%d") != self.date:  # app left running past midnight
            self._load()

    def is_marked(self, name):
        with self._lock:
            self._roll()
            return name in self.marked

    def count_today(self):
        with self._lock:
            self._roll()
            return len(self.marked)

    def mark(self, name):
        """Mark `name` present now. Returns the new row dict, or None if already marked today."""
        with self._lock:
            self._roll()
            if name in self.marked:
                return None
            now = datetime.now()
            row = {"name": name, "date": self.date, "time": now.strftime("%H:%M:%S")}
            with open(ATTENDANCE_FILE, "a", newline="") as f:
                csv.writer(f).writerow([row["name"], row["date"], row["time"]])
            self.marked.add(name)
        print(f"{C.GREEN}[ OK ]{C.RESET} Attendance marked: {name} at {row['time']}")
        return row

    def mark_manual(self, name, date, tm):
        """Admin override for any date. Returns the row, or None if it already exists."""
        with self._lock:
            self._roll()
            if date == self.date:
                return self.mark(name)
            if any(r["name"] == name and r["date"] == date for r in self.rows(date=date)):
                return None
            with open(ATTENDANCE_FILE, "a", newline="") as f:
                csv.writer(f).writerow([name, date, tm])
            return {"name": name, "date": date, "time": tm}

    def rows(self, date=None, start=None, end=None):
        with self._lock:
            ensure_environment()
            with open(ATTENDANCE_FILE, "r") as f:
                rows = [r for r in csv.reader(f) if len(r) >= 3]
        out = []
        for name, d, tm in (r[:3] for r in rows):
            if (name, d) == ("Name", "Date"):
                continue
            if (date and d != date) or (start and d < start) or (end and d > end):
                continue
            out.append({"name": name, "date": d, "time": tm})
        return out

    def remove(self, name, date):
        """Delete the attendance entry for name on date. Returns True if something was removed."""
        with self._lock:
            with open(ATTENDANCE_FILE, "r") as f:
                rows = list(csv.reader(f))
            keep = [r for r in rows if not (len(r) >= 2 and r[0] == name and r[1] == date)]
            if len(keep) == len(rows):
                return False
            with open(ATTENDANCE_FILE, "w", newline="") as f:
                csv.writer(f).writerows(keep)
            self._load()
            return True


# =========================================================================
# FACE PIPELINE
# =========================================================================
def detect_faces(frame, rgb=None):
    """Detect on a shrunken frame, return locations scaled back to full size.
    Pass `rgb` if you already have the RGB version of the frame."""
    s = SETTINGS["scale_factor"]
    if rgb is None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (0, 0), fx=s, fy=s)
    small_locs = face_recognition.face_locations(
        small, number_of_times_to_upsample=SETTINGS["upsample"], model=SETTINGS["detection_model"])
    h, w = rgb.shape[:2]
    locs = []
    for t, r, b, l in small_locs:
        t, r, b, l = (int(v / s) for v in (t, r, b, l))
        locs.append((max(0, t), min(w - 1, r), min(h - 1, b), max(0, l)))
    return locs


def eye_aspect_ratio(eye):
    p = np.asarray(eye, dtype=np.float32)
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3])
    return float((a + b) / (2.0 * c)) if c else 0.0


def _iou(a, b):
    at, ar, ab, al = a
    bt, br, bb, bl = b
    iw, ih = min(ar, br) - max(al, bl), min(ab, bb) - max(at, bt)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / float((ar - al) * (ab - at) + (br - bl) * (bb - bt) - inter)


class _Track(object):
    __slots__ = ("loc", "name", "distance", "enc_time", "offset")

    def __init__(self, loc):
        self.loc = loc
        self.name = "Unknown"
        self.distance = None
        self.enc_time = 0.0     # 0 = never encoded -> encode on the next cycle
        self.offset = None      # box centre minus landmark centroid


class FacePipeline(object):
    """Detect -> track -> (occasionally) encode -> blink liveness."""

    def __init__(self, known_encodings, known_names):
        self.blink = {}         # name -> {"closed", "live_until"} (kept across reloads)
        self.timings = {"detect": 0.0, "landmarks": 0.0, "encode": 0.0}
        self.set_known(known_encodings, known_names)

    def set_known(self, known_encodings, known_names):
        self.known = np.asarray(known_encodings, dtype=np.float64).reshape(-1, 128)
        self.names = list(known_names)
        self.tracks = []
        self.cycle = 0

    def _time(self, key, t0):
        self.timings[key] = 0.8 * self.timings[key] + 0.2 * (time.time() - t0) * 1000.0

    def process(self, frame):
        now = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        if not self.tracks or self.cycle % int(SETTINGS["detect_every"]) == 0:
            t0 = time.time()
            self._match(detect_faces(frame, rgb))
            self._time("detect", t0)
        self.cycle += 1
        if not self.tracks:
            return []

        t0 = time.time()
        landmarks = face_recognition.face_landmarks(rgb, [t.loc for t in self.tracks])
        for t, lm in zip(self.tracks, landmarks):
            self._follow(t, lm, w, h)
        self._time("landmarks", t0)

        stale = [t for t in self.tracks if now - t.enc_time >= SETTINGS["reverify_seconds"]]
        if stale:
            t0 = time.time()
            encs = face_recognition.face_encodings(rgb, [t.loc for t in stale])
            for t, enc in zip(stale, encs):
                self._identify(t, enc, now)
            self._time("encode", t0)

        results = []
        for t, lm in zip(self.tracks, landmarks):
            live = t.name != "Unknown" and self._update_liveness(t.name, lm, now)
            results.append({"loc": t.loc, "name": t.name, "live": live, "distance": t.distance})
        return results

    def _match(self, locs):
        """Keep identities of tracks that overlap a new detection; drop the rest."""
        pairs = sorted(((_iou(t.loc, l), ti, li) for ti, t in enumerate(self.tracks)
                        for li, l in enumerate(locs)), reverse=True)
        used_t, used_l, kept = set(), set(), []
        for iou, ti, li in pairs:
            if iou < 0.3:
                break
            if ti in used_t or li in used_l:
                continue
            used_t.add(ti)
            used_l.add(li)
            t = self.tracks[ti]
            t.loc, t.offset = locs[li], None
            kept.append(t)
        kept.extend(_Track(l) for li, l in enumerate(locs) if li not in used_l)
        self.tracks = kept

    @staticmethod
    def _follow(t, lm, w, h):
        """Between detections, move the box with the landmark centroid (size stays fixed)."""
        pts = np.array([p for feature in lm.values() for p in feature], dtype=np.float32)
        if not len(pts):
            return
        cx, cy = pts.mean(axis=0)
        top, right, bottom, left = t.loc
        bx, by = (left + right) / 2.0, (top + bottom) / 2.0
        if t.offset is None:
            t.offset = (bx - cx, by - cy)
            return
        dx = int(round(cx + t.offset[0] - bx))
        dy = int(round(cy + t.offset[1] - by))
        t.loc = (max(0, top + dy), min(w - 1, right + dx), min(h - 1, bottom + dy), max(0, left + dx))

    def _identify(self, t, enc, now):
        name, d = "Unknown", None
        if len(self.known):
            distances = np.linalg.norm(self.known - enc, axis=1)
            best = int(np.argmin(distances))
            d = float(distances[best])
            if d <= SETTINGS["face_match_tolerance"]:
                name = self.names[best]
        t.name, t.distance, t.enc_time = name, d, now

    def _update_liveness(self, name, landmarks, now):
        st = self.blink.setdefault(name, {"closed": 0, "live_until": 0.0})
        if "left_eye" in landmarks and "right_eye" in landmarks:
            ear = (eye_aspect_ratio(landmarks["left_eye"]) +
                   eye_aspect_ratio(landmarks["right_eye"])) / 2.0
            if ear < SETTINGS["eye_ar_thresh"]:
                st["closed"] += 1
            else:
                if st["closed"] >= SETTINGS["eye_ar_consec_frames"]:
                    st["live_until"] = now + SETTINGS["liveness_seconds"]
                st["closed"] = 0
        return now < st["live_until"]


def face_state(result, log):
    """(state, label) for drawing. state in unknown / present / verified / blink."""
    name = result["name"]
    if name == "Unknown":
        return "unknown", "Unknown"
    if log.is_marked(name):
        return "present", f"{name} - PRESENT"
    if result["live"]:
        return "verified", f"{name} - VERIFIED"
    return "blink", f"{name} - Blink to verify"


STATE_COLORS = {"unknown": (60, 60, 230), "present": (80, 200, 90),
                "verified": (80, 200, 90), "blink": (0, 215, 255)}


# =========================================================================
# EVENTS
# =========================================================================
class EventBus(object):
    """Ring buffer of recent events. IDs start from boot time in ms, so they keep
    increasing across restarts and clients can resume with Last-Event-ID."""

    def __init__(self, maxlen=200):
        self._cond = threading.Condition()
        self._buf = collections.deque(maxlen=maxlen)
        self.last_id = int(time.time() * 1000)

    def emit(self, etype, data):
        with self._cond:
            self.last_id += 1
            self._buf.append({"id": self.last_id, "type": etype, "device_id": DEVICE_ID,
                              "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "data": data})
            self._cond.notify_all()

    def wait(self, after_id, timeout):
        with self._cond:
            self._cond.wait_for(lambda: self.last_id > after_id, timeout)
            return [e for e in self._buf if e["id"] > after_id]


# =========================================================================
# ENGINE (owns the camera; one mode at a time: idle / tracking / registering)
# =========================================================================
class Busy(Exception):
    pass


class Engine(object):
    def __init__(self):
        self.camera = Camera()
        self.log = AttendanceLog()
        self.bus = EventBus()
        self.mode = "idle"
        self.error = None
        self.started = time.time()
        self.results = []
        self.fps = 0.0
        self.last_mark = (None, 0.0)   # (name, time) for the CLI toast
        self.reg = None
        self.pipe = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._reload = False
        self._last_unknown = 0.0

    # ---------------- status ----------------
    def status(self):
        members = get_members()
        timings = self.pipe.timings if (self.pipe and self.mode == "tracking") else {}
        return {
            "device_id": DEVICE_ID, "version": VERSION, "mode": self.mode, "error": self.error,
            "camera_ok": self.camera.running, "fps": round(self.fps, 1),
            "members": len(members), "photos": sum(len(p) for p in members.values()),
            "present_today": self.log.count_today(), "today": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "uptime_s": int(time.time() - self.started),
            "timings_ms": {k: round(v, 1) for k, v in timings.items()},
            "registration": self.registration_state(),
        }

    def _set_mode(self, mode, error=None):
        self.mode, self.error = mode, error
        self.bus.emit("status", self.status())

    # ---------------- tracking ----------------
    def start_tracking(self):
        with self._lock:
            if self.mode != "idle":
                raise Busy("Device is busy (%s)" % self.mode)
            info("Loading registered faces...")
            with DLIB_LOCK:
                encodings, names = load_known_faces()
            ok(f"Loaded {len(set(names))} member(s) from {len(names)} photo(s).")
            if not encodings:
                warn("No registered faces yet - everyone will show as Unknown.")
            if not self.camera.start():
                raise RuntimeError("Camera failed to open.")
            self.pipe = FacePipeline(encodings, names)
            self.results, self.fps, self._reload = [], 0.0, False
            self._stop.clear()
            self._thread = threading.Thread(target=self._track_loop, daemon=True)
            self._thread.start()
            self._set_mode("tracking")

    def stop_tracking(self):
        with self._lock:
            if self.mode != "tracking":
                return
            self._stop.set()
            thread = self._thread
        thread.join(timeout=10)

    def _track_loop(self):
        error, seq, last_t = None, 0, time.time()
        try:
            while not self._stop.is_set():
                if self._reload:
                    self._reload = False
                    with DLIB_LOCK:
                        encodings, names = load_known_faces()
                    self.pipe.set_known(encodings, names)
                    info(f"Reloaded faces: {len(set(names))} member(s).")

                seq, frame = self.camera.read(seq, timeout=1.0)
                if frame is None:
                    if not self.camera.running:
                        error = "Lost camera feed."
                        err(error)
                        break
                    continue

                with DLIB_LOCK:
                    results = self.pipe.process(frame)
                self.results = results

                for r in results:
                    if r["name"] != "Unknown" and r["live"]:
                        row = self.log.mark(r["name"])
                        if row:
                            self.last_mark = (r["name"], time.time())
                            self._emit_attendance(row, distance=r["distance"])
                self._maybe_alert_unknown(frame, results)

                now = time.time()
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - last_t, 1e-3))
                last_t = now
        except Exception as e:
            error = "Tracker crashed: %s" % e
            err(error)
        finally:
            self.camera.stop()
            self.results = []
            with self._lock:
                self._set_mode("idle", error)

    def _emit_attendance(self, row, distance=None, manual=False):
        data = dict(row, member_id=member_id(row["name"]), manual=manual, liveness=not manual)
        if distance is not None:
            data["distance"] = round(distance, 3)
            data["confidence"] = round(max(0.0, 1.0 - distance), 3)
        self.bus.emit("attendance_marked", data)

    def _maybe_alert_unknown(self, frame, results):
        gap = SETTINGS["unknown_alert_seconds"]
        now = time.time()
        if gap <= 0 or now - self._last_unknown < gap:
            return
        for r in results:
            if r["name"] == "Unknown":
                top, right, bottom, left = r["loc"]
                pad = (bottom - top) // 4
                crop = frame[max(0, top - pad):bottom + pad, max(0, left - pad):right + pad]
                if crop.size == 0:
                    return
                scale = 96.0 / crop.shape[1]
                crop = cv2.resize(crop, (96, max(1, int(crop.shape[0] * scale))))
                ok_, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok_:
                    self._last_unknown = now
                    self.bus.emit("unknown_face", {
                        "snapshot": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()})
                return

    # ---------------- remote registration (Nano camera) ----------------
    def registration_state(self):
        reg = self.reg
        if not reg:
            return None
        return {k: v for k, v in reg.items() if not k.startswith("_")}

    def start_registration(self, name):
        name = name.strip().title()
        safe = safe_name(name)
        if not safe:
            raise ValueError("Invalid name.")
        with self._lock:
            if self.mode != "idle":
                raise Busy("Device is busy (%s)" % self.mode)
            person_dir = os.path.join(PHOTOS_DIR, safe)
            if not self.camera.start():
                raise RuntimeError("Cannot access camera.")
            self.reg = {"member_id": safe, "name": name, "photos_saved": 0,
                        "target": TARGET_PHOTOS, "next_pose": POSES[0], "last_error": None,
                        "_dir": person_dir, "_existed": os.path.isdir(person_dir)}
            info(f"Remote registration started for {name}.")
            self._set_mode("registering")
            return self.registration_state()

    def capture(self):
        with self._lock:
            if self.mode != "registering":
                raise Busy("No registration in progress")
            reg = self.reg
            _, frame = self.camera.latest()
            error = "No camera frame yet" if frame is None else \
                save_face_photo(frame, reg["_dir"], reg["member_id"], reg["photos_saved"])
            if error is None:
                reg["photos_saved"] += 1
                reg["next_pose"] = POSES[reg["photos_saved"] % len(POSES)]
                ok(f"Saved photo {reg['photos_saved']} for '{reg['name']}'")
            reg["last_error"] = error
            state = self.registration_state()
            self.bus.emit("registration_progress", state)
            return error is None, state

    def finish_registration(self):
        with self._lock:
            if self.mode != "registering":
                raise Busy("No registration in progress")
            reg, self.reg = self.reg, None
            self.camera.stop()
            if reg["photos_saved"] == 0 and not reg["_existed"]:
                shutil.rmtree(reg["_dir"], ignore_errors=True)
                warn("No photos saved - registration cancelled.")
            else:
                ok(f"Finished. {reg['photos_saved']} new photo(s) saved for {reg['name']}.")
            self._set_mode("idle")
            if reg["photos_saved"]:
                self._members_changed()
            return {k: v for k, v in reg.items() if not k.startswith("_")}

    # ---------------- members ----------------
    def add_member_photos(self, name, images):
        """images: list of BGR numpy arrays. Works in any mode (dlib access is serialised)."""
        name = name.strip().title()
        safe = safe_name(name)
        if not safe:
            raise ValueError("Invalid name.")
        person_dir = os.path.join(PHOTOS_DIR, safe)
        existed = os.path.isdir(person_dir)
        saved, errors = 0, []
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            if max(h, w) > 800:
                s = 800.0 / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)))
            error = save_face_photo(img, person_dir, safe, "u%d" % i)
            if error:
                errors.append({"index": i, "error": error})
            else:
                saved += 1
        if saved == 0 and not existed:
            shutil.rmtree(person_dir, ignore_errors=True)
        if saved:
            ok(f"Uploaded {saved} photo(s) for {name}.")
            self._members_changed()
        return {"member_id": safe, "name": name, "saved": saved, "errors": errors}

    def delete_member(self, mid):
        name, paths = find_member(mid)
        if not name:
            return False
        delete_member(name, paths)
        ok(f"Deleted '{name}'. (Past attendance records are kept.)")
        self._members_changed()
        return True

    def reload_faces(self):
        self._reload = True

    def _members_changed(self):
        self._reload = True     # picked up by the tracker loop if it is running
        self.bus.emit("members_changed", {"members": list_members()})

    # ---------------- attendance (admin) ----------------
    def manual_mark(self, name, date=None, tm=None):
        date = date or datetime.now().strftime("%Y-%m-%d")
        tm = tm or datetime.now().strftime("%H:%M:%S")
        row = self.log.mark_manual(name, date, tm)
        if row:
            self._emit_attendance(row, manual=True)
        return row

    def remove_attendance(self, name, date):
        removed = self.log.remove(name, date)
        if removed:
            self.bus.emit("attendance_removed", {"name": name, "member_id": member_id(name), "date": date})
        return removed

    # ---------------- preview ----------------
    def annotate(self, frame):
        view = frame.copy()
        for r in self.results:
            state, label = face_state(r, self.log)
            color = STATE_COLORS[state]
            top, right, bottom, left = r["loc"]
            cv2.rectangle(view, (left, top), (right, bottom), color, 2)
            cv2.rectangle(view, (left, bottom), (max(right, left + 9 * len(label)), bottom + 22), color, -1)
            cv2.putText(view, label, (left + 4, bottom + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        reg = self.reg
        if reg:
            cv2.putText(view, "Next: %s (%d/%d)" % (reg["next_pose"], reg["photos_saved"], reg["target"]),
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
        return view

    def shutdown(self):
        if self.mode == "tracking":
            self.stop_tracking()
        elif self.mode == "registering":
            self.finish_registration()

