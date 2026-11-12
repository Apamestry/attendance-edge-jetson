#!/usr/bin/env python3
"""
Headless web API for the admin dashboard (runs on the Jetson Nano).
Standard library only - nothing extra to pip install. Python 3.6+.

    python3 server.py                 # API + dashboard on http://<nano-ip>:8000
    python3 server.py --autostart     # also start the attendance tracker immediately

Auth: every /api request needs the key, as header `X-API-Key: <key>` or `?key=<key>`
(the query form is for EventSource / <img>, which cannot send headers).
The key comes from $EDGE_API_KEY, else from .api_key (generated on first run).
See README.md for the full endpoint list.
"""

import os
import re
import hmac
import json
import time
import base64
import secrets
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs, unquote

import cv2
import numpy as np

import edge_core as core

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_INDEX = os.path.join(BASE_DIR, "web", "index.html")
KEY_FILE = os.path.join(BASE_DIR, ".api_key")
MAX_BODY = 25 * 1024 * 1024        # photo uploads are base64 JPEGs
PREVIEW_FPS = 5                    # MJPEG preview rate (only costs CPU while someone watches)
HEARTBEAT_S = 10                   # SSE status heartbeat
CORS_ORIGIN = os.environ.get("EDGE_CORS_ORIGIN", "*")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
TIME_RE = re.compile(r"\d{2}:\d{2}:\d{2}")


class ApiError(Exception):
    def __init__(self, status, message):
        Exception.__init__(self, message)
        self.status = status


def load_api_key():
    key = os.environ.get("EDGE_API_KEY")
    if key:
        return key
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            key = f.read().strip()
        if key:
            return key
    key = secrets.token_urlsafe(24)
    with open(KEY_FILE, "w") as f:
        f.write(key)
    try:
        os.chmod(KEY_FILE, 0o600)
    except OSError:
        pass
    return key


def decode_image(data):
    """base64 or data-URL JPEG/PNG -> BGR numpy image."""
    if not isinstance(data, str):
        raise ValueError("photo must be a base64 string")
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    img = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not a valid image")
    return img


class ThreadingServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    server_version = "AttendanceEdge/" + core.VERSION
    engine = None       # set in main()
    api_key = None

    # ---------------- plumbing ----------------
    def log_message(self, fmt, *args):
        pass    # keep the Nano's terminal quiet; errors are still printed

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", CORS_ORIGIN)
        self.send_header("Access-Control-Allow-Headers", "X-API-Key, Content-Type, Last-Event-ID")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")

    def _send(self, status, body, ctype):
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(status, json.dumps(obj).encode(), "application/json")

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ApiError(413, "Request too large")
        if not length:
            return {}
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except ValueError:
            raise ApiError(400, "Body must be JSON")
        if not isinstance(body, dict):
            raise ApiError(400, "Body must be a JSON object")
        return body

    def _q(self, name, default=None):
        return self.query.get(name, [default])[0]

    def _authorized(self):
        key = self.headers.get("X-API-Key") or self._q("key") or ""
        return hmac.compare_digest(key.encode(), self.api_key.encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_GET(self):    self._dispatch("GET")
    def do_POST(self):   self._dispatch("POST")
    def do_PUT(self):    self._dispatch("PUT")
    def do_DELETE(self): self._dispatch("DELETE")

    def _dispatch(self, method):
        url = urlparse(self.path)
        path = url.path.rstrip("/") or "/"
        self.query = parse_qs(url.query)
        try:
            if method == "GET" and path in ("/", "/index.html"):
                with open(WEB_INDEX, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if not self._authorized():
                raise ApiError(401, "Invalid or missing API key")
            for m, rx, fn in ROUTES:
                if m == method:
                    match = rx.fullmatch(path)
                    if match:
                        return fn(self, *[unquote(g) for g in match.groups()])
            raise ApiError(404, "Not found")
        except ApiError as e:
            self._json({"error": str(e)}, e.status)
        except core.Busy as e:
            self._json({"error": str(e)}, 409)
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            core.err(f"{method} {path}: {e}")
            try:
                self._json({"error": str(e)}, 500)
            except Exception:
                pass

    # ---------------- status / control ----------------
    def get_status(self):
        self._json(self.engine.status())

    def tracker_start(self):
        try:
            self.engine.start_tracking()
        except RuntimeError as e:
            raise ApiError(503, str(e))
        self._json(self.engine.status())

    def tracker_stop(self):
        self.engine.stop_tracking()
        self._json(self.engine.status())

    def reload(self):
        self.engine.reload_faces()
        self._json({"ok": True})

    # ---------------- config ----------------
    def get_config(self):
        rules = {k: list(v) for k, v in core.SETTING_RULES.items()}
        self._json({"values": core.SETTINGS, "rules": rules})

    def put_config(self):
        core.update_settings(self._body())
        self.engine.bus.emit("config_changed", core.SETTINGS)
        self.get_config()

    # ---------------- members ----------------
    def get_members(self):
        self._json({"members": core.list_members()})

    def member_photo(self, mid):
        name, paths = core.find_member(mid)
        if not paths:
            raise ApiError(404, "Member not found")
        with open(paths[0], "rb") as f:
            data = f.read()
        ctype = "image/png" if paths[0].lower().endswith(".png") else "image/jpeg"
        self._send(200, data, ctype)

    def add_member(self):
        body = self._body()
        name, photos = (body.get("name") or "").strip(), body.get("photos") or []
        if not name or not isinstance(photos, list) or not photos:
            raise ApiError(400, "Need 'name' and a non-empty 'photos' list")
        images, errors = [], []
        for i, p in enumerate(photos):
            try:
                images.append(decode_image(p))
            except Exception as e:
                errors.append({"index": i, "error": "Could not decode: %s" % e})
        result = self.engine.add_member_photos(name, images) if images else \
            {"member_id": core.safe_name(name.title()), "name": name.title(), "saved": 0, "errors": []}
        result["errors"] = errors + result["errors"]
        self._json(result, 201 if result["saved"] else 422)

    def delete_member(self, mid):
        if not self.engine.delete_member(mid):
            raise ApiError(404, "Member not found")
        self._json({"ok": True})

    # ---------------- remote registration ----------------
    def register_start(self):
        name = (self._body().get("name") or "").strip()
        if not name:
            raise ApiError(400, "Need 'name'")
        try:
            self._json(self.engine.start_registration(name))
        except RuntimeError as e:
            raise ApiError(503, str(e))

    def register_capture(self):
        saved, state = self.engine.capture()
        self._json(dict(state, saved=saved))

    def register_finish(self):
        self._json(self.engine.finish_registration())

    # ---------------- attendance ----------------
    def get_attendance(self):
        date, start, end = self._q("date"), self._q("from"), self._q("to")
        for v in (date, start, end):
            if v and not DATE_RE.fullmatch(v):
                raise ApiError(400, "Dates must be YYYY-MM-DD")
        self._json({"rows": self.engine.log.rows(date=date, start=start, end=end)})

    def post_attendance(self):
        body = self._body()
        name, date, tm = (body.get("name") or "").strip(), body.get("date"), body.get("time")
        if not name:
            raise ApiError(400, "Need 'name'")
        if (date and not DATE_RE.fullmatch(date)) or (tm and not TIME_RE.fullmatch(tm)):
            raise ApiError(400, "date must be YYYY-MM-DD and time HH:MM:SS")
        row = self.engine.manual_mark(name, date, tm)
        if not row:
            raise ApiError(409, "Already marked for that date")
        self._json(row, 201)

    def delete_attendance(self):
        name, date = self._q("name"), self._q("date")
        if not name or not date or not DATE_RE.fullmatch(date):
            raise ApiError(400, "Need ?name=...&date=YYYY-MM-DD")
        if not self.engine.remove_attendance(name, date):
            raise ApiError(404, "No such entry")
        self._json({"ok": True})

    # ---------------- streams ----------------
    def events(self):
        """Server-Sent Events: attendance_marked, status, members_changed, registration_progress,
        unknown_face, attendance_removed, config_changed. Status heartbeat every HEARTBEAT_S."""
        bus = self.engine.bus
        try:
            last = int(self.headers.get("Last-Event-ID") or self._q("last_id") or bus.last_id)
        except ValueError:
            last = bus.last_id
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self._sse("status", self.engine.status())
        while True:
            evs = bus.wait(last, HEARTBEAT_S)
            for e in evs:
                self._sse(e["type"], e, e["id"])
                last = e["id"]
            if not evs:
                self._sse("status", self.engine.status())

    def _sse(self, etype, data, eid=None):
        msg = ("id: %d\n" % eid if eid else "") + "event: %s\ndata: %s\n\n" % (etype, json.dumps(data))
        self.wfile.write(msg.encode())
        self.wfile.flush()

    def stream(self):
        """MJPEG live preview (with recognition boxes) while the camera is on."""
        cam = self.engine.camera
        if not cam.running:
            raise ApiError(409, "Camera is off - start the tracker or a registration first")
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        seq, interval = 0, 1.0 / PREVIEW_FPS
        while cam.running:
            t0 = time.time()
            seq, frame = cam.read(seq, timeout=1.0)
            if frame is None:
                continue
            ok_, jpg = cv2.imencode(".jpg", self.engine.annotate(frame), [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok_:
                data = jpg.tobytes()
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(data))
                self.wfile.write(data + b"\r\n")
                self.wfile.flush()
            time.sleep(max(0.0, interval - (time.time() - t0)))


def _r(method, pattern, fn):
    return method, re.compile(pattern), fn


ROUTES = [
    _r("GET", r"/api/status", Handler.get_status),
    _r("POST", r"/api/tracker/start", Handler.tracker_start),
    _r("POST", r"/api/tracker/stop", Handler.tracker_stop),
    _r("POST", r"/api/reload", Handler.reload),
    _r("GET", r"/api/config", Handler.get_config),
    _r("PUT", r"/api/config", Handler.put_config),
    _r("GET", r"/api/members", Handler.get_members),
    _r("POST", r"/api/members", Handler.add_member),
    _r("GET", r"/api/members/([^/]+)/photo", Handler.member_photo),
    _r("DELETE", r"/api/members/([^/]+)", Handler.delete_member),
    _r("POST", r"/api/register/start", Handler.register_start),
    _r("POST", r"/api/register/capture", Handler.register_capture),
    _r("POST", r"/api/register/finish", Handler.register_finish),
    _r("GET", r"/api/attendance", Handler.get_attendance),
    _r("POST", r"/api/attendance", Handler.post_attendance),
    _r("DELETE", r"/api/attendance", Handler.delete_attendance),
    _r("GET", r"/api/events", Handler.events),
    _r("GET", r"/api/stream", Handler.stream),
]


def main():
    ap = argparse.ArgumentParser(description="Attendance edge agent - web API")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("EDGE_PORT", 8000)))
    ap.add_argument("--autostart", action="store_true", help="start the attendance tracker on boot")
    args = ap.parse_args()

    Handler.api_key = load_api_key()
    Handler.engine = engine = core.Engine()
    httpd = ThreadingServer((args.host, args.port), Handler)
    core.ok(f"Admin API listening on http://{args.host}:{args.port}  (device: {core.DEVICE_ID})")
    core.info(f"API key: {Handler.api_key}")
    if args.autostart:
        try:
            engine.start_tracking()
        except Exception as e:
            core.err(f"Autostart failed: {e}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        engine.shutdown()
        httpd.server_close()
        core.info("Server stopped.")


if __name__ == "__main__":
    main()

