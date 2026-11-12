# 📸 Face Recognition Attendance System (Jetson Nano + Admin Web UI)

An automated, real-time face recognition attendance system built with Python, OpenCV, and `face_recognition` (dlib), optimized for the **NVIDIA Jetson Nano** with both an interactive CLI and a headless HTTP/SSE server for an online Admin Web UI.

---

## ✨ Features

* ⚡ **Jetson Nano Performance Optimizations:**
  * **Threaded Camera Reader (`Camera`):** Drains the camera buffer in a background thread (`CAP_PROP_BUFFERSIZE = 1`) so recognition always operates on the newest frame with zero USB/CSI lag.
  * **Detect → Track → Re-verify (`FacePipeline`):** Runs the heavy face detector only every `detect_every` cycles (default `3`), tracks faces across intermediate frames via cheap 68-point landmarks, and only runs the 128-D face encoder on new faces or every `reverify_seconds` (default `1.0s`).
  * **Single Landmark Pass + Reliable Blink Liveness:** Reuses a single landmark extraction per cycle for both box tracking and Eye Aspect Ratio (EAR) blink detection.
  * **Zero Extra Dependencies:** Dropped `scipy` in favor of vectorized `numpy` math; the web server uses Python's standard library (`http.server` + `ThreadingMixIn`), compatible with **Python 3.6+** (JetPack 4.x).
* 🌐 **Headless Web API & Online Admin Dashboard (`server.py` + `web/index.html`):**
  * Start/stop the tracker or trigger remote camera registration from any browser.
  * Live **Server-Sent Events (`/api/events`)** for attendance marks, status heartbeats, member changes, registration progress, and throttled unknown-face snapshots.
  * On-demand **MJPEG live preview (`/api/stream`)** with HUD overlays (zero encoding cost when nobody is watching).
  * Upload member photos directly from the browser (auto-resized to `<= 800px` client-side) or capture poses using the Nano's camera remotely.
  * View, filter, export, manually mark, or remove attendance records, and tune recognition settings live.
* 🖥️ **Original Interactive CLI (`main.py`):** Preserves the full terminal menu and OpenCV window HUD with clickable buttons.

---

## 📁 Project Structure

```text
attendance-edge-jetson/
├── edge_core.py        # Shared recognition pipeline, threaded camera, attendance log & engine
├── main.py             # Interactive CLI menu & OpenCV HUD window
├── server.py           # Headless HTTP + SSE + MJPEG server for the Admin Web UI
├── web/
│   └── index.html      # Standalone Admin Web UI (host anywhere or open from the Nano)
├── photos/             # Registered member photos: photos/<Name_Surname>/*.jpg
├── attendance.csv      # Auto-generated attendance log (Name, Date, Time)
├── encodings_cache.pkl # Cached 128-D face encodings
└── settings.json       # Persisted runtime settings (created when saved from the UI)
```

---

## 🚀 Quick Start

### 1. Run the Headless Server + Web UI (Recommended on Nano)

```bash
python3 server.py                 # Starts on http://0.0.0.0:8000
python3 server.py --autostart     # Also starts the attendance tracker immediately
```

On first run, `server.py` generates an API key in `.api_key` and prints it to the terminal (or set `EDGE_API_KEY=<your-key>`).

* **Local network:** Open `http://<nano-ip>:8000` in your browser and paste the API key.
* **Hosted online (e.g., GitHub Pages / Netlify / Vercel):** Host `web/index.html` anywhere, expose the Nano over HTTPS using a tunnel:
  ```bash
  cloudflared tunnel --url http://localhost:8000
  # or: ngrok http 8000
  ```
  Then paste the tunnel URL and API key into the top bar of the hosted Admin UI.

### 2. Run the Interactive CLI Menu

```bash
python3 main.py
```

---

## 🔌 Bidirectional Data Contract (`server.py`)

Every `/api/*` endpoint requires the API key via header `X-API-Key: <key>` or query parameter `?key=<key>`.

### Nano → Web UI (Events & Reads)

| Channel / Endpoint | Data Sent from Nano |
|---|---|
| `GET /api/events` (SSE) | Real-time events: `status` (every 10s + on mode change), `attendance_marked` (`name, member_id, date, time, distance, confidence, liveness, manual`), `attendance_removed`, `members_changed`, `registration_progress`, `unknown_face` (throttled base64 JPEG crop), `config_changed` |
| `GET /api/status` | `device_id, version, mode (idle/tracking/registering), error, camera_ok, fps, members, photos, present_today, today, time, uptime_s, timings_ms (detect/landmarks/encode), registration` |
| `GET /api/members` | `[{id, name, photo_count, registered_at}]` |
| `GET /api/members/<id>/photo` | Member thumbnail image (`image/jpeg` or `image/png`) |
| `GET /api/attendance` | `{rows: [{name, date, time}]}` filtered by `?date=YYYY-MM-DD` or `?from=...&to=...` |
| `GET /api/config` | `{values: {...}, rules: {...}}` |
| `GET /api/stream` | `multipart/x-mixed-replace` MJPEG live camera preview with HUD boxes |

### Web UI → Nano (Commands & Writes)

| Endpoint | Payload / Query | Action on Nano |
|---|---|---|
| `POST /api/tracker/start` | — | Loads encodings, opens camera, starts background recognition thread |
| `POST /api/tracker/stop` | — | Stops recognition thread and releases camera |
| `POST /api/reload` | — | Reloads face encodings from `photos/` without stopping the tracker |
| `POST /api/members` | `{name, photos: ["data:image/jpeg;base64,..."]}` | Validates each photo for 1 clear face, saves to `photos/<id>/`, reloads encodings |
| `DELETE /api/members/<id>` | — | Deletes member's photos (keeps past attendance), reloads encodings |
| `POST /api/register/start` | `{name}` | Opens Nano camera in `registering` mode for pose-guided capture |
| `POST /api/register/capture` | — | Validates & saves current camera frame, advances pose |
| `POST /api/register/finish` | — | Closes camera, cleans up if 0 photos saved, reloads encodings |
| `POST /api/attendance` | `{name, date?, time?}` | Manual admin attendance mark |
| `DELETE /api/attendance` | `?name=...&date=YYYY-MM-DD` | Removes an attendance entry |
| `PUT /api/config` | `{detection_model, upsample, scale_factor, detect_every, reverify_seconds, face_match_tolerance, ...}` | Validates, applies live, and saves to `settings.json` |

---

## 🛠️ Tuning for Jetson Nano

From the **Recognition settings** card in the Web UI (or `settings.json`), you can tune speed vs. distance:

* **`upsample`: `0`** — ~4× faster face detection than `1` (best when people stand within ~1.5 m of the camera).
* **`detection_model`: `"cnn"`** (uses Jetson CUDA GPU) or **`"hog"`** (CPU fallback if GPU memory is tight).
* **`scale_factor`: `0.35` – `0.5`** — smaller values shrink the frame before detection for higher FPS.
* **`detect_every`: `3` – `5`** — how many cycles to track via landmarks between full detections.
* **`reverify_seconds`: `1.0` – `2.0`** — how often to re-run the 128-D face encoder on an already-tracked face.
