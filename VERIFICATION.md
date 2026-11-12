# ✅ Verification Guide: Jetson Nano Server & Admin Web UI

Step-by-step instructions and commands to verify that `server.py`, `edge_core.py`, and `web/index.html` work properly on the Jetson Nano.

---

## Step 1: Pull the `ui-integration` Branch on the Jetson Nano

SSH into your Jetson Nano (or open a terminal on it) and checkout the branch:

```bash
# If you already cloned the original repo:
cd ~/attendance-edge-jetson
git remote add fork https://github.com/ShreyashV38/attendance-edge-jetson.git
git fetch fork
git checkout -b ui-integration fork/ui-integration

# OR if cloning fresh on the Nano:
git clone -b ui-integration https://github.com/ShreyashV38/attendance-edge-jetson.git
cd attendance-edge-jetson
```

*(Recommended)* Lock the Jetson Nano to maximum performance mode so face recognition runs at full speed:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## Step 2: Start the Server on the Nano

Run `server.py` with a custom API key (or omit `EDGE_API_KEY` to let it auto-generate one in `.api_key`):

```bash
export EDGE_API_KEY="mysecretkey"
python3 server.py
```

*(Optional)* To also start the attendance tracker automatically on boot:

```bash
python3 server.py --autostart
```

Find your Nano's local IP address:

```bash
hostname -I
```
*(Example output: `192.168.1.50`)*

---

## Step 3: CLI Sanity Check (API & Auth)

From another terminal on the Nano or from your laptop on the same network, run:

```bash
# 1. Verify authentication blocks unauthorized requests (should return 401)
curl -i http://192.168.1.50:8000/api/status

# 2. Verify status endpoint with API key (should return 200 JSON)
curl -H "X-API-Key: mysecretkey" http://192.168.1.50:8000/api/status
```

Expected response:
```json
{
  "device_id": "jetson-nano",
  "version": "1.1",
  "mode": "idle",
  "error": null,
  "camera_ok": false,
  "fps": 0.0,
  "members": 0,
  "photos": 0,
  "present_today": 0,
  "today": "2026-11-13",
  "time": "2026-11-13T00:10:00",
  "uptime_s": 12,
  "timings_ms": {},
  "registration": null
}
```

---

## Step 4: Connect the Admin Web UI

### Option A — Local Network (Same Wi-Fi / LAN)
1. Open `http://192.168.1.50:8000` in your browser.
2. In the top bar, confirm **Nano URL** is `http://192.168.1.50:8000`, type `mysecretkey` into **API key**, and click **Connect**.
3. Verify the status pill changes from `offline` (red) to `online` (green).

### Option B — Hosted Online (Accessible from Anywhere)
1. Expose the Nano's port `8000` over HTTPS using Cloudflare Tunnel (or ngrok):
   ```bash
   cloudflared tunnel --url http://localhost:8000
   # or: ngrok http 8000
   ```
2. Copy the generated HTTPS URL (e.g., `https://random-words.trycloudflare.com`).
3. Open `web/index.html` (hosted on GitHub Pages / Netlify / Vercel, or opened locally on your computer).
4. Paste the tunnel HTTPS URL into **Nano URL**, enter `mysecretkey`, and click **Connect**.

---

## Step 5: End-to-End Feature Checklist

| # | Test Case | Action in Web UI | Expected Result on Nano & UI |
|---|---|---|---|
| **1** | **Upload Member (Web UI → Nano)** | In **Members**, enter a name, select 3–5 clear face photos, and click **Upload photos**. | Nano validates each image for 1 clear face, saves to `photos/<Name>/`, and the member card + thumbnail appear in the UI. |
| **2** | **Remote Camera Registration (Web UI ⇄ Nano)** | Enter a name and click **Capture on Nano camera**. Stand in front of the Nano, click **Capture** 3–5 times following pose prompts, then click **Finish**. | Mode switches to `registering`, live preview opens, pose counter increments (`1/5` … `5/5`), and encodings reload on finish. |
| **3** | **Start Tracker (Web UI → Nano)** | Click **Start tracker** in the **Device** card. | Mode badge turns green (`tracking`), **camera** shows `on`, and **recognition FPS** + **detect / landmarks / encode ms** populate. |
| **4** | **Live Video Preview (Nano → Web UI)** | Click **Show live preview**. | Real-time annotated MJPEG stream appears with bounding boxes (`Unknown`, `Blink to verify`, or `PRESENT`). |
| **5** | **Blink Liveness & Attendance (Nano → Web UI)** | Stand in front of the Nano camera and **blink naturally**. | Box turns green (`PRESENT`), green toast pops up in the UI, **Live events** shows confidence %, and `attendance.csv` + the UI table add today's entry. |
| **6** | **Manual Mark / Remove (Web UI → Nano)** | Select a member in **Attendance** and click **Mark present (manual)**, then test **Remove**. | Entry is added/removed in `attendance.csv` and updates live across connected browsers. |
| **7** | **Export CSV** | Click **Export CSV** in the **Attendance** card. | Browser downloads `attendance_YYYY-MM-DD.csv`. |
| **8** | **Live Performance Tuning (Web UI → Nano)** | In **Recognition settings**, set `upsample` to `0` (or `detection_model` to `hog`) and click **Save**. | `settings.json` is saved on the Nano and `detect / landmarks / encode ms` drops immediately. |
| **9** | **CLI Coexistence** | Stop `server.py` (`Ctrl+C`) and run `python3 main.py`. | Original terminal menu (options 1–7) and OpenCV window HUD work normally using the optimized pipeline. |
