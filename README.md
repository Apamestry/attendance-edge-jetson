# 📸 Face Recognition Attendance System

An automated, real-time face recognition attendance system built with Python, OpenCV, and Deep Learning face encodings. Featuring an interactive CLI menu, dynamic webcam registration, real-time visual HUD overlays, and automated CSV logging.

---

## ✨ Features

* ⚡ **Interactive CLI Menu:** Seamlessly navigate between tracking, member management, and logging.
* 📸 **Dynamic Member Registration:** Register new users straight from your webcam—with automatic face validation before saving.
* 🎯 **Real-Time Recognition & HUD:** 
  * Live bounding boxes (Green for recognized members, Red for unknown faces).
  * Prominent **3-second on-screen notification banner** (`✓ ATTENDANCE MARKED: [NAME]`) upon successful registration.
* 📁 **Dynamic Face Encodings:** Automatically scans the `photos/` directory on startup. No hardcoded names or manual array updates required.
* 🗑️ **Member Management:** View registered members or remove a user directly from the menu.
* 📊 **Automated CSV Logging:** Logs name, date, and exact timestamp into `attendance.csv`. Automatically avoids logging duplicates for the same person on the same day.
* 🛡️ **Clean Resource Cleanup:** Intercepts shutdown signals (`q` key or `Ctrl + C`) to gracefully release webcam streams and close OpenCV windows without crashing.

---

## 🛠️ Tech Stack

* **Language:** Python 3.12+ / Python 3.14
* **Computer Vision:** [OpenCV](https://opencv.org/) (`opencv-python`)
* **Face Recognition Engine:** [`face_recognition`](https://github.com/ageitgey/face_recognition) (dlib backend)
* **Data Manipulation:** NumPy
* **Storage:** CSV (`attendance.csv`) & Image files (`photos/`)

---

## 📁 Project Structure

```text
face-recognition-attendance-system/
├── photos/             # Directory storing registered member image files (.jpg, .png)
├── attendance.csv      # Auto-generated CSV file storing attendance logs
├── main.py             # Core entry point (CLI menu & recognition pipeline)
├── .gitignore          # Prevents venv, models, and cache files from being tracked
└── README.md           # Project documentation
