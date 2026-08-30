# SIH 2026: Adaptive Path Planner & Collision Avoidance System

![SIH 2026](https://img.shields.io/badge/SIH-2026-orange.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)

An end-to-end, vision-based autonomous navigation system engineered for unstructured, unmarked, and hazardous road environments. Designed and developed for **Smart India Hackathon 2026 (Problem Statement ID: PS26037)**.

---

## 📌 Project Overview

Traditional autonomous navigation relies heavily on clear lane markings, structured urban environments, or costly multi-LiDAR setups. On unstructured roads, unpredictable hazards such as potholes, debris, unorganized traffic, and missing boundaries cause traditional navigation systems to fail.

This project addresses this challenge by fusing **deep learning perception (YOLOv8)** directly with **dynamic 2D occupancy spatial mapping** and a **custom $A^*$ heuristic pathfinding algorithm**. The system converts raw monocular video streams into real-time trajectory plans with a sub-15ms planning latency, streaming live telemetry to an interactive Streamlit dashboard.

---

## ✨ Key Features

* **Real-Time Perception Engine:** Leverages `yolov8n` to detect static and dynamic hazards (potholes, vehicles, pedestrians).
* **Dynamic Occupancy Grid Mapping:** Converts bounding box coordinates into real-time 2D spatial obstacle matrix representations.
* **$A^*$ Trajectory Planner:** Calculates collision-free trajectories incorporating Time-To-Collision (TTC) safety buffers and target speeds.
* **Dual-Panel Telemetry Dashboard:** Renders live camera feeds alongside mapped occupancy grids, displaying real-time speed, steering angles, system latency, and hazard alerts.
* **Multi-Input Support:** Seamlessly handles synthetic simulation streams, uploaded video files, and live local webcam feeds.

---

## 🛠️ Tech Stack & Architecture

| Component | Technologies & Libraries |
| :--- | :--- |
| **Frontend / UI** | Streamlit, HTML5, JavaScript (`app.py`, `frontend/`) |
| **Backend Framework** | FastAPI, Uvicorn (`backend/main.py`) |
| **Computer Vision** | Ultralytics YOLOv8 (`yolov8n.pt`), OpenCV |
| **Path Planning & Math** | Custom $A^*$ Algorithm, NumPy, SciPy |
| **Database & Logging** | SQLite (`backend/events.db`), Python Logging (`backend/logger.py`) |

---

## 📂 Repository Structure

```text
sih-path-planner/
├── app.py                   # Main Streamlit dashboard interface
├── requirements.txt         # Dependency configuration file
├── yolov8n.pt               # Pre-trained YOLOv8 object detection weights
├── backend/
│   ├── main.py              # FastAPI server & route orchestration
│   ├── perception.py        # Video ingestion & YOLOv8 perception pipeline
│   ├── occupancy_grid.py    # Spatial obstacle grid matrix generator
│   ├── path_planner.py      # Custom A* heuristic trajectory calculation
│   ├── logger.py            # SQLite event logging & telemetry tracker
│   └── events.db            # Local database tracking vehicle telemetry
└── frontend/
    ├── index.html           # Embedded UI templates
    └── app.js               # Frontend interaction scripts