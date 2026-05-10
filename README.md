# Automated University Gate ALPR System

A smart gate mockup system designed to streamline vehicle entry for IPB University students. This project replaces traditional RFID tapping with an Automated License Plate Recognition (ALPR) system, integrating a machine learning backend with a web-based dashboard.

## Features

- **((ONPROGRESS))** **Live Webcam Scanning:** Uses frontend frame-polling to process continuous video streams without network bottlenecks.
- **Video Upload Processing:** Upload short .mp4 mockups for instant plate extraction.
- **Advanced Machine Learning Pipeline:**
  - **YOLOv11: License-Plate-Detection** Custom-trained object detection to instantly locate and crop license plates.
  - **fast-plate-ocr:** High-accuracy character recognition fine-tuned for 8 Indonesian license plates.
- **Voting Mechanism:** Aggregates OCR results across multiple frames to eliminate motion blur errors and ensure high confidence.
- **Containerized Backend:** Fully Dockerized FastAPI machine learning service to prevent dependency conflicts and bypass OS file-locking issues.

## Architecture

This project utilizes a microservice architecture to separate the heavy machine learning processing from the user interface.

1. **Frontend (Laravel):** The user dashboard where students register their vehicles and the gate mockup interface operates.
2. **Backend API (FastAPI):** A lightweight Python service dedicated to running YOLO and OCR inference.

## Getting Started

### Prerequisites

- Docker (Recommended) or Python 3.10+
- PHP & Composer (for the Laravel frontend)
- Webcam (for live scanning mockup)

### Option 1: Running with Docker (Recommended)

This is the easiest way to run the machine learning API without setting up Python virtual environments.

1. Build the Docker image:
   ```bash
   docker build -t ipb-gate-api
   ```
2. Run the container:
   ```bash
   docker run -d -p 8001:8001 --name gate-api ipb-gate-api
   ```
   (The API is now running at http://localhost:8001)

### Option 2: Running Locally (Without Docker)

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows use: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the FastAPI server:
   ```bash
   uvicorn api:app --reload --port 8001
   ```

### Starting the Dashboard Website

The Dashboard Website could be accessed at https://github.com/rxmxdhxn/IPB-Smart-Gate-System

Open a separate terminal window and run:

```bash
php artisan serve
```

(The web dashboard will be available at http://localhost:8000)

## API Endpoints

- POST /predict: Accepts a .mp4 video upload, processes frames, applies the voting mechanism, and returns the highest confidence license plate string.
- **((ONPROGRESS))** POST /predict_frame: Accepts a single .jpg image blob (used for the live webcam polling loop) for instant inference.

## Built With

- FastAPI - Python API framework
- Ultralytics YOLOv11 - Object detection
- fast-plate-ocr - Optical Character Recognition
- OpenCV - Image processing
- Docker - Containerization
- Laravel - Web framework

## License & Compliance

This project utilizes models governed by the **GNU AGPLv3 License**. Due to license inheritance from the YOLOv11 base model, this overall system must also comply with AGPLv3.

**Compliance Reminder:**
If you use this model or system in a service or project, you must open-source the code that uses it. For full details, please refer to the GNU AGPLv3 License.

## Acknowledgments & Attributions

This project is made possible by the incredible work of the open-source community:

- **MorseTechLab:** For the fine-tuned [YOLOv11 License Plate Detection Model](https://huggingface.co/morsetechlab/yolov11-license-plate-detection).
- **Ultralytics:** For the base YOLOv11 architecture (AGPLv3).
- **Roboflow Universe:** For the dataset provisioning (CC BY 4.0).
- **ankandrew:** For the highly efficient [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) library.
