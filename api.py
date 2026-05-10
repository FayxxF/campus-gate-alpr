import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fast_plate_ocr import LicensePlateRecognizer
from collections import Counter

app = FastAPI(title="IPB Automated Gate OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


yolo_model = YOLO("license-plate-finetune-v1n.pt")
ocr_model = LicensePlateRecognizer(
    onnx_model_path="best.onnx",
    plate_config_path="plate_config.yaml"
)

@app.post("/predict")
async def predict_license_plate(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        contents = await file.read()
        temp_video.write(contents)
        video_path = temp_video.name
    try:
        cap = cv2.VideoCapture(video_path)
        
        # for local video file
        # video_path = "dummy_plate.mp4"
        # cap = cv2.VideoCapture(video_path)

        plate_votes = []
        frame_count = 0
        # save every 5th frame
        frame_skip = 5

        print("Processing video...")

        while cap.isOpened():
            ret, frame = cap.read()

            # ret = false, video ends
            if not ret:
                break

            frame_count += 1

            # if not every 5 frames, skip.
            if frame_count % frame_skip != 0:
                continue 
            
            results = yolo_model(frame, verbose=False)
            # loop
            for result in results:
                boxes = result.boxes

                for box in boxes:
                    confidence = float(box.conf[0])

                    # crop only if conf is high
                    if confidence > 0.6:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cropped_plate = frame[y1:y2, x1:x2]

                        # validate
                        if cropped_plate.size == 0 :
                            continue
                        
                        # run ocr
                        print("Run Ocr")
                        ocr_result = ocr_model.run(cropped_plate, return_confidence=True)
                        
                        if ocr_result and len(ocr_result) > 0:
                            extracted_text = ocr_result[0].plate
                            plate_votes.append(extracted_text)
                            
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()

        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                print(f"Could not delete temp file: {e}")

    print("Votes collected:", plate_votes)
    
    if not plate_votes:
        return {
            "success": False, 
            "message": "No license plates detected in the video."
        }

    vote_tallies = Counter(plate_votes)
    winning_plate = vote_tallies.most_common(1)[0][0]
    winning_votes = vote_tallies[winning_plate]
    total_votes = len(plate_votes)

    return {
        "success": True,
        "final_plate": winning_plate,
        "voting_stats": {
            "total_frames_analyzed": total_votes,
            "votes_for_winner": winning_votes,
            "confidence_ratio": round(winning_votes / total_votes, 2),
            "all_votes": dict(vote_tallies) 
        }
    }


@app.post("/predict-frame")
async def predict_frame(file: UploadFile = File(...)):
    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {
            "success": False,
            "message": "Invalid image/frame."
        }

    plate_votes = []

    results = yolo_model(frame, verbose=False)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            confidence = float(box.conf[0])

            if confidence > 0.6:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_plate = frame[y1:y2, x1:x2]

                if cropped_plate.size == 0:
                    continue

                ocr_result = ocr_model.run(cropped_plate, return_confidence=True)

                if ocr_result and len(ocr_result) > 0:
                    extracted_text = ocr_result[0].plate
                    plate_votes.append(extracted_text)

    if not plate_votes:
        return {
            "success": False,
            "message": "No license plate detected."
        }

    vote_tallies = Counter(plate_votes)
    winning_plate = vote_tallies.most_common(1)[0][0]

    return {
        "success": True,
        "final_plate": winning_plate,
        "all_votes": dict(vote_tallies)
    }