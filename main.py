from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import io
import numpy as np
from PIL import Image

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

app = FastAPI()
model = YOLO('yolo-model.pt')

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI with Ultralytics!"}

@app.post("/detect-pose/")
async def detect_pose(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img = np.array(image)
    results = model.predict(img)

    keypoints_data = []
    for result in results:
        if result.keypoints:
            for keypoint in result.keypoints:
                keypoints = []
                for item in keypoint.xy.tolist():
                    if len(item) == len(COCO_KEYPOINTS):
                        mapped_keypoints = {COCO_KEYPOINTS[i]: item[i] for i in range(len(COCO_KEYPOINTS))}
                        keypoints.append(mapped_keypoints)
                    else:
                        raise ValueError(f"Invalid number of keypoints, len={len(item)}, expected={len(COCO_KEYPOINTS)} \n {item}")
                keypoints_data.append(keypoints)

    return JSONResponse(content={"keypoints": keypoints_data})