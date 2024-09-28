from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
import io
import numpy as np
from PIL import Image, ImageDraw

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

def draw_keypoints(image, keypoints):
    draw = ImageDraw.Draw(image)
    for keypoint in keypoints:
        for name, (x, y) in keypoint.items():
            draw.ellipse((x-3, y-3, x+3, y+3), fill=(255, 0, 0), outline=(255, 0, 0))
            draw.text((x+5, y-5), name, fill=(255, 255, 255))
    return image


@app.post("/detect-pose-image")
async def detect_pose_image(file: UploadFile = File(...)):
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

    image = draw_keypoints(image, keypoints)

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/jpeg")