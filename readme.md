```export $(grep -v '^#' .env | xargs)```
```docker build --build-arg YOLO_MODEL_URL=$YOLO_MODEL_URL --build-arg OPENAI_API_KEY=$OPENAI_API_KEY -t fastapi-ultralytics .```
```docker run -p 8000:8000 -p 8501:8501 fastapi-ultralytics```