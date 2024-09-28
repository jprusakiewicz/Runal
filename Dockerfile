# Stage 1: Base stage with ultralytics image
FROM ultralytics/ultralytics:latest-python as base

# Set work directory
WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt


FROM base as fastapi

ARG YOLO_MODEL_URL
ARG OPENAI_API_KEY
ENV YOLO_MODEL_URL=${YOLO_MODEL_URL}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

COPY main.py /app/main.py
COPY streamlit_app.py /app/streamlit_app.py
COPY start.sh /app/start.sh

RUN wget -O /app/yolo-model.pt $YOLO_MODEL_URL

# Expose the port FastAPI will run on
EXPOSE 8000 8501
RUN chmod +x start.sh

# Command to run FastAPI using Uvicorn
CMD ["/app/start.sh"]