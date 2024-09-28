# Stage 1: Base stage with ultralytics image
FROM ultralytics/ultralytics:latest-python as base

# Set work directory
WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

# Stage 2: Create FastAPI stage
FROM base as fastapi

# Copy the FastAPI app
COPY main.py /app/main.py
COPY streamlit_app.py /app/streamlit_app.py
COPY start.sh /app/start.sh

# Expose the port FastAPI will run on
EXPOSE 8000 8501
RUN chmod +x start.sh

# Command to run FastAPI using Uvicorn
CMD ["/app/start.sh"]