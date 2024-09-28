#!/bin/bash

# Start FastAPI (Uvicorn) in the background
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in the foreground
streamlit run streamlit_app.py --server.enableCORS=false --server.enableXsrfProtection=false
