import streamlit as st
import requests
import json
import os
openai_api_key = os.getenv('OPENAI_API_KEY')

st.title("Pose Detection and OpenAI GPT-4 API App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    

    pose_url = 'http://localhost:8000/detect-pose/'
    files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
    headers = {'accept': 'application/json'}

    st.write("Sending image for pose detection...")
    pose_response = requests.post(pose_url, headers=headers, files=files)

    if pose_response.status_code == 200:
        pose_detection_result = pose_response.json()
        st.write("Pose detection successful!")
        st.json(pose_detection_result) 

        prompt = f"Act as a professional running trainer and evaluate running posture (look at key aspects of running biomechanics, such as alignment, symmetry, joint angles, and overall body mechanics) based on the YoloV8 Pose Keypoints:  {json.dumps(pose_detection_result)}. Breakdown the keypoints and provide improvements that could be suggested. Make sure that provided keypoints suggest running posture if not answer It is not running image"

        openai_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            'Authorization': f'Bearer {openai_api_key}',
            'Content-Type': 'application/json'
        }

        openai_payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are an AI that analyzes poses based on image data."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300,
            "temperature": 0.7
        }

        st.write("Sending prompt to GPT-4 for a response...")
        openai_response = requests.post(openai_url, headers=headers, json=openai_payload)

        if openai_response.status_code == 200:
            gpt4_result = openai_response.json()['choices'][0]['message']['content']
            st.write("GPT-4's response:")
            st.write(gpt4_result)
        else:
            st.write("Failed to get response from GPT-4. Error:", openai_response.status_code)
    else:
        st.write("Failed to detect pose. Error:", pose_response.status_code)
