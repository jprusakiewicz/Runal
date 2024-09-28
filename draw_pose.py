import json
from PIL import Image, ImageDraw

# JSON data
json_data = '''
{
  "status": "OK",
  "predictions": [
    {
      "nose": {
        "coordinate_x": 527.64111328125,
        "coordinate_y": 78.78046417236328,
        "score": 1.0
      },
      "rightEye": {
        "coordinate_x": 516.3277587890625,
        "coordinate_y": 71.40284729003906,
        "score": 1.0
        },
      "rightEar": {
        "coordinate_x": 483.5880432128906,
        "coordinate_y": 80.02650451660156,
        "score": 1.0
      },
      "leftShoulder": {
        "coordinate_x": 488.0511169433594,
        "coordinate_y": 158.44322204589844,
        "score": 1.0
      },
      "rightShoulder": {
        "coordinate_x": 445.2245178222656,
        "coordinate_y": 136.1300048828125,
        "score": 1.0
      },
      "leftElbow": {
        "coordinate_x": 509.8282165527344,
        "coordinate_y": 242.88140869140625,
        "score": 1.0
      },
      "rightElbow": {
        "coordinate_x": 376.3570861816406,
        "coordinate_y": 205.81297302246094,
        "score": 1.0
      },
      "leftWrist": {
        "coordinate_x": 569.7409057617188,
        "coordinate_y": 242.4142608642578,
        "score": 1.0
      },
      "rightWrist": {
        "coordinate_x": 394.4449768066406,
        "coordinate_y": 273.1732482910156,
        "score": 1.0
      },
      "leftHip": {
        "coordinate_x": 444.3392639160156,
        "coordinate_y": 326.80682373046875,
        "score": 1.0
      },
      "rightHip": {
        "coordinate_x": 444.47021484375,
        "coordinate_y": 323.53619384765625,
        "score": 1.0
      },
      "leftKnee": {
        "coordinate_x": 371.5823059082031,
        "coordinate_y": 441.90576171875,
        "score": 1.0
      },
      "rightKnee": {
        "coordinate_x": 541.8119506835938,
        "coordinate_y": 459.2372131347656,
        "score": 1.0
      },
      "leftAnkle": {
        "coordinate_x": 261.3267517089844,
        "coordinate_y": 536.6879272460938,
        "score": 1.0
      },
      "rightAnkle": {
        "coordinate_x": 550.4119262695312,
        "coordinate_y": 574.0437622070312,
        "score": 1.0
      }
    }
  ]
}
'''

# Parse JSON data
data = json.loads(json_data)
prediction = data['predictions'][0]

# Extract all keypoints without thresholding
keypoints_raw = {}
for keypoint_name, keypoint_data in prediction.items():
    x = keypoint_data['coordinate_x']
    y = keypoint_data['coordinate_y']
    keypoints_raw[keypoint_name] = (x, y)

# Find min and max coordinates for scaling
all_x = [coord[0] for coord in keypoints_raw.values()]
all_y = [coord[1] for coord in keypoints_raw.values()]
min_x, max_x = min(all_x), max(all_x)
min_y, max_y = min(all_y), max(all_y)

# Set image size and padding
img_width, img_height = 800, 800
padding = 50  # Padding around the pose

# Compute scaling factors
scale_x = (img_width - 2 * padding) / (max_x - min_x) if max_x != min_x else 1
scale_y = (img_height - 2 * padding) / (max_y - min_y) if max_y != min_y else 1
scale = min(scale_x, scale_y)

# Map coordinates to image space
keypoints = {}
for keypoint_name, (x_raw, y_raw) in keypoints_raw.items():
    x = int((x_raw - min_x) * scale + padding)
    y = int((y_raw - min_y) * scale + padding)
    keypoints[keypoint_name] = (x, y)

# Define skeleton connections
skeleton = [
    ("nose", "leftEye"),
    ("leftEye", "leftEar"),
    ("nose", "rightEye"),
    ("rightEye", "rightEar"),
    ("nose", "leftShoulder"),
    ("nose", "rightShoulder"),
    ("leftShoulder", "leftElbow"),
    ("leftElbow", "leftWrist"),
    ("rightShoulder", "rightElbow"),
    ("rightElbow", "rightWrist"),
    ("leftShoulder", "leftHip"),
    ("rightShoulder", "rightHip"),
    ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"),
    ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle"),
    ("leftHip", "rightHip"),
]

# Create a white image
img = Image.new('RGB', (img_width, img_height), 'white')
draw = ImageDraw.Draw(img)

# Draw skeleton lines
for kp1, kp2 in skeleton:
    if kp1 in keypoints and kp2 in keypoints:
        x1, y1 = keypoints[kp1]
        x2, y2 = keypoints[kp2]
        draw.line([(x1, y1), (x2, y2)], fill='black', width=2)

# Draw keypoints as circles
for x, y in keypoints.values():
    r = 5  # Radius of the keypoint circle
    draw.ellipse((x - r, y - r, x + r, y + r), fill='red')

# Save the image
img.save('pose_image.png')
