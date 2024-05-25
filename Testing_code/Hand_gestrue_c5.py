import os
import cv2
from ultralytics import YOLO
from moviepy.editor import VideoFileClip
import numpy as np

# Correct path to the model weights
model_path = r"C:\Users\ruban\Documents\company_project\Project_3\trained_mod\model_c5_2\model_c5_2\detect\train\weights\best.pt"

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model weights not found at {model_path}")

# Load the YOLO model
model = YOLO(model_path)

# Open the webcam
cap = cv2.VideoCapture(0)

# Open the video file
video_path = r"C:\Users\ruban\Documents\company_project\Project_3\trained_mod\video_7.mp4"
video = VideoFileClip(video_path)

# Initialize variables
is_video_playing = False
playback_speed = 1.0  # Normal speed
video_duration = video.duration  # Duration of the video in seconds
video_frame_rate = video.fps  # Frame rate of the video
current_video_time = 0  # Current playback time

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam. Exiting...")
        break

    # Make detections
    results = model(frame)

    # Render the detections
    annotated_frame = results[0].plot()

    # Get the predicted class
    predicted_class = int(results[0].boxes.cls[0]) + 1 if len(results[0].boxes) > 0 else 0

    if predicted_class == 1 and not is_video_playing:
        is_video_playing = True
    elif predicted_class == 5 and is_video_playing:
        is_video_playing = False
    elif predicted_class == 2 and is_video_playing:
        playback_speed = 4.0  # Double speed
    elif predicted_class == 4 and is_video_playing:
        playback_speed = 1.0  # Normal speed
    elif predicted_class == 3 and not is_video_playing:
        is_video_playing = True
        current_video_time = 0  # Restart the video

    if is_video_playing:
        # Calculate the current frame to display based on playback speed
        current_video_time += playback_speed / video_frame_rate
        if current_video_time > video_duration:
            print("End of video reached. Stopping playback.")
            is_video_playing = False
        else:
            # Get the current video frame as an image
            video_frame = video.get_frame(current_video_time)
            # Convert the frame to an OpenCV format (BGR)
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Video Player", video_frame)

    cv2.imshow("Hand Gesture Detection", annotated_frame)

    # Check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()
