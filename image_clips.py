import cv2
import numpy as np

# Load the video
video_path = 'squats.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize variables
prev_frame = None
threshold = 10000000
prev_scene_time = 0
min_scene_duration = 2000
image_count = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # If this is the first frame, just store it and continue
    if prev_frame is None:
        prev_frame = frame
        continue

    # Compute the absolute difference between the current frame and the previous frame
    diff = cv2.absdiff(frame, prev_frame)
    diff_sum = np.sum(diff)

    # If the difference exceeds the threshold, consider it a scene change
    if diff_sum > threshold:
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        if current_time - prev_scene_time > min_scene_duration:
            # Save the current frame as an image
            image_name = f"scene_change_{image_count}.jpg"
            cv2.imwrite(image_name, frame)
            image_count += 1
            prev_scene_time = current_time

    # Update the previous frame
    prev_frame = frame

cap.release()
