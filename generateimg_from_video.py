import cv2
import tkinter as tk
from tkinter import filedialog
import os

# Allow user to select a video from their computer
root = tk.Tk()
root.withdraw()  # Hide the main window
file_path = filedialog.askopenfilename(title="Select a Video", filetypes=[("Video Files", "*.mp4;*.avi;*.mkv;*.flv;*.mov")])
if not file_path:
    print("No video selected.")
    exit()

# Create an output directory for images
output_dir = "output_frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
cap = cv2.VideoCapture(file_path)
frameRate = cap.get(5)  # Get the frame rate

frame_count = 0
while(cap.isOpened()):
    frameId = cap.get(1)  # Current frame number
    ret, frame = cap.read()
    if not ret:
        break
    if frameId % (int(frameRate)*7) == 0:  # Extract a frame every 5 seconds
        
        # Save the frame as an image
        img_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(img_filename, frame)
        
        # Display the frame for 5 seconds
        cv2.imshow("Video Frame", frame)
        if cv2.waitKey(7000) & 0xFF == ord('q'):
            break
        
        frame_count += 1

cap.release()
cv2.destroyAllWindows()
