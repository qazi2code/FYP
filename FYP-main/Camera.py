import numpy as np
import cv2
import subprocess
from ultralytics import YOLO
import streamlit as st

# Open the file containing COCO class names
with open("utils/coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Load a pretrained YOLOv8 model
model = YOLO("weights/yolov8x.pt")

# Public RTSP URL
rtsp_url = "rtsp://184.72.239.149/vod/mp4:BigBuckBunny_175k.mov"  # Public RTSP stream URL

# Full path to FFmpeg executable
ffmpeg_path = r'E:\ffmpeg-7.0.1\ffmpeg-2024-06-03-git-77ad449911-full_build\bin\ffmpeg.exe'  # Update this path as necessary

# FFmpeg command to pipe the stream to OpenCV
ffmpeg_command = [
    ffmpeg_path,
    '-i', rtsp_url,
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-r', '30',  # frames per second
    '-s', '640x480',  # size of the frame
    '-an',  # disable audio
    '-'
]

# Streamlit setup
st.title("Live Person Detection from RTSP Stream")

# Start the FFmpeg process
process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, bufsize=10**8)

# Create a placeholder in Streamlit to display frames
frame_placeholder = st.empty()

while True:
    # Read frame-by-frame from FFmpeg stdout
    raw_frame = process.stdout.read(640 * 480 * 3)
    
    if len(raw_frame) != 640 * 480 * 3:
        st.error("Can't receive frame (stream end?). Exiting...")
        break

    # Convert raw frame to numpy array
    frame = np.frombuffer(raw_frame, np.uint8).reshape((480, 640, 3))

    # Predict on the current frame
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    # Counter for persons in the current frame
    frame_person_count = 0

    if len(detect_params[0]) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]  # Get one box
            clsID = box.cls.numpy()[0]

            # Check if the detected object is a person
            if class_list[int(clsID)] == 'person':
                frame_person_count += 1

                # Draw rectangle around the person
                bb = box.xyxy.numpy()[0]
                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    (0, 255, 0),
                    3,
                )

    # Display person count on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame,
        f"Person Count: {frame_person_count}",
        (10, 30),
        font,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Display the frame in Streamlit
    frame_placeholder.image(frame, channels="BGR")

    # Break loop if the Streamlit app is closed
    if st.button("Stop"):
        break

# Release the capture and close Streamlit
process.stdout.close()
st.write("Stream ended")
