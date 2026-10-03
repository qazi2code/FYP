import numpy as np
import cv2
from ultralytics import YOLO

# opening the file in read mode
my_file = open("utils/coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# load a pretrained YOLOv5s model
model = YOLO("weights/yolov8x.pt") 

# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture("inference/videos/cctv.MP4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()

    # Counter to store person counts for this frame
    frame_person_count = 0

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]

            # Check if the detected object is a person
            if class_list[int(clsID)] == 'person':
                frame_person_count += 1

                # Draw rectangle around person
                bb = box.xyxy.numpy()[0]
                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    (0,255,0),
                    3,
                )

    # Display person count for this frame
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

    # Display the resulting frame
    cv2.imshow('PersonDetection', frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
