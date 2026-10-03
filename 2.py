import cv2
from ultralytics import YOLO
import numpy as np

# load a pretrained YOLOv8n model
model = YOLO("yolov8l.pt", "v8")  

# predict on an image
detection_output = model.predict(source="inference/images/bird.jpeg", conf=0.25, save=False) 

# Get the image with the bounding boxes and labels
img = cv2.imread("inference/images/bird.jpeg")

# Extract the number of objects and their labels
num_objects = len(detection_output.pred[0].xyxy)
objects = [obj.names[0] for obj in detection_output.pred[0]]

# Draw bounding boxes and labels on the image
for i in range(num_objects):
    box = detection_output.pred[0].xyxy[i]
    label = objects[i]
    xy = (int(box[0]), int(box[1]))
    xy2 = (int(box[2]), int(box[3]))
    cv2.rectangle(img, xy, xy2, (0, 255, 0), 2)
    cv2.putText(img, label, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the number of objects and their counts on the screen
for obj, count in zip(set(objects), np.bincount(objects)):
    print(f"{obj}: {count}")

# Display the image on the screen
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
