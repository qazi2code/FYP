import cv2
import numpy as np
from ultralytics import YOLO
import os

# Path to the file storing the counter value
counter_file_path = "counter.txt"

# Read the counter value from the file
if os.path.exists(counter_file_path):
    with open(counter_file_path, "r") as file:
        counter = int(file.read().strip())
else:
    counter = 0  # Initialize counter if file does not exist

# Increment the counter
counter += 1

# Save the updated counter value back to the file
with open(counter_file_path, "w") as file:
    file.write(str(counter))

# Load a pretrained YOLOv8n model
model = YOLO("yolov8x.pt", "v8")

# Predict on an image
detection_output = model.predict(source="inference/images/cctv.png", conf=0.25, save=True)

# Class names (assuming COCO dataset, you can modify it based on the dataset your model is trained on)
class_names = model.names

# Generate a color map for the classes
np.random.seed(42)  # Seed for reproducibility
colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype='uint8')

# Count the number of persons
person_count = 0

# Load the image
image = cv2.imread("inference/images/cctv.png")
font = cv2.FONT_HERSHEY_SIMPLEX

# Draw bounding boxes and class names for persons only
for box in detection_output[0].boxes:
    clsID = int(box.cls.numpy()[0])
    class_name = class_names[clsID]
    
    if class_name == "person":
        person_count += 1
        color = [int(c) for c in colors[clsID]]
        x1, y1, x2, y2 = map(int, box.xyxy.numpy()[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)  # Increase thickness to 3
        cv2.putText(image, class_name, (x1, y1 - 10), font, 0.7, color, 2)  # Increase font scale to 0.7 and thickness to 2

# Display total person count on the image
cv2.putText(image, f"Total count of persons: {person_count}", (10, 30), font, 1, (255, 255, 255), 2)

# Save the image with a unique name
output_path = f"runs/detect/class_with_boxes_and_labels_{counter}.jpg"
cv2.imwrite(output_path, image)

# Display the image with resized window
cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Detection", 800, 600)  # Adjust the size as needed
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Saved image with detections to {output_path}")
print(f"Total count of persons: {person_count}")
