from ultralytics import YOLO
import cv2
model = YOLO('../Yolo-Weights/yolov8l.pt')
results = model("inference/images/img.jpg", show=True, save= True)
cv2.setNumThreads(0)  # Disable multithreading
cv2.waitKey(0)