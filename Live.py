import random
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

# Opening the file in read mode
with open("utils/coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Generate random colors for class list
detection_colors = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _ in class_list
]

# Load a pretrained YOLOv8 model (local weights)
model = YOLO("weights/yolov8x.pt")

# Vals to resize video frames | small frame optimize the run
frame_wid = 640
frame_hyt = 480

font = cv2.FONT_HERSHEY_COMPLEX


class TkFrameViewer:
    """Show frames without cv2.imshow (works when opencv-python-headless is installed)."""

    def __init__(self, title: str) -> None:
        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root)
        self.label.pack()
        self._photo = None
        self.running = True
        self.root.bind("<q>", lambda _event: self.stop())
        self.root.bind("<Q>", lambda _event: self.stop())
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

    def show(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(image=image)
        self.label.configure(image=self._photo)
        self.root.update_idletasks()
        self.root.update()

    def stop(self) -> None:
        self.running = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def is_running(self) -> bool:
        try:
            return self.running and self.root.winfo_exists()
        except tk.TclError:
            return False


# Open the video capture (0 for default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    raise SystemExit(1)

viewer = TkFrameViewer("Object Detection — press Q or close window to quit")
print("Live detection started. Press Q in the video window or close it to exit.")

while viewer.is_running():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    frame = cv2.resize(frame, (frame_wid, frame_hyt))
    detect_params = model.predict(source=[frame], conf=0.45, save=False, verbose=False)

    frame_object_counts = {}

    for box in detect_params[0].boxes:
        clsID = int(box.cls.numpy()[0])
        conf = float(box.conf.numpy()[0])
        bb = box.xyxy.numpy()[0]
        class_name = class_list[clsID]

        frame_object_counts[class_name] = frame_object_counts.get(class_name, 0) + 1

        cv2.rectangle(
            frame,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            detection_colors[clsID],
            3,
        )
        cv2.putText(
            frame,
            f"{class_name} {round(conf * 100, 1)}%",
            (int(bb[0]), int(bb[1]) - 10),
            font,
            0.7,
            (255, 255, 255),
            2,
        )

    y_offset = 30
    for obj, count in frame_object_counts.items():
        cv2.putText(
            frame,
            f"{obj}: {count}",
            (10, y_offset),
            font,
            0.5,
            (255, 255, 255),
            2,
        )
        y_offset += 20

    viewer.show(frame)

cap.release()
print("Camera released.")
