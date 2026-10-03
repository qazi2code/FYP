import time
import psutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Object detection simulation function
def object_detection_simulation(image):
    model = YOLO("yolov8x.pt", "v8")
    detection_output = model.predict(source=image, conf=0.25, save=False)
    class_names = model.names
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype='uint8')
    font = cv2.FONT_HERSHEY_SIMPLEX
    num_objects = len(detection_output[0].boxes)

    for box in detection_output[0].boxes:
        clsID = int(box.cls.numpy()[0])
        class_name = class_names[clsID]
        color = [int(c) for c in colors[clsID]]
        x1, y1, x2, y2 = map(int, box.xyxy.numpy()[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(image, class_name, (x1, y1 - 10), font, 0.7, color, 2)

    return image, num_objects

# Crowd counting simulation function
def crowd_counting_simulation(image):
    model = YOLO("yolov8x.pt", "v8")
    detection_output = model.predict(source=image, conf=0.25, save=False)
    class_names = model.names
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype='uint8')
    font = cv2.FONT_HERSHEY_SIMPLEX
    person_count = 0

    for box in detection_output[0].boxes:
        clsID = int(box.cls.numpy()[0])
        class_name = class_names[clsID]

        if class_name == "person":
            person_count += 1
            color = [int(c) for c in colors[clsID]]
            x1, y1, x2, y2 = map(int, box.xyxy.numpy()[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            cv2.putText(image, class_name, (x1, y1 - 10), font, 0.7, color, 2)

    cv2.putText(image, f"Total count of persons: {person_count}", (10, 30), font, 1, (255, 255, 255), 2)
    return image, person_count

# Performance measurement function
def measure_performance(func, *args, **kwargs):
    start_time = time.time()
    cpu_usage_start = psutil.cpu_percent(interval=None)
    memory_usage_start = psutil.virtual_memory().used

    func(*args, **kwargs)

    cpu_usage_end = psutil.cpu_percent(interval=None)
    memory_usage_end = psutil.virtual_memory().used
    end_time = time.time()

    response_time = end_time - start_time
    cpu_usage = cpu_usage_end - cpu_usage_start
    memory_usage = (memory_usage_end - memory_usage_start) / (1024 ** 2)

    return response_time, cpu_usage, memory_usage

# Load the image
image_path = "inference/images/classroom.jpg"
image = cv2.imread(image_path)

# Measure performance for object detection
response_time_od, cpu_usage_od, memory_usage_od = measure_performance(object_detection_simulation, image.copy())

# Measure performance for crowd counting
response_time_cc, cpu_usage_cc, memory_usage_cc = measure_performance(crowd_counting_simulation, image.copy())

# Plotting the results
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
titles = ["Response Time (s)", "CPU Usage (%)", "Memory Usage (MB)"]
tasks = ["Object Detection", "Crowd Counting"]
metrics = [[response_time_od, response_time_cc], [cpu_usage_od, cpu_usage_cc], [memory_usage_od, memory_usage_cc]]
colors = ["blue", "green"]

for i, title in enumerate(titles):
    axs[i].bar(tasks, metrics[i], color=colors)
    axs[i].set_title(title)
    axs[i].set_ylabel(title.split()[0])

plt.tight_layout()
plt.show()
