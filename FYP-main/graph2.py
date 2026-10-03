import time
import psutil
import matplotlib.pyplot as plt
import numpy as np
from graph import object_detection_simulation
from graph import crowd_counting_simulation


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
    memory_usage = (memory_usage_end - memory_usage_start) / (1024 ** 2)  # Convert to MB

    return response_time, cpu_usage, memory_usage

# Dummy images for testing
image_sizes = [(480, 640), (720, 1280), (1080, 1920)]
dummy_images = [np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8) for h, w in image_sizes]

# Performance results
results = {
    "object_detection": {"response_time": [], "cpu_usage": [], "memory_usage": []},
    "crowd_counting": {"response_time": [], "cpu_usage": [], "memory_usage": []}
}

# Measure performance for object detection
for image in dummy_images:
    response_time, cpu_usage, memory_usage = measure_performance(object_detection_simulation, image.copy())
    results["object_detection"]["response_time"].append(response_time)
    results["object_detection"]["cpu_usage"].append(cpu_usage)
    results["object_detection"]["memory_usage"].append(memory_usage)

# Measure performance for crowd counting
for image in dummy_images:
    response_time, cpu_usage, memory_usage = measure_performance(crowd_counting_simulation, image.copy())
    results["crowd_counting"]["response_time"].append(response_time)
    results["crowd_counting"]["cpu_usage"].append(cpu_usage)
    results["crowd_counting"]["memory_usage"].append(memory_usage)

# Plotting the results
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
titles = ["Response Time (s)", "CPU Usage (%)", "Memory Usage (MB)"]
tasks = ["object_detection", "crowd_counting"]
y_labels = ["Time (s)", "CPU (%)", "Memory (MB)"]
colors = ["blue", "green"]

for i, task in enumerate(tasks):
    for j, metric in enumerate(["response_time", "cpu_usage", "memory_usage"]):
        axs[j, i].plot(image_sizes, results[task][metric], marker='o', color=colors[i])
        axs[j, i].set_title(f"{task.replace('_', ' ').title()} - {titles[j]}")
        axs[j, i].set_xlabel("Image Size")
        axs[j, i].set_ylabel(y_labels[j])

plt.tight_layout()
plt.show()
