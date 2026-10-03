import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import random
import threading
import time

# Function to load counter value
def load_counter(counter_file_path):
    if os.path.exists(counter_file_path):
        with open(counter_file_path, "r") as file:
            return int(file.read().strip())
    return 0

# Function to save counter value
def save_counter(counter_file_path, counter):
    with open(counter_file_path, "w") as file:
        file.write(str(counter))

# Load counter value
counter_file_path = "counter.txt"
counter = load_counter(counter_file_path)

# Streamlit UI
st.set_page_config(
    page_title="Object Detection and Crowd Counting",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _reset_app_on_tab_close(_session_token: str) -> None:
    """When the browser tab closes, reset persisted state and stop the server."""

    def _shutdown() -> None:
        time.sleep(0.5)
        if os.path.exists(counter_file_path):
            save_counter(counter_file_path, 0)
        try:
            from streamlit.runtime import get_instance

            runtime = get_instance()
            runtime.stop()
        except Exception:
            pass
        time.sleep(0.3)
        os._exit(0)

    threading.Thread(target=_shutdown, daemon=True).start()


@st.cache_resource(on_release=_reset_app_on_tab_close, scope="session")
def _register_tab_close_handler() -> str:
    return "active"


_register_tab_close_handler()

# Sidebar for selection and file upload
st.sidebar.title("Object Detection and Crowd Counting Web App")
st.sidebar.write("Upload an image or a video to perform object detection or crowd counting.")

# Selection for image or video
media_type = st.sidebar.radio("Choose a media type:", ("Image", "Video"))

# Buttons for object detection and crowd counting
task = st.sidebar.radio("Choose a task:", ("Object Detection", "Crowd Counting"))

# Detection tuning (helps reduce false labels like "motorcycle" on people)
st.sidebar.subheader("Detection settings")
model_choice = st.sidebar.selectbox(
    "Model",
    ("yolov8x.pt (most accurate, slower)", "yolov8l.pt", "yolov8n.pt (fastest)"),
    index=0,
)
model_path = {
    "yolov8x.pt (most accurate, slower)": "weights/yolov8x.pt",
    "yolov8l.pt": "weights/yolov8l.pt",
    "yolov8n.pt (fastest)": "weights/yolov8n.pt",
}[model_choice]

conf_threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.40, 0.05)
iou_threshold = st.sidebar.slider("IoU (overlap) threshold", 0.10, 0.90, 0.45, 0.05)
image_size = st.sidebar.select_slider("Input size (px)", options=[640, 960, 1280], value=960)

# File uploader for image or video
uploaded_file = st.sidebar.file_uploader("Choose a file...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])


@st.cache_resource
def load_yolo_model(weights_path: str):
    return YOLO(weights_path)


def run_detection(model, source, conf: float, iou: float, imgsz: int):
    return model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=False,
        verbose=False,
    )


def draw_box_label(image, x1, y1, x2, y2, label: str, color, font):
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    (text_w, text_h), baseline = cv2.getTextSize(label, font, 0.6, 2)
    y_text = max(y1, text_h + 8)
    cv2.rectangle(
        image,
        (x1, y_text - text_h - 6),
        (x1 + text_w + 6, y_text + baseline),
        color,
        -1,
    )
    cv2.putText(image, label, (x1 + 3, y_text - 4), font, 0.6, (0, 0, 0), 2)

# Main title and instructions
st.title("🔍 Object Detection and Crowd Counting")
st.write("Upload an image or a video to perform object detection or crowd counting using YOLOv8 model.")

if uploaded_file is not None:
    # Increment the counter
    counter += 1
    save_counter(counter_file_path, counter)

    if media_type == "Image":
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        model = load_yolo_model(model_path)
        detection_output = run_detection(
            model, image, conf_threshold, iou_threshold, image_size
        )

        # Class names (assuming COCO dataset)
        class_names = model.names

        # Generate a color map for the classes
        np.random.seed(42)  # Seed for reproducibility
        colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype='uint8')

        font = cv2.FONT_HERSHEY_SIMPLEX

        if task == "Crowd Counting":
            # Count the number of persons
            person_count = 0

            # Draw bounding boxes and class names for persons only
            for box in detection_output[0].boxes:
                clsID = int(box.cls.numpy()[0])
                class_name = class_names[clsID]
                conf = float(box.conf.numpy()[0])

                if class_name == "person":
                    person_count += 1
                    color = [int(c) for c in colors[clsID]]
                    x1, y1, x2, y2 = map(int, box.xyxy.numpy()[0])
                    draw_box_label(
                        image,
                        x1,
                        y1,
                        x2,
                        y2,
                        f"person {conf:.0%}",
                        color,
                        font,
                    )

            # Display total person count on the image
            cv2.putText(image, f"Total count of persons: {person_count}", (10, 30), font, 1, (255, 255, 255), 2)
            st.write(f"Total count of persons: {person_count}")

        elif task == "Object Detection":
            # Count the number of objects
            num_objects = len(detection_output[0].boxes)
            st.write(f"Number of objects: {num_objects}")

            # Draw bounding boxes and class names on the image
            for box in detection_output[0].boxes:
                clsID = int(box.cls.numpy()[0])
                class_name = class_names[clsID]
                conf = float(box.conf.numpy()[0])
                color = [int(c) for c in colors[clsID]]
                x1, y1, x2, y2 = map(int, box.xyxy.numpy()[0])
                draw_box_label(
                    image,
                    x1,
                    y1,
                    x2,
                    y2,
                    f"{class_name} {conf:.0%}",
                    color,
                    font,
                )

            # Display total object count on the image
            cv2.putText(image, f"Total count of objects: {num_objects}", (10, 30), font, 1, (255, 255, 255), 2)

        # Convert image to RGB for display in Streamlit
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the image with bounding boxes
        st.image(image_rgb, caption="Processed Image", use_container_width=True)

    elif media_type == "Video":
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        if not cap.isOpened():
            st.error("Error: Cannot open video file.")
        else:
            model = load_yolo_model(model_path)

            # Generate random colors for class list
            np.random.seed(42)  # Seed for reproducibility
            detection_colors = []
            class_list = model.names
            for i in range(len(class_list)):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                detection_colors.append((b, g, r))

            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Predict on image
                detect_params = run_detection(
                    model, [frame], conf_threshold, iou_threshold, image_size
                )

                frame_object_counts = {}
                font = cv2.FONT_HERSHEY_SIMPLEX

                for box in detect_params[0].boxes:
                    clsID = int(box.cls.numpy()[0])
                    class_name = class_list[clsID]
                    conf = float(box.conf.numpy()[0])
                    bb = box.xyxy.numpy()[0]

                    if task == "Crowd Counting" and class_name != "person":
                        continue

                    frame_object_counts[class_name] = frame_object_counts.get(class_name, 0) + 1
                    color = detection_colors[clsID]
                    x1, y1, x2, y2 = map(int, bb)
                    draw_box_label(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        f"{class_name} {conf:.0%}",
                        color,
                        font,
                    )

                if task == "Crowd Counting":
                    person_count = frame_object_counts.get("person", 0)
                    cv2.putText(
                        frame,
                        f"Person Count: {person_count}",
                        (10, 30),
                        font,
                        1,
                        (255, 255, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        f"Objects this frame: {sum(frame_object_counts.values())}",
                        (10, 30),
                        font,
                        1,
                        (255, 255, 255),
                        2,
                    )
                    for idx, (obj, count) in enumerate(frame_object_counts.items()):
                        cv2.putText(
                            frame,
                            f"{obj}: {count}",
                            (10, 60 + 22 * idx),
                            font,
                            0.55,
                            (255, 255, 255),
                            2,
                        )

                stframe.image(frame, channels="BGR", use_container_width=True)

            cap.release()
        os.remove(tfile.name)

# Add some custom CSS to enhance the appearance
st.markdown("""
    <style>
        .css-18e3th9 {
            padding: 2rem 1rem;
        }
        .css-1d391kg {
            display: none;
        }
        .css-hxt7ib {
            font-size: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)
