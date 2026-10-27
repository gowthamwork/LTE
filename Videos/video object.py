
import cv2
import numpy as np
import os
import time
import threading

# ========== FILE PATHS ==========
folder_path = r'D:\Gowtham\DRONE\Demo\OB'

config_path = os.path.join(folder_path, 'yolov4.cfg')
weights_path = os.path.join(folder_path, 'yolov4.weights')
names_path = os.path.join(folder_path, 'coco.names')
video_path = os.path.join(folder_path, '4.mp4')

# ========== SETTINGS ==========
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Smaller input can improve CPU speed.
# 416 is the standard YOLOv4 input size.
YOLO_INPUT_SIZE = (416, 416)

MAX_DISPLAY_WIDTH = 800
MAX_DISPLAY_HEIGHT = 900

# ========== LOAD LABELS ==========
with open(names_path, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# ========== LOAD YOLO MODEL ==========
net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [
    layer_names[i - 1]
    for i in np.array(
        net.getUnconnectedOutLayers()
    ).flatten()
]

# ========== OPEN VIDEO ==========
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Cannot open video.")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)

if not np.isfinite(fps) or fps <= 0:
    fps = 30.0

frame_delay = 1.0 / fps

print(f"Video FPS: {fps:.2f}")
print("Press Q or ESC to quit.")

# ========== SHARED DATA ==========
lock = threading.Lock()
stop_event = threading.Event()

shared_frame = None
shared_frame_id = 0

latest_boxes = []
latest_confidences = []
latest_class_ids = []

# ========== YOLO DETECTION THREAD ==========
def detection_worker():

    global shared_frame
    global shared_frame_id
    global latest_boxes
    global latest_confidences
    global latest_class_ids

    last_processed_id = -1

    while not stop_event.is_set():

        with lock:
            if shared_frame is None:
                frame = None
                current_id = -1
            else:
                current_id = shared_frame_id

                if current_id == last_processed_id:
                    frame = None
                else:
                    frame = shared_frame.copy()

        if frame is None:
            time.sleep(0.002)
            continue

        last_processed_id = current_id

        height, width = frame.shape[:2]

        # YOLO input
        blob = cv2.dnn.blobFromImage(
            frame,
            1 / 255.0,
            YOLO_INPUT_SIZE,
            swapRB=True,
            crop=False
        )

        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        # Process detections
        for output in outputs:
            for detection in output:

                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                box_width = int(detection[2] * width)
                box_height = int(detection[3] * height)

                x = int(center_x - box_width / 2)
                y = int(center_y - box_height / 2)

                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(width, x + box_width)
                y2 = min(height, y + box_height)

                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([
                    x1, y1, x2 - x1, y2 - y1
                ])
                confidences.append(confidence)
                class_ids.append(class_id)

        # Non-Maximum Suppression
        indexes = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD
        )

        selected_boxes = []
        selected_confidences = []
        selected_class_ids = []

        if len(indexes) > 0:
            for i in np.array(indexes).flatten():
                selected_boxes.append(boxes[i])
                selected_confidences.append(confidences[i])
                selected_class_ids.append(class_ids[i])

        # Publish latest detection results
        with lock:
            latest_boxes = selected_boxes
            latest_confidences = selected_confidences
            latest_class_ids = selected_class_ids


# ========== START DETECTION THREAD ==========
worker = threading.Thread(
    target=detection_worker,
    daemon=True
)
worker.start()

# ========== DISPLAY WINDOW ==========
window_name = "YOLOv4 Real-Time Playback"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(
    window_name,
    MAX_DISPLAY_WIDTH,
    MAX_DISPLAY_HEIGHT
)

# ========== VIDEO PLAYBACK LOOP ==========
next_frame_time = time.perf_counter()

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Video finished.")
            break

        height, width = frame.shape[:2]

        # Publish newest frame for detection thread
        with lock:
            shared_frame = frame.copy()
            shared_frame_id += 1

            boxes = list(latest_boxes)
            confidences = list(latest_confidences)
            class_ids = list(latest_class_ids)

        # Draw most recent detections
        for box, confidence, class_id in zip(
            boxes, confidences, class_ids
        ):

            x, y, w, h = box

            label_name = (
                class_names[class_id]
                if class_id < len(class_names)
                else str(class_id)
            )

            label = f"{label_name} {confidence:.2f}"

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Fit complete frame into display area
        scale = min(
            MAX_DISPLAY_WIDTH / width,
            MAX_DISPLAY_HEIGHT / height
        )

        display_width = max(1, int(width * scale))
        display_height = max(1, int(height * scale))

        display_frame = cv2.resize(
            frame,
            (display_width, display_height),
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow(window_name, display_frame)

        # Maintain target playback FPS
        next_frame_time += frame_delay

        remaining = next_frame_time - time.perf_counter()

        if remaining > 0:
            time.sleep(remaining)
        else:
            # Playback is behind schedule.
            next_frame_time = time.perf_counter()

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            print("Stopped by user.")
            break

finally:
    stop_event.set()
    worker.join(timeout=2.0)
    cap.release()
    cv2.destroyAllWindows()

print("Program closed.")
