
import cv2
import numpy as np
import os
import time

# === File Paths ===
folder_path = r'D:\Gowtham\DRONE\Demo\OB'

config_path = os.path.join(folder_path, 'yolov4.cfg')
weights_path = os.path.join(folder_path, 'yolov4.weights')
names_path = os.path.join(folder_path, 'coco.names')
video_path = os.path.join(folder_path, '1.mp4')

# === Detection Settings ===
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
YOLO_INPUT_SIZE = (416, 416)

# Process every Nth frame
# 1 = detect every frame (slower, more detections)
# 2 = detect every 2nd frame (faster)
# 3 = detect every 3rd frame (even faster)
SKIP_FRAMES = 2

# Maximum display dimensions
MAX_DISPLAY_WIDTH = 800
MAX_DISPLAY_HEIGHT = 900

# === Load Class Labels ===
with open(names_path, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# === Load YOLOv4 Model ===
net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# === Output Layers ===
layer_names = net.getLayerNames()
output_layers = [
    layer_names[i - 1]
    for i in np.array(net.getUnconnectedOutLayers()).flatten()
]

# === Open Video ===
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    raise SystemExit

# === Video Information ===
fps = cap.get(cv2.CAP_PROP_FPS)

if not np.isfinite(fps) or fps <= 0:
    fps = 30.0

frame_delay = max(1, int(1000 / fps))

print(f"Original FPS: {fps:.2f}")
print("Press Q or ESC to exit.")

# === Create Display Window ===
window_name = "YOLOv4 Object Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(
    window_name,
    MAX_DISPLAY_WIDTH,
    MAX_DISPLAY_HEIGHT
)

frame_count = 0
last_boxes = []
last_confidences = []
last_class_ids = []

# === Main Loop ===
while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video.")
        break

    frame_count += 1
    height, width = frame.shape[:2]

    # === Run Detection Only on Selected Frames ===
    run_detection = (
        (frame_count - 1) % SKIP_FRAMES == 0
    )

    if run_detection:
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 255.0,
            size=YOLO_INPUT_SIZE,
            swapRB=True,
            crop=False
        )

        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        # === Process YOLO Detections ===
        for output in outputs:
            for detection in output:

                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence > CONFIDENCE_THRESHOLD:
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

        # === Non-Maximum Suppression ===
        indexes = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD
        )

        # Keep detections for display between inference frames
        last_boxes = []
        last_confidences = []
        last_class_ids = []

        if len(indexes) > 0:
            for i in np.array(indexes).flatten():
                last_boxes.append(boxes[i])
                last_confidences.append(confidences[i])
                last_class_ids.append(class_ids[i])

    # === Draw Latest Detections ===
    for box, confidence, class_id in zip(
        last_boxes, last_confidences, last_class_ids
    ):
        x, y, w, h = box

        label = (
            f"{class_names[class_id]} {confidence:.2f}"
        )

        color = (0, 255, 0)

        cv2.rectangle(
            frame, (x, y), (x + w, y + h), color, 2
        )

        text_y = max(y - 10, 25)

        cv2.putText(
            frame,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # === Resize Entire Frame for Display ===
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

    # === Display Frame ===
    cv2.imshow(window_name, display_frame)

    # Wait according to original video FPS
    key = cv2.waitKey(frame_delay) & 0xFF

    if key == ord('q') or key == 27:
        print("Stopped by user.")
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()

print("Program closed.")

