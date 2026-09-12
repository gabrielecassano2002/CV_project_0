from collections import defaultdict, deque
import cv2 as cv
import numpy as np
from ultralytics import YOLO

# 1. Initialize YOLO model
model = YOLO("yolov8n.pt")

# 2. Setup a rolling history buffer for each tracked object (max length = 10)
# Structure: { track_id: deque([conf1, conf2, ...], maxlen=10) }
confidence_history = defaultdict(lambda: deque(maxlen=10))

cap = cv.VideoCapture(0)

# Get resolution from webcam
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Fallback to 30 FPS if webcam reports 0
fps = cap.get(cv.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

# Create videoWriter object to save the video
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter("obj_recog.avi", fourcc, int(fps), (width, height), isColor=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run tracking (persist=True maintains IDs across video frames)
    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().numpy()

        for box, conf, track_id, cls_id in zip(
            boxes, confidences, track_ids, class_ids
        ):
            # Append current confidence score to the object's specific buffer
            confidence_history[track_id].append(conf)

            # Compute the moving average over the last N frames (up to 10)
            avg_conf = float(np.mean(confidence_history[track_id]))

            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[cls_id]

            # Render custom box with averaged confidence score
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID #{track_id} {class_name} Avg Conf: {avg_conf:.2f}"
            cv.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    cv.imshow("YOLO Real-Time Detection with Temporal Averaging", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
    out.write(frame)  # Write the frame to the output video file

cap.release()
out.release()
cv.destroyAllWindows()