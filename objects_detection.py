import cv2 as cv
from ultralytics import YOLO
import numpy as np

# 1. Import the YOLOv8 model (weights will be downloaded automatically if not present locally)
model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8n mode

# 2. Initialize webcam capture
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(len(model.names), "classes detected by YOLOv8 model:\n", model.names)
# Initialize variables for object detection averaging
w = 10 # averaging window size (number of frames)
confidence_history = np.zeros((len(model.names), w))  # Initialize confidence history array
frame_count = 0  # Initialize frame counter

# for _ in range(13):  # Warm up the camera
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if the frame is read correcltly, ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # 3. Perform inference on the frame
    results = model(frame)
    result = results[0]  # Get the first result (since we are processing one frame at a time)

    if result.boxes is not None and len(result.boxes) > 0:

        # Option A: Extract as NumPy arrays (Best for standard OpenCV work)
        boxes = result.boxes.xyxy.cpu().numpy()  # [[x1, y1, x2, y2], ...]
        confidences = result.boxes.conf.cpu().numpy()  # [conf1, conf2, ...]
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # [0, 67, ...]

        if frame_count < w:
            # Fill the confidence history for the first w frames
            confidence_history[class_ids, frame_count] = confidences
        else:
            # Update the confidence history for subsequent frames
            confidence_history[:,0:-1] = confidence_history[:,1:]  # Shift left
            confidence_history[class_ids, -1] = confidences  # Add new confidences to

            # Detect an object only if detected in the last w frames with an average confidence above a threshold
            avg_confidences = np.mean(confidence_history, axis=1)
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)  # Convert float coordinates to integers
                class_name = model.names[class_id]  # Look up string name from ID

                # Filter or process manually
                if avg_confidences[class_id] > 0.5:
                    print(
                        f"Detected {model.names[class_id]} ({conf:.2f}) at [{box[0]}, {box[1]}, {box[2]}, {box[3]}]"
                    )

                    # Draw custom box and label manually using OpenCV
                    cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv.putText(
                        frame,
                        f"{class_name} {conf:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )

        print("Confidence history:\n", confidence_history[(0,41), :])  # Print the first 5 classes for brevity

    print("\n", boxes, confidences, class_ids, "\n")  # Print the bounding boxes, confidences, and class IDs


    # Display the resulting frame
    cv.imshow("Manual YOLO Parsing", frame)

    if cv.waitKey(1) == ord('q'):
        break

    frame_count += 1  # Increment frame counter

print(confidence_history)