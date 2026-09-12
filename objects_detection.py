import cv2 as cv
from ultralytics import YOLO

# 1. Import the YOLOv8 model (weights will be downloaded automatically if not present locally)
model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8n model

# 2. Initialize webcam capture
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if the frame is read correcltly, ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Make the frame grey-scale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 3. Perform inference on the frame
    results = model(gray)
    print(results)

    # 4. Render the results on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv.imshow('YOLOv8 Object Detection', annotated_frame)

    if cv.waitKey(1) == ord('q'):
        break