import numpy as np
import cv2 as cv
import urllib.request

# 1. Download OpenCV's lightweight YuNet model if not present locally
model_path = "face_detection_yunet_2023mar.onnx"
model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

try:
    urllib.request.urlretrieve(model_url, model_path)
except Exception:
    pass  # Use local file if download fails or already exists

# 2. Initialize webcam capture
cap = cv.VideoCapture(0)
# cap = cv.VideoCapture('output.avi')  # Change the index if you have multiple cameras
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get resolution from webcam
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Fallback to 30 FPS if webcam reports 0
fps = cap.get(cv.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

# 3. Create the YuNet Face Detector
detector = cv.FaceDetectorYN.create(
    model=model_path,
    config="",
    input_size=(width, height),
    score_threshold=0.5,  # Confidence threshold (0.0 to 1.0)
    nms_threshold=0.3,
    top_k=5000,
)

# 4. Change this flag to True if you want to save the video output
save_video = False

# Create videoWriter object to save the video
if save_video:
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter("face_detection.avi", fourcc, int(fps), (width, height), isColor=False)

# Print camera info
# print(f"Camera Resolution: {cap.get(cv.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv.CAP_PROP_FRAME_HEIGHT)}")
# print(f"Camera Framerate: {cap.get(cv.CAP_PROP_FPS)}")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if the frame is read correcltly, ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    detector.setInputSize((frame.shape[1], frame.shape[0]))
    # Perform inference: faces is an array of [x, y, w, h, x_re, y_re, x_le, y_le, ...]
    _, faces = detector.detect(frame)

    # Draw results
    if faces is not None:
        for face in faces:
            # Extract bounding box coordinates
            box = list(map(int, face[:4]))
            confidence = face[14]

            # Draw bounding box
            cv.rectangle(
                frame,
                (box[0], box[1]),
                (box[0] + box[2], box[1] + box[3]),
                (0, 255, 0),
                2,
            )
            cv.putText(
                frame,
                f"Face: {confidence:.2f}",
                (box[0], max(0, box[1] - 10)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Draw 5 facial landmarks (eyes, nose, mouth corners)
            landmarks = list(map(int, face[4:14]))
            for i in range(0, 10, 2):
                cv.circle(
                    frame, (landmarks[i], landmarks[i + 1]), 2, (0, 0, 255), 2
                )
    cv.imshow("OpenCV YuNet Real-Time Face Detection", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

    # # Make the frame grey-scale
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # if save_video:
    #     out.write(gray)  # Write the frame to the output video file

    # # Add text to the frame
    # cv.putText(gray, 'Press q to quit', (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    # # Display the resulting frame
    # cv.imshow('frame', gray)
    # if cv.waitKey(1) == ord('q'):
    #     break

# When everything done, release the capture
cap.release()
if save_video:
    out.release()
cv.destroyAllWindows()