import numpy as np
import cv2 as cv

cap = cv.VideoCapture('output.avi')  # Change the index if you have multiple cameras
if not cap.isOpened():
    print("Error: Video not found.")
    exit()

delay = int(1000 / cap.get(cv.CAP_PROP_FPS))  # Calculate delay based on FPS

# Print camera info
print(f"Camera Resolution: {cap.get(cv.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv.CAP_PROP_FRAME_HEIGHT)}")
print(f"Camera Framerate: {cap.get(cv.CAP_PROP_FPS)}")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if the frame is read correcltly, ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv.imshow('frame', frame)

    if cv.waitKey(delay) & 0xFF == ord("q"):
        print("Playback stopped by user.")
        break

# When everything done, release the capture
cap.release()
# out.release()
cv.destroyAllWindows()