import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
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

# Create videoWriter object to save the video
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter("output.avi", fourcc, int(fps), (width, height), isColor=False)

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

    # Make the frame grey-scale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    out.write(gray)  # Write the frame to the output video file

    # Add text to the frame
    cv.putText(gray, 'Press q to quit', (250, 250), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    # Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv.destroyAllWindows()