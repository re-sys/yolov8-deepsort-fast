from ultralytics import YOLO
import cv2
cv2.namedWindow("preview")
modelname="/home/wu/my_ws/src/temp/handdetect.pt"
model = YOLO(modelname)
vc = cv2.VideoCapture("/home/wu/my_ws/src/model-train/img/1.mp4" )
import cv2
import time

# Assume 'vc' is your video capture object and 'model' is defined above
frame_count = 0
start_time = time.time()

while True:
    rval, frame = vc.read()
    results = model(frame)
    decorated_frame = results[0].plot()
    
    # Update frame count
    frame_count += 1
    elapsed_time = time.time() - start_time

    # Calculate FPS
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
    else:
        fps = 0

    # Draw FPS on the frame
    cv2.putText(decorated_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("preview", decorated_frame)
    waitkey = cv2.waitKey(1)
    if waitkey == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
