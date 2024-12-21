from ultralytics import YOLO
import cv2
import openvino as ov
import torch
import time
# parent_dir = "/home/wu/Lab/yolov8-deepsort-fast/handdetect/"
# # cv2.namedWindow("preview")
# modelname="/home/wu/Lab/yolov8-deepsort-fast/handdetect/metrics/train4/weights/best.pt"
# modelname="/home/wu/Lab/yolov8-deepsort-fast/handdetect/paperrs.pt"
# modelname="/home/wu/Lab/yolov8-deepsort-fast/handdetect/metrics/train4/weights/best_int8_openvino_model"
model = YOLO("gesture_ncnn_model")
# model.predictor.model.input_shape = (3, 320, 320)
video_path="/home/wu/Lab/yolov8-deepsort-fast/handdetect/testmp4/yangtai.mp4"
# video_path="/home/wu/Lab/yolov8-deepsort-fast/handdetect/4.mp4"
vc = cv2.VideoCapture(0)

# core = ov.Core()

# model.to("cpu")
#     # IMAGE_PATH="/home/wu/my_ws/src/yolo_pub/yolo_pub/data/coco_bike.jpg"
# img = cv2.imread("/home/wu/Lab/yolov8-deepsort-fast/handdetect/train_img1/5.mp4_frame0.jpg")
# res = model(img)
# det_model_path =parent_dir + "best_openvino_model/best.xml"
# det_ov_model = core.read_model(det_model_path)
# ov_config = {}
# device = "CPU"

# compiled_model = core.compile_model(det_ov_model, device, ov_config)
# def infer(*args):
#     result = compiled_model(args)
#     return torch.from_numpy(result[0])
# model.predictor.model.pt = False
# model.predictor.inference = infer
# Assume 'vc' is your video capture object and 'model' is defined above
frame_count = 0
start_time = time.time()

while True:
    rval, frame = vc.read()
    if not rval:
        break
    # print(frame.shape)
    # frame = cv2.resize(frame, (320, 320))
    results = model.predict(frame,iou=0.3,imgsz=320,agnostic_nms=True)
    decorated_frame = results[0].plot()
    for res in results:
        # print(res.boxes)
        if len(res.boxes.cls.cpu().numpy()) == 0:
            print("no hand detected")
        for box in res.boxes:
            box = box.cpu().numpy()
            
            class_id = res.names[int(box.cls)]
            # print(int(box.cls), class_id)
            
            # cv2.rectangle(frame, (box.xmin, box.ymin), (box.xmax, box.ymax), (255, 0, 0), 2)
            
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
vc.release()
