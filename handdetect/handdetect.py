from ultralytics import YOLO
import cv2
# cv2.namedWindow("preview")
modelname="/home/wu/Lab/yolov8-deepsort-fast/handdetect/metrics/train4/weights/best.pt"
model = YOLO(modelname)
# video_path="/home/wu/Lab/yolov8-deepsort-fast/handdetect/testmp4/yangtai.mp4"
video_path="/home/wu/Lab/yolov8-deepsort-fast/handdetect/4.mp4"
vc = cv2.VideoCapture(video_path)
import cv2
import time

# Assume 'vc' is your video capture object and 'model' is defined above
frame_count = 0
start_time = time.time()

while True:
    rval, frame = vc.read()
    if not rval:
        break
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
vc.release()
# import cv2
# import os

# # 创建保存图像的目录
# output_dir = 'train2'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # 打开摄像头
# vc = cv2.VideoCapture(0)

# # 初始化变量
# frame_count = 0
# save_count = 0
# is_saving = False

# while True:
#     rval, frame = vc.read()
#     if not rval:
#         print("无法读取视频帧，退出...")
#         break

#     # 更新帧计数
#     frame_count += 1

#     if is_saving and frame_count % 10 == 0:  # 每十帧保存一张图像
#         img_name = os.path.join(output_dir, f'frame_{save_count}.jpg')
#         cv2.imwrite(img_name, frame)
#         print(f'Saved: {img_name}')
#         save_count += 1

#     # 显示帧
#     cv2.imshow("preview", frame)

#     # 处理按键
#     waitkey = cv2.waitKey(1)
#     if waitkey == ord('s'):  # 按键 's' 开始保存视频帧
#         is_saving = True
#         print("开始保存视频帧...")
#     elif waitkey == ord('t'):  # 按键 't' 结束保存
#         is_saving = False
#         print("结束保存视频帧...")
#     elif waitkey == 27:  # 按键 ESC 退出
#         break

# # 清理
# cv2.destroyWindow("preview")
# vc.release()
