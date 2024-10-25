import cv2

# 打开视频文件
video_path = 'output.avi'
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

while True:
    # 逐帧读取视频
    ret, frame = cap.read()
    
    # 如果成功读取到帧，则显示
    if ret:
        cv2.imshow('Video', frame)
        
        # 按下'q'键退出
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# 释放视频捕获对象和关闭所有OpenCV窗口
cap.release()
cv2.destroyAllWindows()
