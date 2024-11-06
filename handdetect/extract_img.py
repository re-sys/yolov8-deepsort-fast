import cv2
import os

# 创建输出目录
output_dir = 'new_train'

# 如果输出目录不存在，创建该文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义函数提取帧
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    frame_count = 0
    saved_count = len(os.listdir(output_dir))  # 获取当前已保存图片的数量
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"读取视频帧失败, 当前帧数: {frame_count}")
            break
            
        # 每10帧显示一张图像
        if frame_count % 10 == 0:
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(0)  # 等待键盘输入
            
            if key == 13:  # 如果按下Enter键 (ASCII码为13)
                img_name = os.path.join(output_dir, f'frame{saved_count}.jpg')
                cv2.imwrite(img_name, frame)
                saved_count += 1
                print(f"保存图像: {img_name}")
            elif key == ord('j'):  # 如果按下 'j' 键
                print("切换到下一帧")
            else:
                break  # 按下其他键则退出

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# 提取指定视频的帧
extract_frames('testmp4/zoulang2.mp4')

print("帧提取完成！")
