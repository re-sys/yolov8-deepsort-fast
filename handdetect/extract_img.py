import cv2
import os
import shutil

# 创建输出目录
output_dir = 'train'

# 如果输出目录已存在，先清空该文件夹
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# 定义函数提取帧
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"读取视频帧失败, 当前帧数: {frame_count}")
            break
            
        # 检查当前帧是否符合条件
        if video_path == '5.mp4':
            if 11 <= frame_count // 10 <= 165:  # 判断frame_count // 10 是否在11和165之间
                if frame_count % 10 == 0:  # 保存每十帧的图像
                    img_name = os.path.join(output_dir, f'{os.path.basename(video_path)}_frame{saved_count}.jpg')
                    cv2.imwrite(img_name, frame)
                    saved_count += 1
        elif video_path == '4.mp4':  # 对于4.mp4提取帧范围在14到231之间
            if 14 <= frame_count // 10 <= 231:  # 判断frame_count // 10 是否在14和231之间
                if frame_count % 10 == 0:  # 保存每十帧的图像
                    img_name = os.path.join(output_dir, f'{os.path.basename(video_path)}_frame{saved_count}.jpg')
                    cv2.imwrite(img_name, frame)
                    saved_count += 1
            
        frame_count += 1

    cap.release()

# # 提取第一个视频的指定帧范围
# extract_frames('3.mp4')

# # 提取第二个视频的指定帧范围
# extract_frames('4.mp4')
extract_frames('5.mp4')

print("帧提取完成！")
