import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

fx = 897.6827392578125
fy = 898.0628051757812
cx = 649.2904052734375
cy = 376.3505859375
Rot = np.array([[0,0,1],[-1,0,0],[0,-1,0]])
# Configure depth and rgb and infrared streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
# config.enable_stream(rs.stream.infrared, 1024, 768, rs.format.y8, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)
# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

align_to = rs.stream.color
align = rs.align(align_to)
model = YOLO("/home/wu/catkin_ws/src/yolov8n_ncnn_model")
def get_center(boxes):
    for box in boxes:
            r = box.xyxy[0].tolist()  # 获取边界框的x1, y1, x2, y2坐标
            if len(r) == 4 and box.conf>0.5:
                x1, y1, x2, y2 = r
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                print(f"对象中心点坐标: ({center_x}, {center_y})")
                return (center_x,center_y)
                
            else:
                print("检测不到对象")
                return (0,0)

def projection(center_x,center_y,depth_value):
    # 计算相机坐标系下的点坐标
    x = (center_x - cx) * depth_value / fx
    y = (center_y - cy) * depth_value / fy
    z = depth_value
    x,y,z = np.dot(Rot, np.array([x,y,z]))
    yaw = np.arctan2(y,x)
    backward_distance = 1
    backward_x = x - backward_distance * np.cos(yaw)
    backward_y = y - backward_distance * np.sin(yaw)
    return x,y,z,yaw,backward_x,backward_y

def get_region(depth_image,center_x,center_y,kernel_size=5):
     # 定义5x5的kernel
    kernel_size = 5
    half_kernel = kernel_size // 2

    # 确保中心点周围的区域在图像边界内
    if (center_x - half_kernel >= 0 and center_x + half_kernel < depth_image.shape[1] and
        center_y - half_kernel >= 0 and center_y + half_kernel < depth_image.shape[0]):
        # 提取5x5区域的深度值
        kernel = depth_image[center_y - half_kernel:center_y + half_kernel + 1,
                            center_x - half_kernel:center_x + half_kernel + 1]
        # 过滤掉零值
        non_zero_kernel = kernel[kernel != 0]
        
        # 检查是否没有非零值
        if non_zero_kernel.size == 0:
            print("中心点周围的5x5区域内没有非零深度值")
            return 0, 0, 0
        
        min_depth_value = np.min(non_zero_kernel)
        min_depth_position = np.argmin(non_zero_kernel)
        min_depth_x = center_x - half_kernel + min_depth_position % kernel_size
        min_depth_y = center_y - half_kernel + min_depth_position // kernel_size
        print(f"最小深度值: {min_depth_value}, 位置: ({min_depth_x}, {min_depth_y})")
        return min_depth_value, min_depth_x, min_depth_y
    else:
        print("中心点周围的5x5区域超出了图像边界")
        return 0, 0, 0
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        flag = True
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        results = model.predict(color_image,imgsz=320,half=True,classes=[0],iou=0.2)
        boxes = results[0].boxes
        color_image = results[0].plot()
        center_x,center_y = get_center(boxes)
        if center_x==0:
            continue
        min_depth_value, min_depth_x, min_depth_y = get_region(depth_image,center_x,center_y)
        min_depth_value = min_depth_value * depth_scale
        projection_result = projection(center_x,center_y,min_depth_value)
        x,y,z = projection_result[:3]
        print(f"相机坐标系下的点坐标: ({projection_result[0]:.2f}, {projection_result[1]:.2f}, {projection_result[2]:.2f}), 偏航角: {projection_result[3]:.2f}, 后方距离: {projection_result[4]:.2f}, {projection_result[5]:.2f}")
        if min_depth_value <= 1 or min_depth_value>=2:
            continue
        # 绘制最小深度值点
        cv2.circle(color_image, (min_depth_x, min_depth_y), 5, (0, 0, 255), -1)
        cv2.putText(color_image, f"({x:.2f}, {y:.2f}, {z:.2f})", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(color_image, f"Min Depth: {min_depth_value:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('color_image', color_image)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        
        # frame = reaults[0].plot()
        # cv2.imshow('color_image', frame)

        # cv2.imshow('depth_image', depth_image)
        # key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        # if key & 0xFF == ord('q') or key == 27:
        #     cv2.destroyAllWindows()
        #     break
except Exception as e:
    print(e)
    pass
finally:
    pipeline.stop()
