#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters

import tf
import tf.transformations as transformations
from geometry_msgs.msg import PoseStamped  # 导入PoseStamped消息类型

from ultralytics import YOLO
import time
import openvino as ov
import torch
# import time

import deep_sort.deep_sort.deep_sort as ds
import tf.transformations
from FSM import State
from FSM import StatePublisher
import threading

bridge = CvBridge()
#内参
fx = 897.6827392578125
fy = 898.0628051757812
cx = 649.2904052734375
cy = 376.3505859375
y_mouse,x_mouse = 100,100
global_results_tracker = None
def display_image(image):
    cv2.imshow("Color Image window", image)
    cv2.waitKey(100)
# parent_dir = "/home/wu/Lab/yolov8-deepsort-fast/"
tracker = ds.DeepSort("/home/wu/catkin_ws/src/my_pkg/cv_pkg/scripts/ckpt.t8")
Rot = np.array([[0,0,1],[-1,0,0],[0,-1,0]])
# tran = np.array([0,0,0.521])
# def mouse_callback(event, x, y, flags, param):
#     global x_mouse, y_mouse
#     if event == cv2.EVENT_LBUTTONDOWN:
#         x_mouse = x
#         y_mouse = y
#         print(f"Mouse clicked at: x={x}, y={y}")

def extract_detections(results, detect_class):
    """
    从模型结果中提取和处理检测信息。
    - results: YoloV8模型预测结果，包含检测到的物体的位置、类别和置信度等信息。
    - detect_class: 需要提取的目标类别的索引。
    参考: https://docs.ultralytics.com/modes/predict/#working-with-results
    """
    
    # 初始化一个空的二维numpy数组，用于存放检测到的目标的位置信息
    # 如果视频中没有需要提取的目标类别，如果不初始化，会导致tracker报错
    detections = np.empty((0, 4)) 
    
    confarray = [] # 初始化一个空列表，用于存放检测到的目标的置信度。
    
    # 遍历检测结果
    # 参考：https://docs.ultralytics.com/modes/predict/#working-with-results
    for r in results:
        for box in r.boxes:
            # 如果检测到的目标类别与指定的目标类别相匹配，提取目标的位置信息和置信度
            if box.cls[0].int() == detect_class:
                x1, y1, x2, y2 = box.xywh[0].int().tolist() # 提取目标的位置信息，并从tensor转换为整数列表。
                conf = round(box.conf[0].item(), 2) # 提取目标的置信度，从tensor中取出浮点数结果，并四舍五入到小数点后两位。
                detections = np.vstack((detections, np.array([x1, y1, x2, y2]))) # 将目标的位置信息添加到detections数组中。
                confarray.append(conf) # 将目标的置信度添加到confarray列表中。
    return detections, confarray # 返回提取出的位置信息和置信度。
def image_show(cv_image):
    cv2.imshow("Image Window", cv_image)
    cv2.waitKey(1)

def image_callback(color_image_msg,depth_image_msg):
    global x_mouse, y_mouse, global_results_tracker
    current_state = rospy.get_param("current_state", "WAITING")  # 默认值为 "WAITING"

    # 检查状态是否为 TRACKING
    if current_state != State.TRACKING:
        print("Not in TRACKING state, return")
        return  # 如果不是 TRACKING，则直接返回，不处理图像
    
    
    try:
        cv_color_image = bridge.imgmsg_to_cv2(color_image_msg, desired_encoding="bgr8")
        cv_aligned_depth_image = bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding="passthrough")  # 假设深度图是直接以浮点格式传递
    except CvBridgeError as e:
        rospy.logerr(f"Could not convert image: {e}")
        return
    
    
    
    detected = False
    results = model.predict(cv_color_image,imgsz=320,half=True,classes=[0],iou=0.3)
    boxes = results[0].boxes
    frame = results[0].plot()
    # try:
    #     print("Image shape:", frame.shape)
    #     print("Image data type:", frame.dtype)
        
    #     show_thread = threading.Thread(target = image_show, args=(frame,))
    #     show_thread.start()
    # except Exception as e:
    #     rospy.logerr(e)
    
    for box in boxes:
        xyxy = box.xyxy.tolist()
        if len(xyxy) == 4 and box.conf>0.5:
            x1, y1, x2, y2 = box.xyxy.tolist()
            x_mouse = (x1 + x2) // 2
            y_mouse = (y1 + y2) // 2
            detected = True
            break
    # detections, confarray = extract_detections(results, 0)
    # resultsTracker = tracker.update(detections, confarray, cv_color_image)
    # # global_results_tracker = resultsTracker
    # for x1, y1, x2, y2, Id in resultsTracker:
    #     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert position to integers.  
    #     # print(f"Id: {Id}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
    #     if Id == 1:
    #         x_mouse = (x1 + x2) // 2
    #         y_mouse = (y1 + y2) // 2
    #         detected = True
    #         # cv2.rectangle(cv_color_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
    #         # cv2.putText(cv_color_image, "LID-" + str(int(Id)), (max(-10, x1), max(40, y1)), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), thickness=2)
    #         # cv2.imshow("Color Image window", cv_color_image)
    #         # cv2.waitKey(1)
    #         break
        
    # print(detected)
        
    
    
    
    
    if 0 <= x_mouse < cv_aligned_depth_image.shape[1] and 0 <= y_mouse < cv_aligned_depth_image.shape[0]  and detected:
        depth = cv_aligned_depth_image[y_mouse, x_mouse] / 1000.0
        # depth = cv_aligned_depth_image[y_mouse, x_mouse] / depth_scale
        cam_x = (x_mouse - cx) * depth / fx
        cam_y = (y_mouse - cy) * depth / fy
        cam_z = depth
        # print("Aligned Depth at center pixel: ", depth)
        x,y,z = np.dot(Rot, np.array([cam_x,cam_y,cam_z]))
        yaw = np.arctan2(y,x)
        backward_distance = 0.5
        backward_x = x - backward_distance * np.cos(yaw)
        backward_y = y - backward_distance * np.sin(yaw)
        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
        # print(x,y,z)
        # cv2.circle(cv_color_image, (x_mouse, y_mouse), 5, (0, 0, 255), -1)
        # cv2.putText(cv_color_image, f"x={x:.2f}, y={y:.2f}, z={z:.2f}", (x_mouse, y_mouse), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
           
    # 创建 PoseStamped 消息并发布
        if depth != 0:
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "camera_link"  # 设置坐标系
            pose_msg.header.stamp = rospy.Time.now()
            
            pose_msg.pose.position.x = backward_x
            pose_msg.pose.position.y = backward_y
            # pose_msg.pose.position.y = y
            
            pose_msg.pose.orientation.x = quaternion[0]
            pose_msg.pose.orientation.y = quaternion[1]
            pose_msg.pose.orientation.z = quaternion[2]
            pose_msg.pose.orientation.w = quaternion[3]


            pose_pub.publish(pose_msg)
            rospy.loginfo("Published goal pose: x=%f, y=%f, z=%f", x, y, z)
            # global_results_tracker = cv_color_image
#     # 创建tf监听器
#     # listener = tf.TransformListener()

#     # 进行坐标转换
#     # try:
#     #     # 等待tf变换可用
#     #     listener.waitForTransform("map", "camera_link", rospy.Time(0), rospy.Duration(4.0))
#     #     # 转换坐标
#     #     pose_msg_in_map = listener.transformPose("map", pose_msg)

#     #     # 发布目标位
#     #     if depth != 0:
#     #         pose_pub.publish(pose_msg_in_map)
#     # except (tf.LookupException, tf.ProjectionException, tf.TransformException) as e:
#     #     rospy.logerr("Transform error: %s", str(e))

#     # print(f"Published goal pose: x={x}, y={y}, z={z}, quaternion={quaternion}")

    
#     # cv2.setMouseCallback("Color Image window", mouse_callback)
#     # cv2.circle(cv_color_image, (x_mouse, y_mouse), 5, (0, 0, 255), -1)
#     # cv2.putText(cv_color_image, f"x={x:.2f}, y={y:.2f}, z={z:.2f}", (x_mouse, y_mouse), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     # # print(f"x={cam_x:.2f}, y={cam_y:.2f}, z={cam_z:.2f}")
#     # # quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
#     # # decorated_frame = results[0].plot()
#     # cv2.imshow("preview", cv_color_image)
#     # cv2.waitKey(1)
    
    
#     # cv2.imshow("Color Image window", cv_color_image)
#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     cv2.destroyAllWindows()


# 创建锁对象
# image_lock = threading.Lock()
# # 全局事件对象，用于标志线程的运行状态
# tracking_thread_event = threading.Event()
# fsm_thread_event = threading.Event()

# def tracking(color_image,depth_image,depth_scale):
#     current_state = rospy.get_param("current_state", "WAITING")  # 默认值为 "WAITING"

#     # 检查状态是否为 TRACKING
#     if current_state != State.TRACKING:
#         return  # 如果不是 TRACKING，则直接返回，不处理图像

#     detected = False
#     results = model.predict(color_image,imgsz=480,half=True)
#     detections, confarray = extract_detections(results, 0)
#     resultsTracker = tracker.update(detections, confarray, color_image)
#     for x1, y1, x2, y2, Id in resultsTracker:
#         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert position to integers.  
#         # print(f"Id: {Id}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
#         if Id == 1:
#             x_mouse = (x1 + x2) // 2
#             y_mouse = (y1 + y2) // 2
#             detected = True
#             cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
#             cv2.putText(color_image, "LID-" + str(int(Id)), (max(-10, x1), max(40, y1)), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), thickness=2)
#             # cv2.imshow("Color Image window", color_image)
#             # cv2.waitKey(1)
#             break

    
#     if 0 <= x_mouse < depth_image.shape[1] and 0 <= y_mouse < depth_image.shape[0] and detected:
#         # depth = depth_image[y_mouse, x_mouse] / 1000.0
#         depth = depth_image[y_mouse, x_mouse] * depth_scale
#         cam_x = (x_mouse - cx) * depth / fx
#         cam_y = (y_mouse - cy) * depth / fy
#         cam_z = depth
#         # print("Aligned Depth at center pixel: ", depth)
#         x,y,z = np.dot(Rot, np.array([cam_x,cam_y,cam_z]))
#         yaw = np.arctan2(y,x)
#         quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
#         # print(x,y,z)
#         cv2.circle(color_image, (x_mouse, y_mouse), 5, (0, 0, 255), -1)
#         cv2.putText(color_image, f"x={x:.2f}, y={y:.2f}, z={z:.2f}", (x_mouse, y_mouse), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow("color_image", color_image)
#         cv2.waitKey(1)    
#     # 创建 PoseStamped 消息并发布
#         if depth != 0:
#             pose_msg = PoseStamped()
#             pose_msg.header.frame_id = "camera_link"  # 设置坐标系
#             pose_msg.header.stamp = rospy.Time.now()
            
#             pose_msg.pose.position.x = x
#             pose_msg.pose.position.y = y
            
#             pose_msg.pose.orientation.x = quaternion[0]
#             pose_msg.pose.orientation.y = quaternion[1]
#             pose_msg.pose.orientation.z = quaternion[2]
#             pose_msg.pose.orientation.w = quaternion[3]


#             pose_pub.publish(pose_msg)
            
#             rospy.loginfo("Published goal pose: x=%f, y=%f, z=%f", x, y, z)    
# def fsm(color_image_copy):
#     state_publisher.image_callback(color_image_copy)
    
# def track(color_image, depth_image, depth_scale):
#     """ 线程函数，用于跟踪 """
#     tracking_thread_event.set()  # 标记线程为运行状态
#     try:
#         # 在这里调用 tracking 函数
#         tracking(color_image, depth_image, depth_scale)
#     finally:
#         tracking_thread_event.clear()  # 清除状态标志
# def fsm_function(color_image_copy):
#     """ 线程函数，用于状态机 """
#     fsm_thread_event.set()  # 标记线程为运行状态
#     try:
#         # 在这里调用 fsm 函数
#         fsm(color_image_copy)
#     finally:
#         fsm_thread_event.clear()  # 清除状态标志

# def timer_callback(event):
#     # 固定的 x, y, z 值
#     x, y, z = 1, 1, 1
#     yaw = np.arctan2(y, x)
    
#     # 计算朝后方0.5米的位置
#     backward_distance = 1
#     backward_x = x - backward_distance * np.cos(yaw)
#     backward_y = y - backward_distance * np.sin(yaw)
    
#     # 计算四元数
#     quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
    
#     # 创建 PoseStamped 消息
#     pose_msg = PoseStamped()
#     pose_msg.header.frame_id = "camera_link"  # 设置坐标系
#     pose_msg.header.stamp = rospy.Time.now()
    
#     pose_msg.pose.position.x = backward_x
#     pose_msg.pose.position.y = backward_y
#     pose_msg.pose.position.z = z  # 如果需要 z 值，可以设置

#     pose_msg.pose.orientation.x = quaternion[0]
#     pose_msg.pose.orientation.y = quaternion[1]
#     pose_msg.pose.orientation.z = quaternion[2]
#     pose_msg.pose.orientation.w = quaternion[3]

#     # 发布姿态消息
#     pose_pub.publish(pose_msg)
#     rospy.loginfo("Published goal pose: x=%f, y=%f, z=%f", backward_x, backward_y, z)

    

if __name__ == '__main__':
    rospy.init_node('get_image', anonymous=True)
    # state_publisher = StatePublisher()
    # 创建订阅对象
    color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    # depth_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)
    align_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
    
    pose_pub = rospy.Publisher('goal_pose', PoseStamped, queue_size=1)
    # timer = rospy.Timer(rospy.Duration(0.5), timer_callback)
    # 创建模型对象
    model = YOLO("/home/wu/catkin_ws/src/yolov8n_ncnn_model")
 
    ts = message_filters.TimeSynchronizer([color_sub,align_sub], 1)  # 10是缓冲区大小
    ts.registerCallback(image_callback)
    rospy.spin()
    while not rospy.is_shutdown():
        print("spinning")
        # x,y,z = 1,1,1
        # yaw = np.arctan2(y,x)
        # backward_distance = 0.5
        # backward_x = x - backward_distance * np.cos(yaw)
        # backward_y = y - backward_distance * np.sin(yaw)
        # quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
        # pose_msg = PoseStamped()
        # pose_msg.header.frame_id = "camera_link"  # 设置坐标系
        # pose_msg.header.stamp = rospy.Time.now()
        
        # pose_msg.pose.position.x = backward_x
        # pose_msg.pose.position.y = backward_y
        # # pose_msg.pose.position.y = y
        
        # pose_msg.pose.orientation.x = quaternion[0]
        # pose_msg.pose.orientation.y = quaternion[1]
        # pose_msg.pose.orientation.z = quaternion[2]
        # pose_msg.pose.orientation.w = quaternion[3]


        # pose_pub.publish(pose_msg)
        # rospy.loginfo("Published goal pose: x=%f, y=%f, z=%f", x, y, z)
        # rospy.spin()
        # for x1, y1, x2, y2, Id in global_results_tracker:
        #     y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert position to integers.  
        # print(f"Id: {Id}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
        # if Id == 1:
        #     x_mouse = (x1 + x2) // 2
        #     y_mouse = (y1 + y2) // 2
        #     detected = True
            # cv2.rectangle(cv_color_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # cv2.putText(cv_color_image, "LID-" + str(int(Id)), (max(-10, x1), max(40, y1)), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), thickness=2)
            # cv2.imshow("Color Image window", cv_color_image)
        # if global_results_tracker is not None:
        #     cv2.imshow("Color Image window", global_results_tracker)
        #     cv2.waitKey()
        
        
   # import pyrealsense2 as rs
    # import numpy as np
    # import cv2
    
    
    # Configure depth and rgb and infrared streams
    # pipeline = rs.pipeline()
    # config = rs.config()
    # config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    # # config.enable_stream(rs.stream.infrared, 1024, 768, rs.format.y8, 30)
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # profile = pipeline.start(config)
    # # Getting the depth sensor's depth scale (see rs-align example for explanation)
    # depth_sensor = profile.get_device().first_depth_sensor()
    # depth_scale = depth_sensor.get_depth_scale()
    # print("Depth Scale is: " , depth_scale)

    # # We will be removing the background of objects more than
    # #  clipping_distance_in_meters meters away
    # clipping_distance_in_meters = 1 #1 meter
    # clipping_distance = clipping_distance_in_meters / depth_scale
    
    # align_to = rs.stream.color
    # align = rs.align(align_to)
    
    

    
    
    # while not rospy.is_shutdown():
    #     try:
    #         while True:
    #             # Get frameset of color and depth
    #             frames = pipeline.wait_for_frames()
    #             # frames.get_depth_frame() is a 640x360 depth image

    #             # Align the depth frame to color frame
    #             aligned_frames = align.process(frames)

    #             # Get aligned frames
    #             aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    #             color_frame = aligned_frames.get_color_frame()

    #             # Validate that both frames are valid
    #             if not aligned_depth_frame or not color_frame:
    #                 continue
                
    #             depth_image = np.asanyarray(aligned_depth_frame.get_data())
    #             color_image = np.asanyarray(color_frame.get_data())
                
    #             #等比例缩小一半
    #             color_image_copy = cv2.resize(color_image, (0,0), fx=0.5, fy=0.5)
                
                
                
    #             hand_detect_thread = threading.Thread(target=tracking, args=(color_image,depth_image,depth_scale))
    #             color_image_thread = threading.Thread(target=fsm, args=(color_image_copy))

    #             # 启动线程
    #             hand_detect_thread.start()
    #             color_image_thread.start()
                
    #             # image_callback(color_image,depth_image,depth_scale)
                
    #     finally:
    #         pipeline.stop()
    # 创建时间同步对象