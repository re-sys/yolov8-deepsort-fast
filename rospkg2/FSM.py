#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from ultralytics import YOLO
import openvino as ov
import torch
import time

class State:
    WAITING = "WAITING"
    TRACKING = "TRACKING"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

class StatePublisher:
    def __init__(self):
        rospy.init_node('state_publisher_node')
        
        self.current_state = State.WAITING
        rospy.set_param("current_state", self.current_state)
        self.model = YOLO("/home/wu/catkin_ws/src/my_pkg/cv_pkg/scripts/gesture_ncnn_model")
        # 创建 cv_bridge 实例
        self.bridge = CvBridge()

        # 创建订阅者
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

    def image_callback(self, img_msg):
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            
            # 在这里进行图像处理，根据图像更新状态
            if cv_image is None:
                print("cv_image is None")
                return

            # 示例图像处理逻辑（您可以根据需要实现自己的条件）
            if self.check_condition(cv_image):
                # 更新参数服务器上的状态
                rospy.set_param("current_state", self.current_state)
                rospy.loginfo("Updated state: %s", self.current_state)

        except CvBridgeError as e:
            rospy.logerr("cv_bridge exception: %s", str(e))

    def check_condition(self, image):
        # 实现您自己的条件检测逻辑，返回 True 或 False
        # 这是一个示例：
        # return True if some_condition else False
        results = self.model.predict(image,imgsz=320,half=True)
        # results[0].plot()
        # print(results)
        # decorated_frame = results[0].plot()
        # cv2.imshow("preview", decorated_frame)
        # cv2.waitKey(1)
        for result in results:
            if len(result.boxes.cls.cpu().numpy()) == 0:
                return False
            for box in result.boxes:
                class_id = int(box.cls)
                if class_id == 0:  # 对应 fist
                    self.current_state = State.TRACKING
                elif class_id == 1:  # 对应 left
                    self.current_state = State.LEFT
                elif class_id == 2:  # 对应 palm
                    self.current_state = State.WAITING
                elif class_id == 3:  # 对应 right
                    self.current_state = State.RIGHT
                print(self.current_state)
                return True

if __name__ == '__main__':
    # 定义全局变量
    # parent_dir = "/home/wu/Lab/yolov8-deepsort-fast/handdetect/"
    # # cv2.namedWindow("preview")
    # modelname="gesture.pt"
    # model = YOLO("/home/wu/catkin_ws/src/my_pkg/cv_pkg/scripts/gesture_ncnn_model")
    
    
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
          # wait for two parallel frames
    state_publisher = StatePublisher()
    # while not rospy.is_shutdown():
    #     try:
    #         while True:
    #             frame = pipeline.wait_for_frames() 
    #             color_frame = frame.get_color_frame()
    #             color_image = np.asanyarray(color_frame.get_data())
    #             # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    #             print(color_image.shape)
    #             state_publisher.image_callback(color_image)
    #     finally:
    #         pipeline.stop()
    # 创建状态发布节点
    
    
    # 保持节点运行
    rospy.spin()
