#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from ultralytics import YOLO
import torch

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
        self.model = YOLO("/home/wu/Lab/yolov8-deepsort-fast/gesture_ncnn_model")
        # self.model = YOLO("/home/wu/Lab/yolov8-deepsort-fast/handdetect/gesture_ncnn_model")
        # 创建 cv_bridge 实例
        self.bridge = CvBridge()
        self.last_image = None
        self.need_image = True
        self.need_show = False 
        self.count_fist = 0
        self.count_left = 0
        self.count_palm = 0
        self.count_right = 0

        # 创建订阅者
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

        # 创建定时器，频率为 5 Hz
        rospy.Timer(rospy.Duration(1.0 / 5.0), self.timer_callback)

    def image_callback(self, img_msg):
        try:
            # 将ROS图像消息转换为OpenCV格式
            if self.need_image:
                self.need_image = False
                self.last_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("cv_bridge exception: %s", str(e))

    def timer_callback(self, event):
        if self.last_image is not None and self.need_image is False:
            # 在这里进行图像处理，根据图像更新状态
            if self.check_condition(self.last_image):
                # 更新参数服务器上的状态
                rospy.set_param("current_state", self.current_state)
                rospy.loginfo("Updated state: %s", self.current_state)
        self.need_image = True  

    def check_condition(self, image):
        # 实现您自己的条件检测逻辑，返回 True 或 False
        # 增强亮度和对比度
        
        
        results = self.model.predict(image, imgsz=320, half=True)
        image = results[0].plot()
        print(self.current_state)
        if self.need_show:
            cv2.imshow("image", image)  
            cv2.waitKey(1)
        for result in results:
            if len(result.boxes.cls.cpu().numpy()) == 0:
                return False
            for box in result.boxes:
                class_id = int(box.cls)
                if class_id == 0:  # 对应 fist
                    self.count_fist += 1
                    if self.count_fist >= 3:
                        self.current_state = State.TRACKING
                        # 重置其他计数器
                        self.count_left = 0
                        self.count_palm = 0
                        self.count_right = 0
                elif class_id == 1:  # 对应 left
                    self.count_left += 1
                    if self.count_left >= 3:
                        self.current_state = State.LEFT
                        # 重置其他计数器
                        self.count_fist = 0
                        self.count_palm = 0
                        self.count_right = 0
                elif class_id == 2:  # 对应 palm
                    self.count_palm += 1
                    if self.count_palm >= 3:
                        self.current_state = State.WAITING
                        # 重置其他计数器
                        self.count_fist = 0
                        self.count_left = 0
                        self.count_right = 0
                elif class_id == 3:  # 对应 right
                    self.count_right += 1
                    if self.count_right >= 3:
                        self.current_state = State.RIGHT
                        # 重置其他计数器
                        self.count_fist = 0
                        self.count_left = 0
                        self.count_palm = 0
                print(self.current_state)
                # 只有在状态改变时才返回 True
                if self.count_fist >= 3 or self.count_left >= 3 or self.count_palm >= 3 or self.count_right >= 3:
                    return True
                # if class_id == 0:  # 对应 fist
                #     self.current_state = State.TRACKING
                # elif class_id == 1:  # 对应 left
                #     self.current_state = State.LEFT
                # elif class_id == 2:  # 对应 palm
                #     self.current_state = State.WAITING
                # elif class_id == 3:  # 对应 right
                #     self.current_state = State.RIGHT
                # print(self.current_state)
                # return True

if __name__ == '__main__':
    state_publisher = StatePublisher()
    # 保持节点运行
    rospy.spin()
