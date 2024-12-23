#!/usr/bin/env python

import rospy
import math
from geometry_msgs.msg import Pose2D, Twist

class FollowGoalNode:
    def __init__(self):
        rospy.init_node('follow_goal_node', anonymous=True)

        # 订阅goalpose话题
        self.goalpose_sub = rospy.Subscriber('goalpose', Pose2D, self.goalpose_callback)

        # 发布cmd_vel话题
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

        # 初始化目标点
        self.goalpose = None

        # 控制循环的频率
        self.rate = rospy.Rate(10)  # 10 Hz

    def goalpose_callback(self, msg):
        # 当接收到新的目标点时更新
        self.goalpose = msg

    def run(self):
        while not rospy.is_shutdown():
            if self.goalpose is not None:
                # 获取当前机器人位置（这里假设通过其他方式获取，例如定位系统）
                current_x = 0.0  # 当前x坐标
                current_y = 0.0  # 当前y坐标
                current_theta = 0.0  # 当前角度

                # 计算目标点与当前点之间的距离和角度差
                distance = self.calculate_distance(current_x, current_y, self.goalpose.x, self.goalpose.y)
                angle_diff = self.calculate_angle_difference(current_theta, self.goalpose.x, self.goalpose.y)

                # 创建Twist消息
                twist_msg = Twist()

                # 设置线速度和角速度
                twist_msg.linear.x = distance * 0.1  # 线速度与距离成正比，这里乘以一个小数来限制速度
                twist_msg.angular.z = angle_diff * 0.5  # 角速度与角度差成正比，这里乘以一个小数来限制速度

                # 发布Twist消息
                self.cmd_vel_pub.publish(twist_msg)

            # 控制循环频率
            self.rate.sleep()

    def calculate_distance(self, x1, y1, x2, y2):
        # 计算两个点之间的距离
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_angle_difference(self, current_theta, goal_x, goal_y):
        # 计算当前角度与目标点角度之间的差值
        goal_angle = math.atan2(goal_y, goal_x)
        angle_diff = goal_angle - current_theta

        # 将角度差限制在[-pi, pi]之间
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        return angle_diff

if __name__ == '__main__':
    try:
        node = FollowGoalNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
