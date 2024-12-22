#!/usr/bin/env python

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped
import tf
from std_msgs.msg import String
from geometry_msgs.msg import Quaternion

class NavClient:
    def __init__(self):
        self.ac = actionlib.SimpleActionClient('move_base', MoveBaseAction)  # 初始化行动客户端
        self.goal_pose_sub = rospy.Subscriber('goal_pose', PoseStamped, self.goal_pose_callback)  # 订阅目标位姿
        self.current_goal = PoseStamped()
        self.has_new_goal = False
        self.count = 0

        rospy.Timer(rospy.Duration(1.0), self.state_check_callback)  # 创建周期性检查状态的定时器
        
        self.wait_for_server()  # 等待连接到行动服务器

    def wait_for_server(self):
        rospy.loginfo("Waiting for move_base action server...")
        self.ac.wait_for_server(rospy.Duration(5.0))
        rospy.loginfo("Connected to move_base action server.")

    def goal_pose_callback(self, msg):
        self.current_goal = msg  # 保存接收到的目标
        self.has_new_goal = True  # 设置标志为 True
        rospy.loginfo("Received new goal")

    def state_check_callback(self, event):
        try:
            current_state = rospy.get_param("current_state")  # 从参数服务器获取状态
            rospy.loginfo("Current state: %s", current_state)
        except KeyError:
            rospy.logwarn("Current state not found in parameter server")
            return
        if current_state == "WAITING":
            rospy.loginfo("Current state is WAITING. Cancelling the goal...")
            self.ac.cancel_all_goals()  # 取消所有目标

        elif current_state == "LEFT":   
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "base_link"  # 设置为 map 帧
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = 0.0
            goal.target_pose.pose.position.y = 0.0
            goal.target_pose.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, 20 * (3.14159 / 180)))  # 20 度
            self.ac.send_goal(goal)
            self.ac.wait_for_result(rospy.Duration(1.0))
            rospy.set_param("current_state", "WAITING")  # 更新参数服务器状态

        elif current_state == "RIGHT":
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "base_link"  # 设置为 map 帧
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = 0.0
            goal.target_pose.pose.position.y = 0.0
            goal.target_pose.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, -20 * (3.14159 / 180)))  # -20 度

            rospy.loginfo("Sending goal to rotate -20 degrees")
            self.ac.send_goal(goal)
            self.ac.wait_for_result(rospy.Duration(1.0))
            rospy.set_param("current_state", "WAITING")  # 更新参数服务器状态

        elif current_state == "TRACKING":
            goal = MoveBaseGoal()
            if self.has_new_goal:
                goal.target_pose.header.frame_id = "camera_link" # 匹配当前目标帧
                goal.target_pose.header.stamp = rospy.Time.now()
                goal.target_pose.pose = self.current_goal.pose

                rospy.logwarn("Sending goal from received message")
                self.ac.send_goal(goal)
                self.has_new_goal = False  # 重置标志
            else:
                rospy.loginfo("No new goal received, start rolling to find new goal")
                self.count += 1
                if self.count > 10:
                    self.count = 0
                    goal.target_pose.header.frame_id = "base_link"  # 设置为 map 帧
                    goal.target_pose.header.stamp = rospy.Time.now()
                    goal.target_pose.pose.position.x = 0.0
                    goal.target_pose.pose.position.y = 0.0
                    goal.target_pose.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, 180 * (3.14159 / 180)))  # 360 度
                    rospy.loginfo("Sending goal to rotate 360 degrees")
                    self.ac.send_goal(goal)
                    self.ac.wait_for_result(rospy.Duration(1.0))

if __name__ == "__main__":
    rospy.init_node("nav_client")  # 初始化节点
    nav_client = NavClient()  # 创建 NavClient 对象
    rospy.spin()  # 保持节点运行
