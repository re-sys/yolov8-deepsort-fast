import rospy
from geometry_msgs.msg import Pose2D, Twist
import math
from FSM import State
class LocalPlanNode:
    def __init__(self):
        rospy.init_node('local_plan_node', anonymous=True)

        # 订阅goalpose话题
        self.goalpose_sub = rospy.Subscriber('goalpose', Pose2D, self.goalpose_callback)

        # 发布cmd_vel话题
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

        # 初始化目标点
        self.goalpose = None
        self.no_goal_count = 0 
        self.process = False  # 目标点是否已处理
        # 标志变量，表示是否已经处理了当前的目标点
        self.goal_reached = True
        self.max_linear_speed = 0.5  # 线速度的最大值
        self.max_angular_speed = 1.0  # 角速度的最大值

        # 控制循环的频率
        self.rate = rospy.Rate(10)  # 10 Hz
        rospy.loginfo("LocalPlanNode started")

    def goalpose_callback(self, msg):
        # 当接收到新的目标点时更新
        self.goalpose = msg
        self.process = False  # 新的目标点到达，设置为未处理状态
        self.no_goal_count = 0  # 重置计数器

    def run(self):
        while not rospy.is_shutdown():
            
            # 获取当前机器人位置（这里假设通过其他方式获取，例如定位系统）
            try:
                current_state = rospy.get_param("current_state")  # 从参数服务器获取状态
                # rospy.loginfo("Current state: %s", current_state)
            except KeyError:
                rospy.logwarn("Current state not found in parameter server")
                return
            if current_state == "WAITING":
                rospy.loginfo("Current state is WAITING. Cancelling the goal...")
            elif current_state == "TRACKING":
                rospy.loginfo("Current state is TRACKING. Tracking the goal...")
                self.track_goal()
            elif current_state == "LEFT":
                rospy.loginfo("Current state is LEFT. Rotating left...")
                self.rotate_left(self.max_angular_speed)
            elif current_state == "RIGHT":
                rospy.loginfo("Current state is RIGHT. Rotating right...")
                self.rotate_right(self.max_angular_speed)
            # 控制循环频率
            self.rate.sleep()

    def calculate_distance(self, x1, y1, x2, y2):
        # 计算两个点之间的距离
        if x2>=0:
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        else:
            return 0
    def rotate_left(self, angular_speed):
        """
        发布向左旋转的Twist消息
        :param angular_speed: 角速度大小
        """
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = angular_speed
        rospy.loginfo("Rotating left with angular speed: %f", angular_speed)
        self.cmd_vel_pub.publish(twist_msg)

    def rotate_right(self, angular_speed):
        """
        发布向右旋转的Twist消息
        :param angular_speed: 角速度大小
        """
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = -angular_speed
        rospy.loginfo("Rotating right with angular speed: %f", angular_speed)
        self.cmd_vel_pub.publish(twist_msg)
        
    def track_goal(self):
        if self.goalpose is not None and not self.process:
            current_x = 0.0  # 当前x坐标
            current_y = 0.0  # 当前y坐标
            current_theta = 0.0  # 当前角度

            # 计算目标点与当前点之间的距离和角度差
            distance = self.calculate_distance(current_x, current_y, self.goalpose.x, self.goalpose.y)
            angle_diff = self.calculate_angle_difference(current_theta, self.goalpose.x, self.goalpose.y)
            # angle_diff = -self.goalpose.y

            # 创建Twist消息
            twist_msg = Twist()

            
            twist_msg.linear.x, twist_msg.angular.z = self.limit_velocity(distance, angle_diff)
            rospy.loginfo("distance: %f, angle_diff: %f", distance, angle_diff)
            
            # 发布Twist消息
            rospy.loginfo("cmd_vel: %f, %f", twist_msg.linear.x, twist_msg.angular.z)
            self.cmd_vel_pub.publish(twist_msg)
            self.process = True  # 设置为目标点已处理状态
        elif self.process:
            # 如果已经处理了当前的目标点，发布停止指令
            if self.no_goal_count <3 :
                self.no_goal_count += 1
                twist_msg = Twist()
                twist_msg.linear.x, twist_msg.angular.z= self.get_linx_angz()
                rospy.loginfo("cmd_vel: %f, %f", twist_msg.linear.x, twist_msg.angular.z)
                self.cmd_vel_pub.publish(twist_msg)
            else:
                stop_msg = Twist()
                stop_msg.linear.x = 0.0
                stop_msg.angular.z = 0.0
                rospy.loginfo("goal_reached")
                self.cmd_vel_pub.publish(stop_msg)
        self.goal_reached = True  # 设置为目标点已处理状态
    def calculate_angle_difference(self, current_theta, goal_x, goal_y):
        # 计算当前角度与目标点角度之间的差值
        print(f"goal_x: {goal_x}, goal_y: {goal_y}")
        if goal_x >=0:
            goal_angle = math.atan2(goal_y, goal_x)  # 注意：这里应该是 math.atan2(goal_y, goal_x)
        else:
            goal_angle = math.atan2(-goal_y, -goal_x)
        angle_diff = goal_angle - current_theta

        # 将角度差限制在[-pi, pi]之间
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        return angle_diff
    def limit_velocity(self, distance, angle_diff):
        """
        根据距离和角度差限制线速度和角速度的绝对值
        :param distance: 目标点与当前点之间的距离
        :param angle_diff: 目标点与当前点之间的角度差
        :return: 限制后的线速度和角速度
        """
        max_linear_x = 0.2  # 假设最大线速度为0.2 m/s
        max_angular_z = 0.5  # 假设最大角速度为0.5 rad/s

        linear_x = distance * 0.3
        angular_z = angle_diff * 1

        # 限制线速度和角速度的绝对值
        linear_x = min(abs(linear_x), max_linear_x) * (1 if linear_x >= 0 else -1)
        angular_z = min(abs(angular_z), max_angular_z) * (1 if angular_z >= 0 else -1)

        return linear_x, angular_z
    def get_linx_angz(self):
        current_x = 0.0  # 当前x坐标
        current_y = 0.0  # 当前y坐标
        current_theta = 0.0  # 当前角度

        # 计算目标点与当前点之间的距离和角度差
        distance = self.calculate_distance(current_x, current_y, self.goalpose.x, self.goalpose.y)
        angle_diff = self.calculate_angle_difference(current_theta, self.goalpose.x, self.goalpose.y)
        return self.limit_velocity(distance, angle_diff)
        
if __name__ == '__main__':
    try:
        node = LocalPlanNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
