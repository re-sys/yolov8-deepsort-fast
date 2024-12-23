import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import rospy
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from FSM import State
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose2D
fx = 897.6827392578125
fy = 898.0628051757812
cx = 649.2904052734375
cy = 376.3505859375
Rot = np.array([[0,0,1],[-1,0,0],[0,-1,0]])
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
# model = YOLO("/home/wu/catkin_ws/src/yolov8n_ncnn_model")





class ObjectTrackerNode:
    def __init__(self, cx, cy, fx, fy, Rot, model_path,use_pipeline=True):
        """
        初始化ROS节点和相机参数
        :param cx: 相机中心x坐标
        :param cy: 相机中心y坐标
        :param fx: 相机焦距x
        :param fy: 相机焦距y
        :param Rot: 旋转矩阵
        :param model_path: 模型路径
        """
        rospy.init_node('object_tracker_node', anonymous=True)
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.Rot = Rot
        self.model = YOLO(model_path)
        self.needshow=False
        if use_pipeline:
            # Configure depth and rgb streams
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            self.profile = self.pipeline.start(self.config)
            
            # Getting the depth sensor's depth scale
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print("Depth Scale is: ", self.depth_scale)

            # Align the depth frame to color frame
            self.align_to = rs.stream.color
            self.align = rs.align(self.align_to)
            self.timer_callback(None)
        else:
        # 创建订阅者
            self.bridge = CvBridge()
            self.color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
            self.align_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
            
            # 创建时间同步器
            self.ts = message_filters.TimeSynchronizer([self.color_sub, self.align_sub], 2)  # 10是缓冲区大小
            self.ts.registerCallback(self.image_callback)
            
        
        # 创建发布者
        # self.pose_pub = rospy.Publisher('goal_pose', PoseStamped, queue_size=1)
        self.goalpose_pub = rospy.Publisher('goalpose', Pose2D, queue_size=10)

        # self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        

    def timer_callback(self, event):
        """
        定时器回调函数，每100毫秒执行一次
        :param event: 定时器事件
        """
        
        # 这里可以添加需要定期执行的任务，例如检查状态
        frames = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return
        
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        results = self.model.predict(color_image,imgsz=320,half=True,classes=[0],iou=0.2)
        color_image = results[0].plot()
        boxes = results[0].boxes
        cx, cy = self.get_center(boxes)
        if cx:
            min_depth_value, min_depth_x, min_depth_y = self.get_region(depth_image,cx,cy)
            min_depth_value = min_depth_value*self.depth_scale
            x, y, z, yaw, backward_x, backward_y = self.projection(cx, cy, min_depth_value,backward_distance=1.5)
            # if min_depth_value <= 1 or min_depth_value>=2:
            #     return
        # 绘制最小深度值点
            cv2.circle(color_image, (min_depth_x, min_depth_y), 5, (0, 0, 255), -1)
            cv2.putText(color_image, f"({x:.2f}, {y:.2f}, {z:.2f})", (min_depth_x + 10, min_depth_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(color_image, f"Min Depth: {min_depth_value:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.color_image = color_image
            if 2>min_depth_value>1.5:
                
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "camera_link"
                pose_msg.pose.position.x = backward_x
                pose_msg.pose.position.y = backward_y
                pose_msg.pose.position.z = z
                pose_msg.pose.orientation.x = 0
                pose_msg.pose.orientation.y = 0
                pose_msg.pose.orientation.z = np.sin(yaw/2)
                pose_msg.pose.orientation.w = np.cos(yaw/2)
                self.pose_pub.publish(pose_msg)
                print(f"({backward_x:.2f}, {backward_y:.2f}, {z:.2f}), {yaw:.2f} yaw")
            else:
                print("min_depth_value is too small, return")
                return
            
    def get_center(self, boxes):
        """
        计算检测到对象的中心点坐标
        :param boxes: 边界框列表
        :return: 中心点坐标 (center_x, center_y)
        """
        for box in boxes:
            r = box.xyxy[0].tolist()  # 获取边界框的x1, y1, x2, y2坐标
            if len(r) == 4 and box.conf > 0.5:
                x1, y1, x2, y2 = r
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                rospy.loginfo(f"对象中心点坐标: ({center_x}, {center_y})")
                return center_x, center_y
        else:
            rospy.loginfo("检测不到对象")
            return 0, 0

    def projection(self, center_x, center_y, depth_value,backward_distance=1):
        """
        计算相机坐标系下的点坐标
        :param center_x: 对象中心点x坐标
        :param center_y: 对象中心点y坐标
        :param depth_value: 深度值
        :return: 三维坐标 (x, y, z) 和偏航角 yaw 以及后退方向坐标 (backward_x, backward_y)
        """
        x = (center_x - self.cx) * depth_value / self.fx
        y = (center_y - self.cy) * depth_value / self.fy
        z = depth_value
        x, y, z = np.dot(self.Rot, np.array([x, y, z]))
        yaw = np.arctan2(y, x)
        
        backward_x = x - backward_distance * np.cos(yaw)
        backward_y = y - backward_distance * np.sin(yaw)
        return x, y, z, yaw, backward_x, backward_y

    def get_region(self, depth_image, center_x, center_y, kernel_size=5):
        """
        提取中心点周围区域的最小深度值及其位置
        :param depth_image: 深度图像
        :param center_x: 中心点x坐标
        :param center_y: 中心点y坐标
        :param kernel_size: 核心区域大小，默认为5
        :return: 最小深度值及其位置 (min_depth_value, min_depth_x, min_depth_y)
        """
        half_kernel = kernel_size // 2

        # 确保中心点周围的区域在图像边界内
        if (center_x - half_kernel >= 0 and center_x + half_kernel < depth_image.shape[1] and
            center_y - half_kernel >= 0 and center_y + half_kernel < depth_image.shape[0]):
            # 提取核心区域的深度值
            kernel = depth_image[center_y - half_kernel:center_y + half_kernel + 1,
                                center_x - half_kernel:center_x + half_kernel + 1]
            # 过滤掉零值
            non_zero_kernel = kernel[kernel != 0]

            # 检查是否没有非零值
            if non_zero_kernel.size == 0:
                rospy.loginfo("核心区域中没有非零深度值")
                return 0, 0, 0

            min_depth_value = np.min(non_zero_kernel)
            min_depth_position = np.argmin(non_zero_kernel)
            min_depth_x = center_x - half_kernel + min_depth_position % kernel_size
            min_depth_y = center_y - half_kernel + min_depth_position // kernel_size
            # rospy.loginfo(f"最小深度值: {min_depth_value}, 位置: ({min_depth_x}, {min_depth_y})")
            return min_depth_value, min_depth_x, min_depth_y
        else:
            rospy.loginfo("核心区域超出了图像边界")
            return 0, 0, 0

    def image_callback(self, color_msg, depth_msg):
        """
        图像回调函数，处理彩色图像和深度图像
        :param color_msg: 彩色图像消息
        :param depth_msg: 对齐后的深度图像消息
        """
        current_state = rospy.get_param("current_state", "WAITING")  # 默认值为 "WAITING"

    # 检查状态是否为 TRACKING
        if current_state != State.TRACKING:
            print("Not in TRACKING state, return")
            return  # 如果不是 TRACKING，则直接返回，不处理图像 
        # 这里假设我们已经有一个方法来将ros图像消息转换为numpy数组
        try:
            color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")  # 假设深度图是直接以浮点格式传递
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert image: {e}")
            return
        # color_image = self._ros_image_to_numpy(color_msg)
        # depth_image = self._ros_image_to_numpy(depth_msg)
        results = self.model.predict(color_image,imgsz=320,half=True,classes=[0],iou=0.2)
        color_image = results[0].plot()
        boxes = results[0].boxes
        cx, cy = self.get_center(boxes)
        if cx:
            min_depth_value, min_depth_x, min_depth_y = self.get_region(depth_image,cx,cy)
            min_depth_value = min_depth_value/1000
            x, y, z, yaw, backward_x, backward_y = self.projection(cx, cy, min_depth_value,backward_distance=1)
            # if min_depth_value <= 1 or min_depth_value>=2:
            #     return
        # 绘制最小深度值点
            if self.needshow:
                cv2.circle(color_image, (min_depth_x, min_depth_y), 5, (0, 0, 255), -1)
                cv2.putText(color_image, f"({x:.2f}, {y:.2f}, {z:.2f})", (min_depth_x + 10, min_depth_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(color_image, f"Min Depth: {min_depth_value:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('color_image', color_image)
                cv2.waitKey(1)
            # self.color_image = color_image
            # print(f"({x:.2f}, {y:.2f}, {z:.2f}), {yaw:.2f}")
            if min_depth_value>0.5:
                pose2d_msg = Pose2D()
                pose2d_msg.x = backward_x
                pose2d_msg.y = backward_y
                rospy.loginfo(f"({backward_x:.2f}, {backward_y:.2f})")
                # 发布Pose2D消息
                self.goalpose_pub.publish(pose2d_msg)
            else:
                rospy.loginfo(f"min_depth_value is{min_depth_value}, return")
                return
                # 发布PoseStamped消息
                # pose_msg = PoseStamped()
                # pose_msg.header.stamp = rospy.Time.now()  

            

            # if 2>min_depth_value>1.5:
                
            #     pose_msg = PoseStamped()
            #     pose_msg.header.stamp = rospy.Time.now()
            #     pose_msg.header.frame_id = "camera_link"
            #     pose_msg.pose.position.x = backward_x
            #     pose_msg.pose.position.y = backward_y
            #     pose_msg.pose.position.z = z
            #     pose_msg.pose.orientation.x = 0
            #     pose_msg.pose.orientation.y = 0
            #     pose_msg.pose.orientation.z = np.sin(yaw/2)
            #     pose_msg.pose.orientation.w = np.cos(yaw/2)
            #     self.pose_pub.publish(pose_msg)
            #     print(f"({backward_x:.2f}, {backward_y:.2f}, {z:.2f}), {yaw:.2f} yaw")
            # else:
            #     print("min_depth_value is too small, return")
                # return
       

    def _ros_image_to_numpy(self, ros_image):
        """
        将ROS图像消息转换为numpy数组
        :param ros_image: ROS图像消息
        :return: numpy数组
        """
        # 这里假设图像格式是16UC1（16位单通道），如果是其他格式需要相应调整
        if ros_image.encoding == "16UC1":
            depth_image = np.frombuffer(ros_image.data, dtype=np.uint16).reshape(ros_image.height, ros_image.width)
        elif ros_image.encoding == "rgb8":
            color_image = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(ros_image.height, ros_image.width, 3)
            return color_image
        else:
            rospy.logerr(f"不支持的图像编码: {ros_image.encoding}")
            return None

    def _yaw_to_quaternion(self, yaw):
        """
        将偏航角转换为四元数
        :param yaw: 偏航角
        :return: 四元数
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        return [0, 0, sy, cy]

if __name__ == '__main__':
    # 假设我们已经有了相机参数
    fx = 897.6827392578125
    fy = 898.0628051757812
    cx = 649.2904052734375
    cy = 376.3505859375
    Rot = np.array([[0,0,1],[-1,0,0],[0,-1,0]])

    # 模型路径
    model_path = "/home/wu/catkin_ws/src/yolov8n_ncnn_model"

    # 创建ObjectTrackerNode实例
    tracker_node = ObjectTrackerNode(cx, cy, fx, fy, Rot, model_path,use_pipeline=False)
    rospy.spin()
    
    # while True:
    #     try:
    #         tracker_node.timer_callback(None)
    #         color_image = tracker_node.color_image
    #         # print("color_image:",color_image.shape)
    #         cv2.imshow('color_image', color_image)
    #         key = cv2.waitKey(1)
    #         # Press esc or 'q' to close the image window
    #         if key & 0xFF == ord('q') or key == 27:
    #             cv2.destroyAllWindows()
    #             break
    #     except Exception as e:
    #         print(e)
    # 保持节点运行
    


# 使用示例
# 假设我们已经有了相机参数

# tracker = ObjectTracker(cx, cy, fx, fy, Rot)

# 假设我们已经有了边界框列表 `boxes` 和深度图像 `depth_image`
# boxes = [...]
# depth_image = [...]

# 计算对象中心点坐标
# center_x, center_y = tracker.get_center(boxes)

# 计算投影
# x, y, z, yaw, backward_x, backward_y = tracker.projection(center_x, center_y, depth_value)

# 获取区域最小深度值及其位置
# min_depth_value, min_depth_x, min_depth_y = tracker.get_region(depth_image, center_x, center_y)

def get_center(boxes):
    for box in boxes:
            r = box.xyxy[0].tolist()  # 获取边界框的x1, y1, x2, y2坐标
            if len(r) == 4 and box.conf>0.5:
                x1, y1, x2, y2 = r
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                # print(f"对象中心点坐标: ({center_x}, {center_y})")
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
        # print(f"最小深度值: {min_depth_value}, 位置: ({min_depth_x}, {min_depth_y})")
        return min_depth_value, min_depth_x, min_depth_y
    else:
        print("中心点周围的5x5区域超出了图像边界")
        return 0, 0, 0
# try:
#     tracker = ObjectTracker(cx, cy, fx, fy, Rot)
#     while True:
#         # Get frameset of color and depth
#         frames = pipeline.wait_for_frames()
#         # frames.get_depth_frame() is a 640x360 depth image

#         # Align the depth frame to color frame
#         aligned_frames = align.process(frames)

#         # Get aligned frames
#         aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
#         color_frame = aligned_frames.get_color_frame()

#         # Validate that both frames are valid
#         if not aligned_depth_frame or not color_frame:
#             continue
#         flag = True
#         depth_image = np.asanyarray(aligned_depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())
#         results = model.predict(color_image,imgsz=320,half=True,classes=[0],iou=0.2)
#         boxes = results[0].boxes
#         color_image = results[0].plot()
#         center_x,center_y = get_center(boxes)
#         if center_x==0:
#             continue
#         min_depth_value, min_depth_x, min_depth_y = get_region(depth_image,center_x,center_y)
#         min_depth_value = min_depth_value * depth_scale
#         projection_result = projection(center_x,center_y,min_depth_value)
#         x,y,z = projection_result[:3]
#         print(f"相机坐标系下的点坐标: ({projection_result[0]:.2f}, {projection_result[1]:.2f}, {projection_result[2]:.2f}), 偏航角: {projection_result[3]:.2f}, 后方距离: {projection_result[4]:.2f}, {projection_result[5]:.2f}")
#         if min_depth_value <= 1 or min_depth_value>=2:
#             continue
#         # 绘制最小深度值点
#         cv2.circle(color_image, (min_depth_x, min_depth_y), 5, (0, 0, 255), -1)
#         cv2.putText(color_image, f"({x:.2f}, {y:.2f}, {z:.2f})", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.putText(color_image, f"Min Depth: {min_depth_value:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow('color_image', color_image)
#         key = cv2.waitKey(1)
#         # Press esc or 'q' to close the image window
#         if key & 0xFF == ord('q') or key == 27:
#             cv2.destroyAllWindows()
#             break
        
#         # frame = reaults[0].plot()
#         # cv2.imshow('color_image', frame)

#         # cv2.imshow('depth_image', depth_image)
#         # key = cv2.waitKey(1)
#         # Press esc or 'q' to close the image window
#         # if key & 0xFF == ord('q') or key == 27:
#         #     cv2.destroyAllWindows()
#         #     break
# except Exception as e:
#     print(e)
#     pass
# finally:
#     pipeline.stop()
