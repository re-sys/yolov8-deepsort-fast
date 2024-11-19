import tempfile
from pathlib import Path
import numpy as np
import cv2 # opencv-python
from ultralytics import YOLO
import time

# import time

import deep_sort.deep_sort.deep_sort as ds


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

# 视频处理
def detect_and_track(input_path: str, output_path: str, detect_class: int, model, tracker) -> Path:
    """
    处理视频，检测并跟踪目标。
    - input_path: 输入视频文件的路径。
    - output_path: 处理后视频保存的路径。
    - detect_class: 需要检测和跟踪的目标类别的索引。
    - model: 用于目标检测的模型。
    - tracker: 用于目标跟踪的模型。
    """
    frame = cv2.imread("/home/wu/Lab/yolov8-deepsort-fast/handdetect/train_img1/5.mp4_frame6.jpg")
    results = model.predict(frame)
    detections, confarray = extract_detections(results, detect_class)
    resultsTracker = tracker.update(detections, confarray, frame)
    
    frame = cv2.imread("/home/wu/Lab/yolov8-deepsort-fast/handdetect/train_img1/5.mp4_frame6.jpg")
    results = model.predict(frame)
    detections, confarray = extract_detections(results, detect_class)
    resultsTracker = tracker.update(detections, confarray, frame)
    
    frame = cv2.imread("/home/wu/Lab/yolov8-deepsort-fast/handdetect/train_img1/5.mp4_frame6.jpg")
    results = model.predict(frame)
    detections, confarray = extract_detections(results, detect_class)
    resultsTracker = tracker.update(detections, confarray, frame)
    for x1, y1, x2, y2, Id in resultsTracker:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert position to integers.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(frame, "LID-" + str(int(Id)), (max(-10, x1), max(40, y1)), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), thickness=2)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    
    cap = cv2.VideoCapture(input_path)  # 使用OpenCV打开视频文件。
    if not cap.isOpened():  # 检查视频文件是否成功打开。
        print(f"Error opening video file {input_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 获取视频的分辨率（宽度和高度）。
    
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video = cv2.VideoWriter(output_path, fourcc, fps, size) 
    # 对每一帧图片进行读取和处理

 
    fps = 0  # Initialize a variable for frames per second
    frame_count = 0  # Count the number of frames processed
    start_time = time.time()  # Start time
            # 假设 center_points 是一个列表，用于存储中心点的坐标
    tracker_points = {}

    # 假设 max_points 是最大的点数，比如30
    max_points = 30
    # 定义颜色映射字典
    colors = {
        0: (255, 0, 0),    # 红色
        1: (0, 255, 0),    # 绿色
        2: (0, 0, 255),    # 蓝色
        3: (255, 255, 0),  # 青色
        4: (255, 0, 255),  # 品红
        # 可以继续添加更多编号对应的颜色
    }

    while True:
        success, frame = cap.read()  # Read video frame by frame.
        if not success:
            break
    
        results = model.predict(frame)
        # frame_org = results[0].plot()
        detections, confarray = extract_detections(results, detect_class)
        resultsTracker = tracker.update(detections, confarray, frame)

   



# Continue with the rest of the processing...
 
        for x1, y1, x2, y2, Id in resultsTracker:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert position to integers.

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            if Id not in tracker_points:
                tracker_points[Id] = []
            # 只保留最近的30个点
            tracker_points[Id].append((center_x, center_y))

    # 只保留最近的30个点
            if len(tracker_points[Id]) > max_points:
                tracker_points[Id].pop(0)

            # 绘制当前tracker的轨迹
            for i in range(1, len(tracker_points[Id])):
                cv2.line(frame, tracker_points[Id][i - 1], tracker_points[Id][i], (0, 255, 0), 2)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # 获取对应的颜色，如果Id超出了颜色映射的范围，则使用默认颜色
            color = colors.get(Id, (255, 255, 255))  # 默认白色

            # 绘制矩形框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            # 将置信度显示在矩形框上
            # cv2.putText(frame, str(round(confarray[Id], 2)), (max(-10, x1), max(40, y1)), fontScale=1.5, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), thickness=2)
            cv2.putText(frame, "LID-" + str(int(Id)), (max(-10, x1), max(40, y1)), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=color ,thickness=2)
            
        # # Update frame count
        frame_count += 1
    
        # Calculate FPS every second
        current_time = time.time()
        if current_time - start_time >= 1.0:  # Update if a second has passed
            fps = frame_count
            frame_count = 0  # Reset frame count
            start_time = current_time  # Reset start time
    
        # # Display the FPS on the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), fontScale=1.5, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), thickness=2)
        # # print consuming time
        # combined_frame = np.hstack((frame_org, frame))  # 横向拼接两个帧

        # cv2.imshow("Combined Frame", combined_frame)  # 显示合并后的帧
        cv2.imshow("frame", frame)  # Show the current frame.
        output_video.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop on 'q' key press.
            break
    frame = cv2.imread("/home/wu/Lab/yolov8-deepsort-fast/handdetect/train_img1/5.mp4_frame6.jpg")
    results = model.predict(frame)
    detections, confarray = extract_detections(results, detect_class)
    resultsTracker = tracker.update(detections, confarray, frame)
    frame = cv2.imread("/home/wu/Lab/yolov8-deepsort-fast/handdetect/train_img1/5.mp4_frame6.jpg")
    results = model.predict(frame)
    detections, confarray = extract_detections(results, detect_class)
    resultsTracker = tracker.update(detections, confarray, frame)
    frame = cv2.imread("/home/wu/Lab/yolov8-deepsort-fast/handdetect/train_img1/5.mp4_frame6.jpg")
    results = model.predict(frame)
    detections, confarray = extract_detections(results, detect_class)
    resultsTracker = tracker.update(detections, confarray, frame)
    for x1, y1, x2, y2, Id in resultsTracker:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert position to integers.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(frame, "LID-" + str(int(Id)), (max(-10, x1), max(40, y1)), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), thickness=2)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    cap.release()  # Release video file.
    output_video.release()
    cv2.destroyAllWindows()  # Close all windows.
        
    
    return -1


if __name__ == "__main__":
    # 指定输入视频的路径。
    ######
    # input_path = "/home/wu/Lab/yolov8-deepsort-fast/handdetect/testmp4/zoulang2.mp4"  ######
    input_path = "/home/wu/Desktop/video/xuexiao2.mp4"  ######
    parent_dir = "/home/wu/Lab/yolov8-deepsort-fast/"
    # 输出文件夹，默认为系统的临时文件夹路径
    output_path = parent_dir + "output.avi"  # 创建一个临时目录用于存放输出视频。

    # 加载yoloV8模型权重
    import openvino as ov
    import torch
    
    core = ov.Core()
    model = YOLO("yolov8n.pt")
    model.to("cpu")
    # IMAGE_PATH="/home/wu/my_ws/src/yolo_pub/yolo_pub/data/coco_bike.jpg"
    img = cv2.imread("demo.png")
    res = model(img)
    
    # 加载OpenVINO模型
    det_model_path =parent_dir + "yolov8n_openvino_model/yolov8n.xml"
    det_ov_model = core.read_model(det_model_path)
    
    ov_config = {}
    device = "CPU"
    
    compiled_model = core.compile_model(det_ov_model, device, ov_config)
    def infer(*args):
        result = compiled_model(args)
        return torch.from_numpy(result[0])
    model.predictor.model.pt = False
    model.predictor.inference = infer
    # 设置需要检测和跟踪的目标类别
    # yoloV8官方模型的第一个类别为'person'
    detect_class = 0
    print(f"detecting {model.names[detect_class]}") # model.names返回模型所支持的所有物体类别

    # 加载DeepSort模型
    tracker = ds.DeepSort(parent_dir + "checkpoint/ckpt.t8")
    # tracker=None
    detect_and_track(input_path, output_path, detect_class, model, tracker)
