#include "inference.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <chrono> // 添加这个头文件

int main(int argc, char **argv) {
	// Check if the correct number of arguments is provided
	// if (argc != 3) {
	// 	std::cerr << "usage: " << argv[0] << " <model_path> <video_path>" << std::endl;
	// 	return 1;
	// }
	
	// Get the model and video path from the command-line arguments
	const std::string model_path = "/home/wu/Lab/ultralytics/examples/YOLOv8-OpenVINO-CPP-Inference/build/yolov8n_openvino_model/yolov8n.xml";
	const std::string video_path ="/home/wu/Desktop/jia3.mp4";
	
	// Initialize video capture
	cv::VideoCapture cap(video_path);
	
	// Check if the video was successfully opened
	if (!cap.isOpened()) {
		std::cerr << "ERROR: Unable to open video file" << std::endl;
		return 1;
	}
	
	// Define the confidence and NMS thresholds
	const float confidence_threshold = 0.5;
	const float NMS_threshold = 0.5;
	
	// Initialize the YOLO inference with the specified model and parameters
	yolo::Inference inference(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);

	// Process video frames
	cv::Mat frame;
	while (true) {
		// Capture a new frame from the video
		cap >> frame;

		// Check if the frame is empty (end of video)
		if (frame.empty()) {
			break;
		}
		
		// 记录推理开始时间
		auto start_time = std::chrono::high_resolution_clock::now();

		// Run inference on the current frame
		inference.RunInference(frame);
		
		// 记录推理结束时间
		auto end_time = std::chrono::high_resolution_clock::now();
		
		// 计算推理时间
		std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
		std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;

		// Display the frame with the detections
		cv::imshow("Video Frame", frame);

		// Exit if 'q' is pressed
		if (cv::waitKey(30) == 'q') {
			break;
		}
	}

	// Release video capture object
	cap.release();
	cv::destroyAllWindows();

	return 0;
}
