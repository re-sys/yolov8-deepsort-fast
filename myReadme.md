添加了使用openvino部署，并且缩小了deep模型中的特征提取
时间分析：
1:  640x640 12 persons, 1 bench, 52.9ms
    Speed: 8.1ms preprocess, 52.9ms inference, 1.4ms postprocess per image at shape (1, 3, 640, 640)
    Time taken for _get_features: 0.04415 seconds
    大致上yolo模型耗时52.9ms，特征提取耗时44.15ms，后处理耗时1.4ms，总耗时54.3ms，速度约为98ms/image。

2.  640x640 16 persons, 3 bicycles, 1 motorcycle, 1 potted plant, 53.1ms
    Speed: 7.5ms preprocess, 53.1ms inference, 1.4ms postprocess per image at shape (1, 3, 640, 640)
    Time taken for tracker.predict: 0.00053 seconds
    Time taken for tracker.update: 0.00177 seconds