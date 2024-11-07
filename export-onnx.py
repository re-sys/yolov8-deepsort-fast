import torch
import torchvision.transforms as transforms
from deep.ShuffleNetV2 import  shufflenet_v2_x0_5 # 替换成你的Net模型的导入路径

# 1. 初始化模型
model_path = '/home/wu/Lab/ShuffleNet-deepsort/checkpoint/ckpt.t8'  # 替换成模型权重的路径
model = shufflenet_v2_x0_5(reid=True)
state_dict = torch.load(model_path, map_location='cpu')['net_dict']  # 加载模型权重
model.load_state_dict(state_dict)
model.eval()  # 设置模型为评估模式
onnx_file_path = 'model.onnx'  # 指定ONNX文件保存路径
# 2. 准备输入张量
dummy_input = torch.randn(1, 3, 64, 128)  # 示例输入 (batch_size=1, channels=3, height=64, width=128)

# 3. 转换为ONNX格式
onnx_file_path = 'model.onnx'  # 指定ONNX文件保存路径
torch.onnx.export(model, dummy_input, onnx_file_path, 
                  export_params=True,        # 保存模型参数
                  opset_version=11,         # ONNX操作集版本
                  do_constant_folding=True,  # 是否进行常量折叠
                  input_names=['input'],     # 输入名称
                  output_names=['output'],    # 输出名称
                  dynamic_axes={'input': {0: 'batch_size'},    # 动态批处理大小
                                'output': {0: 'batch_size'}})
print("模型已成功转换为ONNX格式并保存至 {}".format(onnx_file_path))
