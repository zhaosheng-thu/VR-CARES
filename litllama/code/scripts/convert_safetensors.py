from safetensors.torch import load_file
import torch
import sys
import os

def load_and_print_weights(bin_path):
    try:
        # 加载.bin文件
        weights = torch.load(bin_path, map_location=torch.device('cpu'))
        # 获取state_dict中的权重信息
        state_dict = weights['state_dict'] if 'state_dict' in weights else weights
        # 打印权重信息
        print("Weight information:")
        for key, value in state_dict.items():
            print(f"Key: {key}, Shape: {value.shape}")
    except Exception as e:
        print(f"Failed to load or print weights: {e}")


path = '/root/szhao/model-weights/Llama-3-8B/model-00001-of-00004.safetensors'
if not os.path.exists(path):
    print('path not exists')
    sys.exit(0)

device = 'cpu'

weights = load_file(path, device=device)
print("Weight information:")
for key, value in weights.items():
    print(f"Key: {key}, Shape: {value.shape}")


name = os.path.splitext(path)[0].split('model')[-1]
file_path = os.path.dirname(os.path.splitext(path)[0]) + f'/pytorch_model{name}.bin'
print("path", file_path)

torch.save(weights, file_path)
# load_and_print_weights(file_path)