import torch
import torch.nn as nn

import argparse, random
from tqdm import tqdm

import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset

from networks.srfbn_arch import SRFBN
from solvers import create_solver




epoch = 200
epoch_is_best = True



parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
#    parser.add_argument('-param', default=None, help='additional parameter for model_saving')
opt = option.parse(parser.parse_args().opt)

solver = create_solver(opt)

def fed_avg(models):
    # 创建一个新的模型实例，用于存储聚合后的模型参数
    aggregated_model = SRFBN(in_channels=3, out_channels=2, num_features=32, num_steps=4, num_groups=3, upscale_factor=4)

    # 遍历聚合模型的参数
    for param_name, param in aggregated_model.named_parameters():
        # 添加'module.'前缀来匹配模型参数字典中的键
        param_name = 'module.' + param_name

        # 计算每个参数的平均值
        param_avg = torch.mean(torch.stack([model['state_dict'][param_name] for model in models]), dim=0)

        # 去掉参数名中的'module.'前缀
        if param_name.startswith('module.'):
            param_name = param_name[len('module.'):]

        # 将参数平均值赋值给聚合模型
        aggregated_model.state_dict()[param_name].copy_(param_avg)

    return aggregated_model

# 创建一个列表，用于存储每个客户端的模型
client_models = []

for client_idx in range(5):
    # 根据客户端的模型文件路径导入模型
    model_path = f"experiments/Model_Test/model{client_idx}.pth"
    model = torch.load(model_path)

    # 将导入的模型添加到模型列表中
    client_models.append(model)


# 使用FedAvg算法对模型进行聚合
aggregated_model = fed_avg(client_models)

for key, value in aggregated_model.state_dict().items():
    print(key, value.shape)

#保存模型
save_path = "experiments/Model_Test/aggregated_model.pth"
#torch.save(aggregated_model.state_dict(), save_path)
save_checkpoint(self, epoch, is_best)
