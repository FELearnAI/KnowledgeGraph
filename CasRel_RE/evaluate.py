import os
import sys
import torch
import pandas as pd

# 将项目根目录添加到Python路径，以确保可以正确导入所有模块
# os.path.abspath(__file__) 获取当前文件的绝对路径
# os.path.dirname() 获取目录名
# os.path.join(..., '..') 上溯一级目录到项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from model.CasrelModel import CasRel
from config import Config
from utils.data_loader import get_all_dataloader
from train import model2dev  # 从train.py导入评估函数
from path_utils import get_model_save_path

def evaluate_model():
    """
    加载训练好的最佳模型，并在开发集上进行评估。
    """
    print("--- 1. 初始化配置和模型 ---")
    conf = Config()
    model = CasRel(conf)
    model.to(conf.device)

    print("--- 2. 加载最佳模型权重 ---")
    # 使用路径管理工具获取模型保存目录
    save_dir = get_model_save_path('CasRel_RE')
    model_path = os.path.join(save_dir, 'casrel_best.pth')

    if not os.path.exists(model_path):
        print(f"错误: 在 {model_path} 未找到模型文件。请先运行 train.py 进行训练。")
        sys.exit(1)

    # 加载模型状态字典
    model.load_state_dict(torch.load(model_path, map_location=conf.device))
    print(f"成功从 {model_path} 加载模型。")

    print("--- 3. 加载验证数据集 ---")
    # 获取数据加载器，我们只需要验证集
    dataloaders = get_all_dataloader()
    dev_loader = dataloaders["dev"]

    print("--- 4. 开始评估 ---")
    # 调用评估函数
    _, _, _, _, _, _, df = model2dev(model, dev_loader)
    
    print("--- 评估结果 ---")
    print(df)
    print("--- 评估完成 ---")

if __name__ == '__main__':
    evaluate_model()