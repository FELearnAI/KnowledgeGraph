import json
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

# 设置基础目录
BASE_DIR = '/Users/dyq/PycharmProjects/KnowledgeGraph/LSTM_CRF'


# 定义配置类
class Config():
    def __init__(self):
        # 1. 设备配置
        # 重要知识点：设备选择
        # 优先使用MPS(Mac的GPU加速)，否则使用CPU
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # 2. 数据路径配置
        # 原始数据目录
        self.data_origin = f'{BASE_DIR}/data_origin'
        # 训练数据文件路径
        self.train_path = f'{BASE_DIR}/data/train.txt'
        # 词汇表文件路径
        self.vocab_path = f'{BASE_DIR}/data/vocab.txt'
        # 加载标签映射
        self.tag2id = json.load(open(f'{BASE_DIR}/data/tag2id.json', encoding='utf-8'))
        self.target = list(self.tag2id.keys())  # 目标标签列表
        # 加载标签字典(中文标签到英文标签的映射)
        self.labels = json.load(open(f'{BASE_DIR}/data/labels.json', encoding='utf-8'))

        # 3. 模型超参数
        # 重要知识点：模型超参数
        self.embedding_dim = 768  # 词嵌入维度
        self.hidden_dim = 256  # LSTM隐藏层维度
        self.dropout = 0.2  # Dropout比率
        
        # 4. 训练参数
        self.batch_size = 32  # 批次大小
        self.epochs = 10  # 训练轮次
        self.lr = 2e-5  # 学习率
        
        # 5. 模型选择
        # 可选: 'BiLSTM' 或 'BiLSTM_CRF'
        self.model = 'BiLSTM_CRF'  # 默认使用BiLSTM_CRF模型
