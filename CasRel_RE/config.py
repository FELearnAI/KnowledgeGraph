# 导入必要的库
import torch  # PyTorch深度学习框架
from transformers import BertTokenizer  # Hugging Face的transformers库，用于加载BERT分词器
import json  # 用于处理JSON格式的数据

class Config(object):
    """
    一个用于存储所有模型和训练相关配置的类。
    将所有配置集中管理，方便修改和维护。
    """
    def __init__(self):
        """
        初始化配置类的所有参数。
        """
        super().__init__()
        
        # --- 1. 设备配置 ---
        # 自动检测并选择可用的最佳计算设备（GPU优先）
        if torch.cuda.is_available():
            # 如果NVIDIA GPU可用，则使用CUDA
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            # 如果在Apple Silicon上运行，则使用Metal Performance Shaders (MPS)
            self.device = torch.device('mps')
        else:
            # 否则，使用CPU
            self.device = torch.device('cpu')

        # --- 2. BERT模型相关配置 ---
        # BERT模型的本地路径。使用相对路径以保证项目的可移植性。
        self.bert_path = "CasRel_RE/bert-base-chinese"
        # 从指定路径加载BERT分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # BERT模型的隐藏层维度（对于bert-base-chinese是768）
        self.bert_dim = 768

        # --- 3. 数据集路径配置 ---
        # 使用相对路径定义所有数据文件，增强可移植性
        self.train_path = "CasRel_RE/data/train.json"  # 训练集文件路径
        self.dev_path = "CasRel_RE/data/dev.json"      # 验证集文件路径
        self.test_path = "CasRel_RE/data/test.json"     # 测试集文件路径
        self.rel_data = "CasRel_RE/data/relation_raw.json"  # 关系到ID的映射文件路径

        # --- 4. 关系数据加载 ---
        # 从JSON文件加载ID到关系的映射
        self.id2rel = json.load(open(self.rel_data, encoding="utf-8"))
        # 创建一个反向的映射：从关系到ID
        self.rel2id = {rel: int(id) for id, rel in self.id2rel.items()}
        # 计算关系类型的总数
        self.num_rel = len(self.id2rel)

        # --- 5. 训练超参数 ---
        self.epochs = 5  # 训练的总轮数
        self.learning_rate = 1e-5  # 优化器的学习率
        self.batch_size = 2  # 每个训练批次的大小
        self.patience = 3  # 早停机制的“耐心”，即连续patience个epoch验证集性能没有提升就停止训练

# --- 示例使用 ---
# 这部分代码只有在直接运行此文件时才会执行（例如 `python config.py`）
if __name__ == "__main__":
    # 创建一个Config实例
    config = Config()
    # 打印加载的关系映射，以验证配置是否正确
    print("--- 关系ID到关系的映射 ---")
    print(config.id2rel)
    print("--- 关系到关系ID的映射 ---")
    print(config.rel2id)
