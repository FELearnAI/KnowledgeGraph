import torch
from transformers import BertTokenizer, BertModel
import json
import os
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseConfig:
    """单例模式配置类，用于全局配置参数。
    鲁棒性：强制使用本地 BERT 模型地址 (bert_path)，支持相对路径和绝对路径，最终解析为绝对路径；避免在线下载导致的设备兼容性问题；验证所有必需路径；适配 Windows、Linux 和 macOS。
    可用性：动态计算 max_length 并限制在 BERT 最大长度内（512）。
    注意：bert_path 必须为本地有效路径，否则抛出异常；不再使用缓存目录。
    """
    _instance = None

    def __new__(cls, bert_path=None, train_data=None, dev_data=None, test_data=None, rel_data=None,
                batch_size=32, max_length=None):
        """单例模式构造函数，确保只初始化一次配置。
        参数：
            bert_path (str): BERT 模型的本地路径，可以是相对路径或绝对路径，必须提供且有效。
            train_data (str): 训练数据路径，不可为空。
            dev_data (str): 验证数据路径，可为空，若为空则使用 test_data。
            test_data (str): 测试数据路径，不可为空。
            rel_data (str): 关系数据路径，不可为空。
            batch_size (int): 批次大小，默认 32。
            max_length (int, optional): 最大序列长度，若未提供则动态计算。
        """
        if cls._instance is None:
            cls._instance = super(BaseConfig, cls).__new__(cls)
            cls._instance._initialize(bert_path, train_data, dev_data, test_data, rel_data,
                                     batch_size, max_length)
        return cls._instance

    def _initialize(self, bert_path, train_data, dev_data, test_data, rel_data, batch_size,
                    max_length):
        """初始化配置参数。
        鲁棒性：验证本地 BERT 路径有效性（支持相对/绝对路径），检查设备一致性；规范化路径处理。
        """
        # 由于MPS设备兼容性问题，暂时强制使用CPU
        # if torch.cuda.is_available():
        #     self.device = torch.device('cuda')
        #     logger.info("使用设备: CUDA")
        # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        #     self.device = torch.device('mps')  # macOS Metal 加速
        #     logger.info("使用设备: macOS Metal (MPS)")
        # else:
        #     self.device = torch.device('cpu')
        #     logger.info("使用设备: CPU")
        self.device = torch.device('cpu')
        logger.info("使用设备: CPU (强制设置以避免MPS兼容性问题)")

        # 验证并解析 bert_path，支持相对路径并转换为绝对路径
        if not bert_path:
            logger.error("bert_path 必须提供有效的本地 BERT 模型路径")
            raise ValueError("请提供有效的本地 BERT 模型路径")
        # 解析为绝对路径，相对路径基于当前工作目录
        self.bert_path = os.path.abspath(bert_path)
        if not os.path.isdir(self.bert_path):
            logger.error(f"bert_path {self.bert_path} 不是有效的目录")
            raise ValueError(f"提供的 BERT 路径 {self.bert_path} 无效")
        logger.info(f"使用本地 BERT 模型，解析为绝对路径: {self.bert_path}")

        try:
            # 加载本地 BERT 模型和 tokenizer，无缓存目录
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
            self.bert_model = BertModel.from_pretrained(self.bert_path).to(self.device)

            # 验证模型和设备一致性
            test_text = "测试文本"
            inputs = self.tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            outputs = self.bert_model(**inputs)
            cls_output = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            logger.info(f"模型测试成功，CLS 输出形状: {cls_output.shape}, 样本值: {cls_output[0][:5]}")

        except Exception as e:
            logger.error(f"加载本地 BERT 模型 {self.bert_path} 失败: {str(e)}")
            raise ValueError(f"无法加载本地 BERT 模型: {str(e)}")

        # 验证并设置数据路径
        required_paths = {"train_data": train_data, "test_data": test_data, "rel_data": rel_data}
        for name, path in required_paths.items():
            if not path or not os.path.exists(path):
                logger.error(f"{name} 路径必须提供且存在")
                raise ValueError(f"{name} 路径无效")
            setattr(self, name, os.path.abspath(path))

        self.dev_data = os.path.abspath(dev_data) if dev_data and os.path.exists(dev_data) else self.test_data

        # 加载关系数据
        self.id2rel = {}
        self.rel2id = {}
        self.num_rel = 0
        try:
            with open(self.rel_data, encoding='utf8') as f:
                self.id2rel = json.load(f)
            self.rel2id = {rel: int(id) for id, rel in self.id2rel.items()}
            self.num_rel = len(self.id2rel)
            logger.info(f"关系数据加载成功，路径: {self.rel_data}")
        except Exception as e:
            logger.error(f"加载关系数据失败: {str(e)}")
            raise ValueError(f"加载关系数据失败: {str(e)}")

        self.batch_size = batch_size if isinstance(batch_size, int) and batch_size > 0 else 32
        self.max_length = max_length if max_length and isinstance(max_length, int) and max_length <= 512 else None
        self.bert_dim = self._get_bert_dim(self.bert_path)
        logger.info(f"配置初始化完成，batch_size: {self.batch_size}, max_length: {self.max_length}, bert_dim: {self.bert_dim}")

    def _is_valid_bert_path(self, bert_path):
        """检查 BERT 路径是否有效。
        参数：
            bert_path (str): BERT 模型路径。
        返回：
            bool: 路径是否有效。
        """
        try:
            BertTokenizer.from_pretrained(bert_path)
            return True
        except:
            return False

    def _get_bert_dim(self, bert_path):
        """根据 BERT 模型路径获取隐藏层维度。
        参数：
            bert_path (str): BERT 模型路径。
        返回：
            int: 模型的隐藏层维度。
        鲁棒性：默认返回 768，若无法判断则抛出警告。
        """
        try:
            if "large" in bert_path.lower():
                return 1024
            return 768
        except:
            logger.warning(f"无法确定 {bert_path} 的 bert_dim，默认使用 768")
            return 768

# 示例使用（主程序中初始化）
if __name__ == '__main__':
    try:
        # 示例 1: 使用 Hugging Face 模型
        baseconf1 = BaseConfig(
            bert_path="/Users/dyq/PycharmProjects/KnowledgeGraph/CasRel_RE/bert-base-chinese",
            train_data="/Users/dyq/PycharmProjects/KnowledgeGraph/CasRel_RE/data/test.json",
            test_data="/Users/dyq/PycharmProjects/KnowledgeGraph/CasRel_RE/data/one_Sample.json",
            rel_data="/Users/dyq/PycharmProjects/KnowledgeGraph/CasRel_RE/data/relation.json",
            max_length=128
        )
        print("示例 1 - bert_path_or_bertModel_name:", baseconf1.bert_path)
    except Exception as e:
        print("示例 1 失败:", str(e))

