# 导入必要的库
import torch  # PyTorch深度学习框架
from transformers import BertTokenizer, BertModel  # Hugging Face的transformers库，用于加载BERT模型和分词器
import json  # 用于处理JSON格式的数据
import os  # 用于与操作系统交互，如处理文件路径
import logging  # 用于记录程序运行时的信息
import time  # 用于时间相关操作

# 配置日志记录器
# level=logging.INFO: 设置日志级别为INFO，即只记录INFO、WARNING、ERROR、CRITICAL级别的日志
# format=...: 定义日志的输出格式，包括时间、日志级别和日志消息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # 创建一个名为当前模块名的日志记录器

class BaseConfig:
    """
    单例模式配置类，用于全局管理项目的所有配置参数。
    
    设计原则:
    - 鲁棒性: 强制使用本地BERT模型路径，避免因网络问题或Hugging Face服务变更导致的不稳定性。
               路径可以是相对的或绝对的，最终都会被解析为绝对路径。
               对所有必需的文件路径进行存在性验证。
               代码兼容Windows、Linux和macOS。
    - 可用性: 如果用户未指定最大序列长度(max_length)，则会动态计算，但会限制在BERT的最大支持长度（通常是512）之内。
    
    注意:
    - `bert_path` 必须是一个包含BERT模型文件（如`pytorch_model.bin`）和配置文件（如`config.json`）的有效本地目录。
    - 该配置类不再使用缓存目录来存储下载的模型，以确保使用的是指定的本地版本。
    """
    _instance = None  # 用于存储唯一的类实例（单例模式）

    def __new__(cls, bert_path=None, train_data=None, dev_data=None, test_data=None, rel_data=None,
                batch_size=32, max_length=None):
        """
        单例模式的构造方法。当第一次创建实例时，它会初始化所有配置；后续再尝试创建时，会直接返回已存在的实例。
        
        参数:
            bert_path (str): BERT模型的本地目录路径。必须提供且有效。
            train_data (str): 训练数据集的JSON文件路径。
            dev_data (str): 验证数据集的JSON文件路径。如果为None，则会使用`test_data`作为验证集。
            test_data (str): 测试数据集的JSON文件路径。
            rel_data (str): 关系到ID映射的JSON文件路径。
            batch_size (int): 训练和评估时每个批次的大小，默认为32。
            max_length (int, optional): 输入序列的最大长度。如果为None，则会根据数据动态调整。
        """
        # 检查是否已存在实例
        if cls._instance is None:
            # 如果不存在，则创建一个新实例
            cls._instance = super(BaseConfig, cls).__new__(cls)
            # 调用初始化方法设置所有配置参数
            cls._instance._initialize(bert_path, train_data, dev_data, test_data, rel_data,
                                     batch_size, max_length)
        # 返回（新的或已存在的）实例
        return cls._instance

    def _initialize(self, bert_path, train_data, dev_data, test_data, rel_data, batch_size,
                    max_length):
        """
        初始化所有配置参数。此方法在`__new__`中被调用，且只在第一次创建实例时执行一次。
        
        设计原则:
        - 鲁棒性: 验证本地BERT路径的有效性，并进行设备一致性检查。所有路径都被规范化为绝对路径。
        """
        # --- 1. 设备配置 ---
        # 确定运行设备（CPU, CUDA, 或 MPS for Apple Silicon）
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info("检测到CUDA，使用设备: CUDA")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            logger.info("检测到Apple Silicon MPS，使用设备: macOS Metal (MPS)")
        else:
            self.device = torch.device('cpu')
            logger.info("未检测到GPU，使用设备: CPU")

        # --- 2. BERT模型及分词器配置 ---
        # 验证并解析bert_path
        if not bert_path:
            logger.error("bert_path 必须提供！这是一个必需的参数。")
            raise ValueError("请提供有效的本地 BERT 模型路径。")
        
        # 将bert_path转换为绝对路径，以支持相对路径输入
        self.bert_path = os.path.abspath(bert_path)
        if not os.path.isdir(self.bert_path):
            logger.error(f"提供的BERT路径 '{self.bert_path}' 不是一个有效的目录。")
            raise ValueError(f"提供的 BERT 路径 '{self.bert_path}' 无效。")
        logger.info(f"使用本地 BERT 模型，解析后的绝对路径为: {self.bert_path}")

        try:
            # 从指定的本地路径加载分词器和BERT模型
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
            self.bert_model = BertModel.from_pretrained(self.bert_path).to(self.device)

            # 执行一个简单的模型测试，以确保模型和设备都工作正常
            logger.info("正在测试加载的BERT模型和设备一致性...")
            test_text = "这是一个测试文本"
            inputs = self.tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():  # 在测试时关闭梯度计算
                outputs = self.bert_model(**inputs)
            cls_output = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            logger.info(f"模型测试成功！CLS向量输出形状: {cls_output.shape}")

        except Exception as e:
            logger.error(f"从路径 '{self.bert_path}' 加载本地 BERT 模型失败: {str(e)}")
            raise ValueError(f"无法加载本地 BERT 模型: {str(e)}")

        # --- 3. 数据集路径配置 ---
        # 验证所有必需的数据路径是否存在
        required_paths = {"train_data": train_data, "test_data": test_data, "rel_data": rel_data}
        for name, path in required_paths.items():
            if not path or not os.path.exists(path):
                logger.error(f"必需的数据路径 '{name}' 未提供或文件不存在于: {path}")
                raise ValueError(f"路径 '{name}' 无效或文件不存在。")
            # 将验证通过的路径设置为类的属性，并转换为绝对路径
            setattr(self, name, os.path.abspath(path))

        # 如果提供了dev_data路径，则使用它；否则，使用test_data作为验证集
        self.dev_data = os.path.abspath(dev_data) if dev_data and os.path.exists(dev_data) else self.test_data
        logger.info(f"训练数据路径: {self.train_data}")
        logger.info(f"验证数据路径: {self.dev_data}")
        logger.info(f"测试数据路径: {self.test_data}")

        # --- 4. 关系数据加载 ---
        self.id2rel = {}  # 初始化ID到关系的映射字典
        self.rel2id = {}  # 初始化关系到ID的映射字典
        self.num_rel = 0  # 初始化关系类型的数量
        try:
            # 打开并加载关系JSON文件
            with open(self.rel_data, encoding='utf8') as f:
                self.id2rel = json.load(f)
            # 创建反向映射
            self.rel2id = {rel: int(id) for id, rel in self.id2rel.items()}
            self.num_rel = len(self.id2rel)
            logger.info(f"关系数据加载成功，共 {self.num_rel} 种关系。")
        except Exception as e:
            logger.error(f"从路径 '{self.rel_data}' 加载关系数据失败: {str(e)}")
            raise ValueError(f"加载关系数据失败: {str(e)}")

        # --- 5. 超参数配置 ---
        # 验证并设置批次大小
        self.batch_size = batch_size if isinstance(batch_size, int) and batch_size > 0 else 32
        # 验证并设置最大序列长度
        self.max_length = max_length if max_length and isinstance(max_length, int) and max_length <= 512 else None
        # 获取BERT模型的隐藏层维度
        self.bert_dim = self._get_bert_dim(self.bert_path)
        logger.info(f"配置初始化完成。批次大小: {self.batch_size}, 最大序列长度: {'动态计算' if self.max_length is None else self.max_length}, BERT维度: {self.bert_dim}")

    def _is_valid_bert_path(self, bert_path):
        """
        检查给定的BERT路径是否包含一个有效的分词器。
        
        参数:
            bert_path (str): BERT模型的路径。
        返回:
            bool: 如果路径有效则返回True，否则返回False。
        """
        try:
            # 尝试从路径加载分词器，如果成功则路径有效
            BertTokenizer.from_pretrained(bert_path)
            return True
        except Exception:
            # 如果加载失败，则路径无效
            return False

    def _get_bert_dim(self, bert_path):
        """
        根据BERT模型的名称（路径中是否包含'large'）来推断其隐藏层的维度。
        
        参数:
            bert_path (str): BERT模型的路径。
        返回:
            int: 模型的隐藏层维度（例如，base模型为768，large模型为1024）。
        
        鲁棒性: 如果无法从路径名中判断，则默认返回768，并打印警告。
        """
        try:
            # 如果模型名称中包含'large'，则认为是large模型
            if "large" in bert_path.lower():
                return 1024
            # 否则，默认为是base模型
            return 768
        except Exception:
            logger.warning(f"无法从路径 '{bert_path}' 判断BERT维度，将默认使用768。")
            return 768

# --- 示例使用 ---
# 这部分代码只有在直接运行此文件时才会执行（例如 `python Base_Conf.py`）
if __name__ == '__main__':
    try:
        # 这是一个如何使用BaseConfig类的示例
        logger.info("--- 开始BaseConfig示例 ---")
        baseconf_instance = BaseConfig(
            bert_path="/Users/dyq/Work/KnowledgeGraph/CasRel_RE/bert-base-chinese",  # 提供本地BERT模型路径
            train_data="/Users/dyq/Work/KnowledgeGraph/CasRel_RE/data/train.json",  # 提供训练数据路径
            dev_data="/Users/dyq/Work/KnowledgeGraph/CasRel_RE/data/dev.json",      # 提供验证数据路径
            test_data="/Users/dyq/Work/KnowledgeGraph/CasRel_RE/data/test.json",    # 提供测试数据路径
            rel_data="/Users/dyq/Work/KnowledgeGraph/CasRel_RE/data/relation.json", # 提供关系数据路径
            max_length=128,  # 设置最大序列长度
            batch_size=16    # 设置批次大小
        )
        print("\n--- 配置加载成功 ---")
        print(f"设备: {baseconf_instance.device}")
        print(f"BERT路径: {baseconf_instance.bert_path}")
        print(f"关系数量: {baseconf_instance.num_rel}")
        print(f"关系到ID的映射: {baseconf_instance.rel2id}")
        print("--- 示例结束 ---\n")
        
    except Exception as e:
        print(f"示例执行失败: {str(e)}")
