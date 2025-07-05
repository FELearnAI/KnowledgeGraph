# 导入必要的库
from torch.utils.data import DataLoader, Dataset  # 从PyTorch导入DataLoader和Dataset类
from pprint import pprint  # 导入pprint，用于美化输出，特别适合层次结构的数据
from .process import *  # 从同级目录的process.py文件中导入所有内容
import json  # 用于处理JSON数据
import os  # 用于与操作系统交互，如路径处理
import logging  # 用于日志记录
from .Base_Conf import BaseConfig  # 从同级目录的Base_Conf.py导入BaseConfig类

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # 创建一个日志记录器

class MyDataset(Dataset):
    """
    自定义的PyTorch数据集类，用于从JSON文件中加载和处理数据。
    
    使用说明:
    - 初始化时需要提供一个配置实例(`baseconf`)和数据集类型(`dataset_type`)。
    - 它会自动根据`dataset_type`从`baseconf`中找到对应的数据文件路径，并加载数据。
    
    参数:
        baseconf (BaseConfig): 包含所有配置（如数据路径）的配置实例。
        dataset_type (str): 指定要加载的数据集类型，必须是 'train', 'dev', 或 'test' 之一。
    
    鲁棒性:
    - 会检查提供的数据路径是否有效和存在。
    - 会处理JSON文件解析过程中可能发生的错误。
    """

    def __init__(self, baseconf, dataset_type):
        """
        数据集类的构造函数。
        """
        self.baseconf = baseconf  # 保存配置实例
        
        # 根据`dataset_type`确定要加载的数据文件路径
        if dataset_type == "train":
            data_path = baseconf.train_data
        elif dataset_type == "dev":
            data_path = baseconf.dev_data
        elif dataset_type == "test":
            data_path = baseconf.test_data
        else:
            # 如果提供了无效的类型，则记录错误并抛出异常
            logger.error(f"无效的数据集类型: {dataset_type}")
            raise ValueError(f"dataset_type 必须是 'train', 'dev' 或 'test'")

        # 检查数据路径是否有效
        if not data_path or not os.path.exists(data_path):
            logger.error(f"数据路径 '{data_path}' 不存在或未在配置中提供。")
            raise FileNotFoundError(f"数据路径 '{data_path}' 不存在或未提供。")
        
        try:
            # 从JSON文件中逐行读取数据，并解析为Python对象列表
            self.dataset = [json.loads(line) for line in open(data_path, encoding="utf8")]
            logger.info(f"成功加载数据集: {os.path.abspath(data_path)}")
        except Exception as e:
            # 如果文件解析失败，则记录错误并抛出异常
            logger.error(f"解析数据文件 '{data_path}' 失败: {str(e)}")
            raise ValueError(f"解析数据文件 '{data_path}' 失败: {str(e)}")

    def __len__(self):
        """
        返回数据集中样本的总数。
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        根据给定的索引`index`，返回一个数据样本。
        """
        # 检查索引是否越界
        if index >= len(self.dataset):
            logger.warning("请求的索引超出了数据集的范围。")
            raise IndexError("索引超出了数据集范围")
        
        # 获取指定索引的数据内容
        content = self.dataset[index]
        # 安全地获取'text'字段，如果不存在则返回空字符串
        text = content.get('text', '')
        # 安全地获取'spo_list'字段（三元组列表），如果不存在则返回空列表
        spo_list = content.get('spo_list', [])
        
        # 返回文本和对应的三元组列表
        return text, spo_list

def get_dataloader(baseconf):
    """
    创建并返回训练、验证和测试集的DataLoader。
    
    使用说明:
    - 根据`baseconf`中的数据路径，为每个数据集（train, dev, test）创建一个DataLoader。
    
    参数:
        baseconf (BaseConfig): 包含所有配置的实例。
    
    返回:
        dict: 一个包含'train', 'dev', 'test'的DataLoader的字典。如果某个数据集的路径无效，则对应的项会是None。
    
    鲁棒性:
    - 会处理数据路径缺失或无效的情况。
    """
    dataloaders = {}  # 初始化一个空字典来存储DataLoaders

    # --- 创建训练集DataLoader ---
    if baseconf.train_data:
        try:
            train_data = MyDataset(baseconf, "train")  # 创建训练集Dataset实例
            dataloaders['train'] = DataLoader(
                dataset=train_data,  # 指定数据集
                batch_size=baseconf.batch_size,  # 设置批次大小
                shuffle=True,  # 在每个epoch开始时打乱数据顺序
                collate_fn=lambda x: collate_fn(x, baseconf),  # 指定自定义的批处理函数
                drop_last=True  # 丢弃最后一个不完整的批次
            )
            logger.info(f"训练集DataLoader创建成功，路径: {baseconf.train_data}")
        except Exception as e:
            logger.error(f"初始化训练集DataLoader失败: {str(e)}")

    # --- 创建验证集DataLoader ---
    if baseconf.dev_data:
        try:
            dev_data = MyDataset(baseconf, "dev")  # 创建验证集Dataset实例
            dataloaders['dev'] = DataLoader(
                dataset=dev_data,
                batch_size=baseconf.batch_size,
                shuffle=True,  # 通常验证集不需要打乱，但这里设置为True
                collate_fn=lambda x: collate_fn(x, baseconf),
                drop_last=True
            )
            logger.info(f"验证集DataLoader创建成功，路径: {baseconf.dev_data}")
        except Exception as e:
            logger.error(f"初始化验证集DataLoader失败: {str(e)}")

    # --- 创建测试集DataLoader ---
    if baseconf.test_data:
        try:
            test_data = MyDataset(baseconf, "test")  # 创建测试集Dataset实例
            dataloaders['test'] = DataLoader(
                dataset=test_data,
                batch_size=baseconf.batch_size,
                shuffle=True,  # 通常测试集不需要打乱
                collate_fn=lambda x: collate_fn(x, baseconf),
                drop_last=True
            )
            logger.info(f"测试集DataLoader创建成功，路径: {baseconf.test_data}")
        except Exception as e:
            logger.error(f"初始化测试集DataLoader失败: {str(e)}")

    return dataloaders  # 返回包含所有DataLoaders的字典

# --- 示例代码（被注释掉） ---
# if __name__ == '__main__':
#     try:
#         # 这是一个如何直接测试此文件功能的示例
#         baseconf = BaseConfig(bert_path=None, train_data="train.json", test_data="test.json", rel_data="relation.json")
#         dataloaders = get_dataloader(baseconf)
#         for name, loader in dataloaders.items():
#             if loader is not None:
#                 # 打印每个dataloader的第一个批次的数据
#                 print(f"{name}: {next(iter(loader))}")
#     except Exception as e:
#         logger.error(f"主程序执行错误: {str(e)}")
''