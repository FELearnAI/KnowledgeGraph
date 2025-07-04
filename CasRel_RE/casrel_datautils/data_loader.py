from torch.utils.data import DataLoader, Dataset
from pprint import pprint  # 美化输出 层次结构输出
from .process import *
import json
import os
import logging
from .Base_Conf import BaseConfig
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MyDataset(Dataset):
    """自定义数据集类，用于加载 JSON 格式的数据。
    使用说明：初始化时传入配置实例和数据集类型，自动从 baseconf 获取对应数据路径并解析 JSON 行数据。
    参数：
        baseconf (BaseConfig): 配置实例，包含数据路径。
        dataset_type (str): 数据集类型，'train'、'dev' 或 'test'。
    鲁棒性：检查数据路径并处理解析错误。
    """

    def __init__(self, baseconf, dataset_type):
        self.baseconf = baseconf
        # 根据 dataset_type 从 baseconf 获取数据路径
        if dataset_type == "train":
            data_path = baseconf.train_data
        elif dataset_type == "dev":
            data_path = baseconf.dev_data
        elif dataset_type == "test":
            data_path = baseconf.test_data
        else:
            logger.error(f"无效的 dataset_type: {dataset_type}")
            raise ValueError(f"dataset_type 必须是 'train', 'dev' 或 'test'")

        if not data_path or not os.path.exists(data_path):
            logger.error(f"数据路径 {data_path} 不存在或未提供")
            raise FileNotFoundError(f"数据路径 {data_path} 不存在或未提供")
        try:
            self.dataset = [json.loads(line) for line in open(data_path, encoding='utf8')]
            logger.info(f"数据集加载成功，路径: {os.path.abspath(data_path)}")
        except Exception as e:
            logger.error(f"解析数据文件 {data_path} 失败: {str(e)}")
            raise ValueError(f"解析数据文件 {data_path} 失败: {str(e)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index >= len(self.dataset):
            logger.warning("索引超出数据集范围")
            raise IndexError("索引超出数据集范围")
        content = self.dataset[index]
        text = content.get('text', '')  # 容错处理，若 text 缺失则返回空字符串
        spo_list = content.get('spo_list', [])  # 容错处理，若 spo_list 缺失则返回空列表
        return text, spo_list


def get_dataloader(baseconf):
    """获取数据加载器。
    使用说明：根据配置实例中的数据路径返回训练、验证和测试 DataLoader。
    参数：
        baseconf (BaseConfig): 配置实例，包含数据路径和相关设置。
    返回：
        dict: 包含 train_dataloader, dev_dataloader, test_dataloader 的字典，若数据路径为空或文件不存在则对应项为 None。
    鲁棒性：处理缺失数据路径和无效 DataLoader。
    """
    dataloaders = {}

    if baseconf.train_data:
        try:
            train_data = MyDataset(baseconf, "train")
            dataloaders['train'] = DataLoader(
                dataset=train_data,
                batch_size=baseconf.batch_size,
                shuffle=True,
                collate_fn=lambda x: collate_fn(x, baseconf),
                drop_last=True
            )
            logger.info(f"train_dataloader 创建成功，路径: {baseconf.train_data}")
        except Exception as e:
            logger.error(f"初始化 train_dataloader 失败: {str(e)}")
    if baseconf.dev_data:
        try:
            dev_data = MyDataset(baseconf, "dev")
            dataloaders['dev'] = DataLoader(
                dataset=dev_data,
                batch_size=baseconf.batch_size,
                shuffle=True,
                collate_fn=lambda x: collate_fn(x, baseconf),
                drop_last=True
            )
            logger.info(f"dev_dataloader 创建成功，路径: {baseconf.dev_data}")
        except Exception as e:
            logger.error(f"初始化 dev_dataloader 失败: {str(e)}")
    if baseconf.test_data:
        try:
            test_data = MyDataset(baseconf, "test")
            dataloaders['test'] = DataLoader(
                dataset=test_data,
                batch_size=baseconf.batch_size,
                shuffle=True,
                collate_fn=lambda x: collate_fn(x, baseconf),
                drop_last=True
            )
            logger.info(f"test_dataloader 创建成功，路径: {baseconf.test_data}")
        except Exception as e:
            logger.error(f"初始化 test_dataloader 失败: {str(e)}")

    return dataloaders




# if __name__ == '__main__':
#     try:
#         baseconf = BaseConfig(bert_path=None, train_data="train.json", test_data="test.json", rel_data="relation.json")
#         dataloaders = get_data(baseconf)
#         for name, loader in dataloaders.items():
#             if loader is not None:
#                 print(f"{name}: {next(iter(loader))}")
#     except Exception as e:
#         logger.error(f"主程序执行错误: {str(e)}")