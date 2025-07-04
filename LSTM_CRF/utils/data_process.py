# 步骤1：导入必要的库和模块
import json  # 用于处理JSON数据（本代码未直接使用，可能为后续扩展预留）
import os  # 用于文件路径操作和目录管理
from LSTM_CRF.config import Config # 自定义配置类，存储数据路径和标签字典

# 步骤2：设置工作目录
cur = os.getcwd()  # 获取当前工作目录路径
print('当前数据处理默认工作目录：', cur)  # 打印工作目录，便于调试和确认路径

# 步骤3：初始化配置对象
conf = Config()  # 创建Config实例，加载数据路径和标签字典等配置信息


# 步骤4：定义TransferData类，用于处理数据并生成BIO格式的训练数据
class TransferData:
    def __init__(self):
        """
        步骤4.1：初始化方法，设置文件路径和标签字典
        变量：
            self.label_dict: 标签字典，映射中文标签到英文（如"身体部位" -> "BODY"）
            self.origin_path: 原始数据目录路径
            self.train_filepath: 训练数据输出文件路径
        """
        self.label_dict = conf.labels  # 从Config获取标签字典
        self.origin_path = conf.data_origin  # 从Config获取原始数据目录路径
        self.train_filepath = conf.train_path  # 从Config获取训练数据输出文件路径

    def transfer(self):
        """
        步骤4.2：核心方法，处理原始数据，生成BIO格式的训练数据
        子步骤：
            1. 创建训练输出文件
            2. 遍历原始数据目录，筛选包含"original"的文件
            3. 读取原始文本和标注文件，生成字符级BIO标签
            4. 将字符和标签写入训练文件
        """
        # 步骤4.2.1：创建训练输出文件，使用UTF-8编码
        with open(self.train_filepath, 'w', encoding='utf-8') as fr:
            print("处理后的数据存放路径：=>", self.train_filepath)  # 打印输出路径，便于调试

            # 步骤4.2.2：递归遍历原始数据目录
            for root, dirs, files in os.walk(self.origin_path):
                for file in files:
                    # 步骤4.2.3：拼接文件路径并筛选原始文件
                    filepath = os.path.join(root, file)  # 拼接完整文件路径
                    if 'original' not in filepath:  # 只处理文件名包含"original"的文件
                        continue

                    # 步骤4.2.4：获取对应的标注文件路径
                    label_filepath = filepath.replace('.txtoriginal', '')  # 去掉".txtoriginal"后缀

                    # 步骤4.2.5：读取标注文件，生成字符索引到BIO标签的映射
                    # 重要知识点：BIO标注映射
                    # 通过read_label_text方法解析标注文件，生成每个字符位置的BIO标签
                    res_dict = self.read_label_text(label_filepath)

                    # 步骤4.2.6：读取原始文本文件
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()  # 读取文件内容并去除首尾空白
                        # 步骤4.2.7：遍历文本content的每个字符，生成BIO标签
                        # 重要知识点：字符级BIO标注
                        # 通过res_dict.get(idx, 'O')获取每个字符的BIO标签，默认为'O'(非实体)
                        for idx, char in enumerate(content):
                            char_label = res_dict.get(idx, 'O')  # 获取字符的BIO标签，默认为'O'（非实体）
                            # 步骤4.2.8：写入字符和标签到训练文件
                            fr.write(char + '\t' + char_label + '\n')

            # 步骤4.2.9：数据处理完成，打印提示信息
            print("数据处理完成！")

    def read_label_text(self, label_filepath):
        """
        步骤4.3：解析标注文件，生成字符索引到BIO标签的映射
        参数：
            label_filepath: 标注文件路径
        返回：
            res_dict: 字典，键为字符索引，值为BIO标签（如"B-BODY", "I-BODY"）
        子步骤：
            1. 读取标注文件，逐行解析
            2. 提取实体信息（文本、起始索引、结束索引、标签）
            3. 根据BIO格式生成标签
        """
        res_dict = {}  # 初始化结果字典，存储字符索引到BIO标签的映射
        # 步骤4.3.1：逐行读取标注文件
        for line in open(label_filepath, 'r', encoding='utf-8'):
            # 步骤4.3.2：解析每行数据 line.strip().split('\t')
            # 格式示例："右髋部  21  23  身体部位"
            res = line.strip().split('\t')  # 去除空白，按制表符分割
            # res示例：['右髋部', '21', '23', '身体部位']

            # 步骤4.3.3：提取起始索引、结束索引和标签
            start = int(res[1])  # 实体起始索引
            end = int(res[2])  # 实体结束索引
            label = res[3]  # 实体标签（如"身体部位"）

            # 步骤4.3.4：将中文标签转换为英文标签
            label_tag = self.label_dict.get(label)  # 如"身体部位" -> "BODY"

            # 步骤4.3.5：为每个字符生成BIO标签
            # 重要知识点：BIO标注规则
            # B-标签：实体的开始位置
            # I-标签：实体的中间或结束位置
            # O：非实体位置（在res_dict中不存在的索引默认为'O'）
            for i in range(start, end + 1):
                if i == start:
                    tag = "B-" + label_tag  # 实体开始位置，标记为B-标签
                else:
                    tag = "I-" + label_tag  # 实体中间或结束位置，标记为I-标签
                res_dict[i] = tag  # 存储索引和标签

        # 步骤4.3.6：返回字符索引到BIO标签的映射
        return res_dict


# 步骤5：程序入口，执行数据处理
if __name__ == '__main__':
    # 步骤5.1：创建TransferData实例
    handler = TransferData()
    # 步骤5.2：调用transfer方法，执行数据处理
    handler.transfer()