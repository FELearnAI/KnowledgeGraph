import torch
from lstmAtten_datautils.base_Conf import BaseConfig
from pathlib import Path


class Config():
    def __init__(self):
        # 项目根目录
        self.root_path = Path(__file__).parent.parent / 'Bilstm_Attention_RE'
        # 数据目录
        self.data_path = self.root_path / 'data'
        # 1 定义设备
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print('self.device:', self.device)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 2 数据集路径
        self.train_data_path = self.data_path / 'train.txt'
        self.test_data_path = self.data_path / 'test.txt'
        self.rel_data_path = self.data_path / 'relation2id.txt'
        self.vocab_data_path = self.data_path / 'vocab.txt'

        # 3 模型参数
        # 词嵌入维度
        self.embedding_dim = 128
        # 位置嵌入维度 相对位置编码的嵌入维度 ->
        self.pos_dim = 32
        # BiLSTM 隐藏层维度
        self.hidden_dim = 200
        # 句子最大长度 经过句子长度分布分析而来
        self.max_length = 70
        # conf.pos_size
        self.pos_size = 150

        # 4 训练参数
        self.epochs = 5
        self.batch_size = 32
        self.learning_rate = 1e-3

        self.base_conf = BaseConfig(
            train_data_path=self.train_data_path,
            test_data_path=self.test_data_path,
            vocab_path=self.vocab_data_path,
            rel_path=self.rel_data_path,
            max_length=70,
            batch_size=32
        )


if __name__ == '__main__':
    conf = Config()
    print('conf.root_path:', conf.root_path)
    print(conf.train_data_path)
    print(conf.batch_size)

    # 普通字符串中，反斜杠 \ 表示转义，\t \n \\
    # 想表示原始字符串：
    # 1 开头加 r，表示不去转义，仍为原始字符串
    # 2 Python 中双反斜杠 表示 单反斜杠，\\
    # 3 原始字符串中，不能有 \n 等转义字符，只能是 \t \n \\
    # 4 原始字符串中，不能有单引号，只能是双引号
    # 5 原始字符串中，不能有双引号，只能是单引号
    # 6 原始字符串中，不能有反斜杠，只能是单反斜杠
    # 7 原始字符串中，不能有回车，只能是换行