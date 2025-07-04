import json
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence  # 用于处理变长序列的填充
from LSTM_CRF.utils.common import *
from LSTM_CRF.config import *

datas, word2id = build_data()


class NerDataset(Dataset):
    def __init__(self, datas):
        super().__init__()
        self.datas = datas  # 数据列表，每项包含[sample_x, sample_y]

    def __len__(self):
        return len(self.datas)  # 返回数据集大小

    def __getitem__(self, item):
        x = self.datas[item][0]  # 获取第item条数据的特征(字符序列)
        y = self.datas[item][1]  # 获取第item条数据的标签(BIO标签序列)
        return x, y


def collate_fn(batch):
    # 重要知识点：批处理函数，处理变长序列
    # 1. 将一个batch的数据转换为ID表示
    # x_train、y_train 格式为 [tensor0, tensor1, tensor2,,,]
    # 列表嵌套tensor的设计便于后续使用pad_sequence进行填充

    # 将字符转换为ID
    x_train = [torch.tensor([word2id[char] for char in data[0]]) for data in batch]
    # 将标签转换为ID
    y_train = [torch.tensor([conf.tag2id[label] for label in data[1]]) for data in batch]

    # 2. 使用pad_sequence填充序列到相同长度
    # 重要知识点：序列填充
    # x_train填充0(PAD的ID) --> input_ids_pad
    # y_train填充11(标签填充值) --> labels_pad
    input_ids_pad = pad_sequence(x_train, batch_first=True, padding_value=0)
    labels_pad = pad_sequence(y_train, batch_first=True, padding_value=11)
    
    # 3. 创建attention mask，标记非填充位置(1)和填充位置(0)
    # 重要知识点：注意力掩码
    # 掩码用于在模型中区分真实内容和填充内容
    attention_mask = (input_ids_pad != 0).long()
    return input_ids_pad, labels_pad, attention_mask


def get_data():
    # 获取数据加载器
    # 加载(x,y)数据对和word2id词表
    datas, word2id = build_data()
    
    # 重要知识点：训练集/验证集划分
    # 总样本7836，datas[:6200]占比约80%作为训练集
    train_dataset = NerDataset(datas[:6200])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=conf.batch_size,
                              collate_fn=collate_fn,  # 自定义批处理函数
                              drop_last=True)  # 丢弃最后不足一个batch的数据

    # 剩余数据作为验证集
    dev_dataset = NerDataset(datas[6200:])
    dev_loader = DataLoader(dataset=dev_dataset,
                           batch_size=conf.batch_size,
                           collate_fn=collate_fn,
                           drop_last=True)
    return train_loader, dev_loader


if __name__ == '__main__':
    # train_dataloader, dev_dataloader = get_data()

    # train_dataloader数据加载器，是1个可迭代对象，但不是迭代器
    # iter()创建迭代器，支持逐个访问，next()拿到下一个元素
    # inputs, labels, mask = next(iter(train_dataloader))
    # print("next(iter(train_dataloader))以下是一个批次的shape:")
    # print(f"inputs.shape=>{inputs.shape}")
    # print(f"labels.shape=>{labels.shape}")
    # print(f"mask.shape=>{mask.shape}")
    # print(f"inputs[0]为一条样本x=>{inputs[0]}")
    # print(f"labels[0]为一条样本y={labels[0]}")
    # print(f"lmask[0]为一条样本mask={mask[0]}")
    #
    #
    #
    # print(len(inputs))

    # dataset验证
    train_dataset = NerDataset(datas[:6200])
    # 数据切片
    print(train_dataset[:5])
