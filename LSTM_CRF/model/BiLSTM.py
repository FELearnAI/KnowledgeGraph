import torch.nn as nn
from LSTM_CRF.utils.data_loader import *
from LSTM_CRF.config import *
from LSTM_CRF.utils.common import build_data

config = Config()
datas, word2id = build_data()


class NERLSTM(nn.Module):
    def __init__(self, config):
        # super(BiLSTM, self).__init__()  # 旧式写法
        super().__init__()  # 新式写法
        # 初始化模型参数
        self.name = 'BiLSTM'  # 模型名称
        self.embedding_dim = config.embedding_dim  # 词嵌入维度
        self.hidden_dim = config.hidden_dim  # LSTM隐藏层维度
        self.vocab_size = len(word2id)  # 词汇表大小
        self.tag2id = config.tag2id  # 标签到ID的映射
        self.tag_size = len(self.tag2id)  # 标签数量
        self.dropout_conf = config.dropout  # dropout比率

        # 重要知识点：模型组件定义
        # 1. 词嵌入层：将输入的词ID转换为密集向量表示
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # 2. 双向LSTM层：捕获序列的上下文信息
        # bidirectional=True表示使用双向LSTM
        # batch_first=True表示输入张量的形状为(batch_size, seq_len, input_size)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, bidirectional=True, batch_first=True)
        
        # 3. Dropout层：防止过拟合
        self.dropout = nn.Dropout(self.dropout_conf)
        
        # 4. 全连接层：将LSTM输出映射到标签空间
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)

    # 重要知识点：清晰数据维度变化
    def forward(self, x, mask):
        # x维度是：[batch_size, seq_len]，包含词的ID
        # embedding的维度：[batch_size, seq_len, embedding_dim]
        embedding = self.word_embeds(x)
        
        # LSTM输出维度：[batch_size, seq_len, hidden_dim]
        outputs, hidden = self.lstm(embedding)
        
        # 应用mask，忽略padding位置
        # mask维度：[batch_size, seq_len, 1]
        outputs = outputs * mask.unsqueeze(-1)
        
        # 应用dropout
        outputs = self.dropout(outputs)
        
        # 线性映射到标签空间，维度：[batch_size, seq_len, tag_size]
        outputs = self.hidden2tag(outputs)
        return outputs


if __name__ == '__main__':
    train_loader, dev_loader = get_data()
    inputs, labels, mask = next(iter(train_loader))
    print('inputs.shape:\t', inputs.shape)
    print('labels.shape:\t', labels.shape)

    model = NERLSTM(config)
    print(model(inputs, mask).shape)
    print(model)
