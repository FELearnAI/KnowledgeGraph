import torch.nn as nn
# pip install pytorch-crf==0.7.2
from torchcrf import CRF  # 导入PyTorch-CRF库，提供CRF层的实现
from LSTM_CRF.utils.data_loader import *
from LSTM_CRF.utils.common import *

from LSTM_CRF.utils.common import build_data

# 加载数据和词汇表
datas, word2id = build_data()


class NERLSTM_CRF(nn.Module):
    def __init__(self, conf):  # 初始化模型
        super().__init__()
        # 初始化模型参数
        self.name = "BiLSTM_CRF"  # 模型名称，用于区分不同模型
        self.embedding_dim = conf.embedding_dim  # 词嵌入维度
        self.hidden_dim = conf.hidden_dim  # LSTM隐藏层维度
        self.vocab_size = len(word2id)  # 词汇表大小
        self.tag2id = conf.tag2id  # 标签到ID的映射
        self.tag_size = len(conf.tag2id)  # 标签数量
        
        # 重要知识点：模型组件定义
        # 1. 词嵌入层：将输入的词ID转换为密集向量表示
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # 2. 双向LSTM层：捕获序列的上下文信息
        # bidirectional=True表示使用双向LSTM，能同时考虑前后文信息
        # batch_first=True表示输入张量的形状为(batch_size, seq_len, input_size)
        # hidden_dim // 2确保双向拼接后的维度等于hidden_dim
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            bidirectional=True, batch_first=True)
        
        # 3. Dropout层：防止过拟合
        self.dropout_conf = conf.dropout  # dropout比率
        self.dropout = nn.Dropout(self.dropout_conf)
        
        # 4. 全连接层：将LSTM输出映射到标签空间，得到发射分数
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        
        # 5. CRF层：学习标签之间的转移约束，实现全局最优标签序列
        # 重要知识点：CRF层能够学习标签间的依赖关系，如B-XXX后面更可能是I-XXX而不是B-YYY
        self.crf = CRF(self.tag_size, batch_first=True)

    def get_lstm2linear(self, x):
        # 计算LSTM和线性层的输出（发射分数矩阵）
        # 重要知识点：发射分数表示每个位置属于各个标签的可能性
        out = self.word_embeds(x)  # 词嵌入，形状: [batch_size, seq_len, embedding_dim]
        out, _ = self.lstm(out)  # LSTM编码，形状: [batch_size, seq_len, hidden_dim]
        out = self.dropout(out)  # 应用dropout
        out = self.hidden2tag(out)  # 线性映射到标签空间，形状: [batch_size, seq_len, tag_size]
        return out

    def forward(self, x, mask):  # 前向传播，用于预测
        '''
        重要知识点：CRF解码过程
        crf 需要 mask.bool() - 确保只考虑有效位置的标签
        解码用 crf.decode() - 使用Viterbi算法找出最可能的标签序列
        解码结果是list，不是tensor - 需要特别注意处理方式
        '''
        out = self.get_lstm2linear(x)  # 获取发射分数
        out = out * mask.unsqueeze(-1)  # 应用mask，忽略padding位置
        # 重要知识点：CRF.decode实现了Viterbi算法，找出最可能的标签序列
        out = self.crf.decode(out, mask.bool())  # 返回最可能的标签序列列表
        return out

    def log_likelihood(self, x, tags, mask):  # 计算损失函数
        # 重要知识点：CRF损失计算
        # CRF损失 = -log(目标序列概率/所有可能序列概率之和)
        # 负号是因为优化目标是最小化损失，而我们想最大化目标序列概率
        out = self.get_lstm2linear(x)  # 获取发射分数
        out = out * mask.unsqueeze(-1)  # 应用mask
        # 计算负对数似然损失，reduction='mean'表示对batch取平均
        return -self.crf(out, tags, mask.bool(), reduction='mean')


if __name__ == '__main__':
    train_dataloader, dev_dataloader = get_data()
    inputs, labels, mask = next(iter(train_dataloader))

    # conf=Config()
    conf = Config()
    model = NERLSTM_CRF(conf)

    # 解码1条最优路径
    path = model(inputs, mask)[0]
    id2tag = {v: k for k, v in conf.tag2id.items()}
    path = [id2tag[i] for i in path]

    # 计算损失
    loss = model.log_likelihood(inputs, labels, mask).item()
    print('最优路径：%s\n平均损失：%.4f' % (path, loss))
