import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.data_loader import get_all_loader
from lstmAtten_datautils.process import relation2id, word2id
from config import Config

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# sys.path.append(root_dir)
sys.path.insert(0, root_dir)
conf = Config()
word2id_dict = word2id(conf.base_conf)
relation2id = relation2id(conf.base_conf)


class BiLSTM_ATT(nn.Module):
    def __init__(self, conf):
        '''
        BiLSTM+注意力机制模型，用于关系分类任务
        :param conf: 配置文件对象，包含模型参数
        :param vocab_size: 词汇表大小（去重后的单词总数）
        :param pos_size: 位置编码的数量（如0到149）
        :param tag_size: 关系类型标签的数量
        '''
        super().__init__()
        self.device = conf.device
        self.vocab_size = len(word2id_dict) + 1
        # 单词嵌入的维度
        self.embedding_dim = conf.embedding_dim
        self.pos_size = conf.pos_size
        # 位置嵌入的维度
        self.pos_dim = conf.pos_dim
        # LSTM输出维度（实际隐藏层维度为hidden_dim//2，因为是双向LSTM）
        self.hidden_dim = conf.hidden_dim
        self.tag_size = len(relation2id)

        # 步骤1：定义单词嵌入层，将单词转为向量
        self.word_embed = nn.Embedding(self.vocab_size, self.embedding_dim)
        # 步骤2：定义实体1位置嵌入层，表示单词相对实体1的距离
        self.pos1_embed = nn.Embedding(self.pos_size, self.pos_dim)
        # 步骤3：定义实体2位置嵌入层，表示单词相对实体2的距离
        self.pos2_embed = nn.Embedding(self.pos_size, self.pos_dim)
        # 步骤4：定义双向LSTM层，捕获句子上下文信息
        self.lstm = nn.LSTM(self.embedding_dim + self.pos_dim * 2, self.hidden_dim // 2, bidirectional=True,
                            batch_first=True)
        # 步骤5：定义注意力权重，形状[hidden_dim, 1]，对应论文w ∈ ℝ^{d_w}
        self.weight = nn.Parameter(torch.randn(conf.hidden_dim, 1))
        # 步骤6：定义输出层，将句子表示映射到关系标签
        self.out = nn.Linear(self.hidden_dim, self.tag_size)
        # 步骤7：定义Dropout层，防止过拟合
        self.dropout_embed = nn.Dropout(p=0.3)
        self.dropout_lstm = nn.Dropout(p=0.3)
        self.dropout_atten = nn.Dropout(p=0.5)

    def attention(self, H):
        '''
        计算注意力权重并生成句子表示，严格按照论文公式，仅保留核心转置
        :param H: LSTM输出，形状[batch_size, seq_len, hidden_dim]
        :return: 句子表示，形状[batch_size, hidden_dim]，对应论文h* ∈ ℝ^{d_w}
        '''
        # 步骤1：计算M = tanh(H)，公式：M = tanh(H)
        # H: [batch_size, seq_len, hidden_dim]， # M: [batch_size, seq_len, hidden_dim]
        device = H.device  # 从 H 获取设备，不依赖 conf
        M = torch.tanh(H)
        # 步骤2：计算α = softmax(w^T M)，公式：α = softmax(w^T M)
        # M: [batch_size, seq_len, hidden_dim]   # self.weight: [hidden_dim, 1]   # w^T M: [batch_size, seq_len, 1]
        # print('M.device:', M.device)
        # print(f'M.device: {M.device}, self.weight.device: {self.weight.device}')
        alpha_scores = torch.matmul(M, self.weight)  # [batch_size, seq_len, 1]
        alpha_scores = alpha_scores.squeeze(-1)  # [batch_size, seq_len]
        # softmax得到α: [batch_size, seq_len]
        alpha = F.softmax(alpha_scores, dim=-1)
        # 步骤3：计算r = H α^T，公式：r = H α^T
        # H: [batch_size, seq_len, hidden_dim] # H^T: [batch_size, hidden_dim, seq_len] # α: [batch_size, seq_len, 1]
        r = torch.bmm(H.transpose(1, 2), alpha.unsqueeze(-1))  # [batch_size, hidden_dim, 1]
        # 移除多余维度，得到r: [batch_size, hidden_dim]
        r = r.squeeze(-1)
        # 步骤4：计算h* = tanh(r)，公式：h* = tanh(r)
        # h_star: [batch_size, hidden_dim]
        h_star = torch.tanh(r)
        return h_star

    def forward(self, sentence, pos1, pos2):
        '''
        模型前向传播
        :param sentence: 输入句子，形状[batch_size, seq_len]
        :param pos1: 实体1位置编码，形状[batch_size, seq_len]
        :param pos2: 实体2位置编码，形状[batch_size, seq_len]
        :return: 预测关系类型分数，形状[batch_size, tag_size]
        '''
        # 调试：打印输入形状
        # print("=== 前向传播 ===")
        # print("sentence shape:", sentence.shape)
        # print("pos1 shape:", pos1.shape)
        # print("pos2 shape:", pos2.shape)

        # 步骤1：将句子、实体1和实体2位置转为嵌入向量并拼接
        embeds = torch.cat((self.word_embed(sentence),
                            self.pos1_embed(pos1),
                            self.pos2_embed(pos2)), dim=-1)

        # 步骤2：对嵌入应用Dropout，防止过拟合
        embeds = self.dropout_embed(embeds)

        # 步骤3：通过双向LSTM，捕获上下文信息
        lstm_out, _ = self.lstm(embeds)  # [batch_size, seq_len, hidden_dim]
        lstm_out = self.dropout_lstm(lstm_out)

        # 步骤4：应用注意力机制，提取关键信息
        sentence_repr = self.attention(lstm_out)  # [batch_size, hidden_dim]
        sentence_repr = self.dropout_atten(sentence_repr)

        # 步骤5：通过输出层映射到关系类型分数
        output = self.out(sentence_repr)  # [batch_size, tag_size]

        return output


if __name__ == '__main__':
    # 主函数：测试模型输入输出形状
    loaders = get_all_loader()
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    model = BiLSTM_ATT(conf).to(conf.device)
    for datas, positionE1, positionE2, labels, _, _, _ in train_loader:
        # 检查索引有效性
        # if (datas >= model.vocab_size).any() or (datas < 0).any():
        #     print("错误：datas 包含无效索引！")
        # if (positionE1 >= model.pos_size).any() or (positionE1 < 0).any():
        #     print("错误：positionE1 包含无效索引！")
        # if (positionE2 >= model.pos_size).any() or (positionE2 < 0).any():
        #     print("错误：positionE2 包含无效索引！")

        # 移动输入张量到 MPS 设备
        datas = datas.to(conf.device)
        positionE1 = positionE1.to(conf.device)
        positionE2 = positionE2.to(conf.device)

        print(f'输入句子形状--->{datas.shape}')
        print(f'实体1位置编码形状--->{positionE1.shape}')
        print(f'实体2位置编码形状--->{positionE2.shape}')
        # 前向传播
        output = model(datas, positionE1, positionE2)
        print(f'模型输出形状--->{output.shape}')
