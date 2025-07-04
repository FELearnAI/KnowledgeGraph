import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import BertModel
from tqdm import tqdm
from CasRel_RE.utils.data_loader import *
from CasRel_RE.config import Config

class CasRel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.bert = BertModel.from_pretrained(conf.bert_path)
        # 定义第一个线性层，来判断主实体的头部位置
        self.sub_heads_linear = nn.Linear(conf.bert_dim, 1)
        # 定义第二个线性层，来判断主实体的尾部位置
        self.sub_tails_linear = nn.Linear(conf.bert_dim, 1)
        # 定义第三个线性层，来判断客实体的头部位置以及关系类型
        self.obj_heads_linear = nn.Linear(conf.bert_dim, conf.num_rel)
        # 定义第四个线性层，来判断客实体的尾部位置以及关系类型
        self.obj_tails_linear = nn.Linear(conf.bert_dim, conf.num_rel)

    def get_encoded_text(self, token_ids, mask):
        encoded_text = self.bert(token_ids, attention_mask=mask)[0]
        return encoded_text

    #头实体位置预测
    def get_subs(self, encoded_text):
        #encoded_text 维度 [batch_size,seq_len,bert_dim]

        #获得头实体开始位置 预测结果 pre_sub_heads维度[batch_siez,seq_len,1]
        pre_sub_heads = torch.sigmoid(self.sub_heads_linear(encoded_text))

        # 获得尾实体开始位置 预测结果 pre_sub_heads维度[batch_siez,seq_len,1]
        pre_sub_tails = torch.sigmoid(self.sub_tails_linear(encoded_text))

        return pre_sub_heads, pre_sub_tails

   #尾实体位置预测
    def get_objs_for_specific_sub(self, sub_head2tail, sub_len, encoded_text):
        '''
        将subject实体信息融合原始句子中：将主实体字向量实现平均，然后加在当前句子的每一个字向量上，进行计算
        :param sub_head2tail:shape-->【16，1, 200】（原始维度：sub_head2tail维度 [batch_size,seq_len]，处理为sub_head2tail维度 [batch_size,1,seq_len]）
        :param sub_len:shape--->[16,1]
        :param encoded_text:.shape[16，200，768]
        :return:
            pred_obj_heads-->shape []
            pre_obj_tails-->shape []
        '''
        # 1. 提取主实体特征,这里sub_head2tail是一个实体位置信息的掩码mask，值为 1 的位置表示主实体。相乘得到 得到融合结果（紫色）sub,sub.shape--->[bacth_size,1,768]
        #  sub_head2tail维度 [batch_size,seq_len,1] ; encoded_text维度 [batch_size,seq_len,768];  sub.shape--->[bacth_size,1,768]
        sub = torch.matmul(sub_head2tail, encoded_text)# 将主实体特征和编码后的文本进行融合

        #2. 扩展 sub_len 维度, sub_len原始维度[batch_size,1],扩展之后的维度sub_len.shape --->[batch_size,1,1]
        sub_len = sub_len.unsqueeze(1) # 主实体长度（扩维）

        # 3.平均主实体特征[batch_size,seq_len,768]，避免实体长度或者实体过短
        sub = sub / sub_len # 平均主实体信息 sub.shape--->[bacth_size,1,768]

        # 4. 融合主实体特征 与 文本特征
        encoded_text = encoded_text + sub #将处理后的实体特征和原始编码后的文本进行融合

        # 5. 预测客体实体头部和尾部
        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        pre_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))
        return pred_obj_heads, pre_obj_tails

    def forward(self, input_ids, mask, sub_head2tail, sub_len):
        '''
        :param input_ids: shape-->[batch_size, seq_len]
        :param mask: shape-->[batch_size, seq_len]
        :param sub_head2tail: shape-->[batch_size, seq_len]
        :param sub_len: shape-->[batch_size,1]
        :return:
        '''
        # todo: encode_text.shape--->[batch_size,seq_len,bert_dim]
        # 1. encoded_text，对输入文本信息进行编码
        encoded_text = self.get_encoded_text(input_ids, mask)

        # 2. 通过self.get_subs进行头实体位置预测pred_sub_heads, pre_sub_tails
        pred_sub_heads, pre_sub_tails = self.get_subs(encoded_text)

        # 3. 尾实体位置预测
        # 3.1 sub_head2tail，对头实体位置信息进行增加维度1
        sub_head2tail = sub_head2tail.unsqueeze(1)

        # 3.2 通过self.get_objs_for_specific_sub 进行尾实体位置预测  pred_obj_heads, pre_obj_tails
        pred_obj_heads, pre_obj_tails =self.get_objs_for_specific_sub(sub_head2tail, sub_len,encoded_text)

        # 4. 组装返回结果
        result_dict = {'pred_sub_heads': pred_sub_heads,
                       'pred_sub_tails': pre_sub_tails,
                       'pred_obj_heads': pred_obj_heads,
                       'pred_obj_tails': pre_obj_tails,
                       'mask': mask}
        return result_dict

    def compute_loss(self,
                     pred_sub_heads, pred_sub_tails,
                     pred_obj_heads, pred_obj_tails,
                     mask,
                     sub_heads, sub_tails,
                     obj_heads, obj_tails):
        '''
        计算损失
        :param pred_sub_heads:[batch_size, seq_len, 1]
        :param pred_sub_tails:[batch_size, seq_len, 1]
        :param pred_obj_heads:[batch_size, seq_len, 18]
        :param pred_obj_tails:[batch_size, seq_len, 18]
        :param mask: shape-->[batch_size, seq_len]
        :param sub_heads: shape-->[batch_size, seq_len]
        :param sub_tails: shape-->[batch_size, seq_len]
        :param obj_heads: shape-->[batch_size, seq_len, 18]
        :param obj_tails: shape-->[batch_size, seq_len, 18]
        :return:
        '''
        # todo:sub_heads.shape,sub_tails.shape, mask-->[bacth_size, seq_len]
        # todo:obj_heads.shape,obj_tails.shape-->[batch_size, seq_len, 18]
        # 1.获取类别数据
        rel_count = obj_heads.shape[-1]

        # 2. 对mask进行扩展，并复制18次。[batch_size, seq_len]=》[batch_size, seq_len,1] =》repeat 18次 ==》[batch_size, seq_len, 18]
        rel_mask = mask.unsqueeze(-1).repeat(1, 1, rel_count)

        # 3. 计算损失,头实体位置损失用mask，尾实体位置损失用rel_mask
        loss_1 = self.loss(pred_sub_heads, sub_heads, mask)
        loss_2 = self.loss(pred_sub_tails, sub_tails, mask)
        loss_3 = self.loss(pred_obj_heads, obj_heads, rel_mask)
        loss_4 = self.loss(pred_obj_tails, obj_tails, rel_mask)

        # 4. 对损失进行求和并返回

        return loss_1 + loss_2 + loss_3 + loss_4

    def loss(self, pred, gold, mask):
        # 1. 对最后一维度如果1维数据，则进行 squeeze压缩
        pred = pred.squeeze(-1)

        # 2.采用二元交叉熵损失nn.BCELoss，对每个位置进行loss计算
        los = nn.BCELoss(reduction='none')(pred, gold)

        # 3. 掩码掉无效位置的loss数据，计算loss均值
        los = torch.sum(los * mask) / torch.sum(mask)
        return los



if __name__ == '__main__':

    conf=Config ()
    model = CasRel(conf)
    model.to(conf.device)
    dataloaders=get_all_dataloader()
    train_loader=dataloaders["train"]
    test_loader = dataloaders["test"]
    dev_loader = dataloaders["dev"]

    for idx, (inputs, labels) in enumerate(tqdm(dev_loader)):
        print("==============sub头实体预测====================")
        encoded_text=model.get_encoded_text(inputs['input_ids'], inputs['mask'])
        print("bert对文本进行encoded_text编码数据：",encoded_text.shape)
        #预测主实体的开始位置与结束位置
        pred_sub_heads, pred_sub_tails = model.get_subs(encoded_text)
        print("主实体头预测结果：",pred_sub_heads.shape)
        print("主实体尾预测结果：",pred_sub_tails.shape)
        print("==============客头实体预测================== sub_head2tail, sub_len, encoded_text)==")
        sub_head2tail = inputs['sub_head2tail'].unsqueeze(1)
        pred_obj_heads, pre_obj_tails = model.get_objs_for_specific_sub(sub_head2tail, inputs['sub_len'], encoded_text)
        logits = model(**inputs)

        print("===============loss计算=======================")
        loss=model.loss(pred_sub_heads,labels["sub_heads"],mask=inputs['mask'])
        print("loss  ==>",loss.item())

        loss_1=model.compute_loss(pred_sub_heads, pred_sub_tails, pred_obj_heads, pre_obj_tails, inputs['mask'],
                                  labels["sub_heads"], labels["sub_tails"],labels["obj_heads"], labels["obj_tails"])
        print("=============================")
        print(loss_1)
        break



    # 因为本次模型借助BERT做fine_tuning， 因此需要对模型中的大部分参数进行L2正则处理防止过拟合，包括权重w和偏置b
    # prepare optimzier
    # named_parameters()获取模型中的参数和参数名字
    # param_optimizer = list(model.named_parameters())
    # print(f'param_optimizer--->{param_optimizer}')
    # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]  # no_decay中存放不进行权重衰减的参数{因为bert官方代码对这三项免于正则化}
    # # any()函数用于判断给定的可迭代参数iterable是否全部为False，则返回False，如果有一个为True，则返回True
    # # 判断param_optimizer中所有的参数。如果不在no_decay中，则进行权重衰减;如果在no_decay中，则不进行权重衰减
    # optimizer_grouped_parameters = [
    #     {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
    #     {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    #

