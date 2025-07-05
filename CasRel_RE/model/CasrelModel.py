import torch
import torch.nn as nn  # 导入PyTorch的神经网络模块
from transformers import BertModel  # 从Hugging Face导入BERT模型
from tqdm import tqdm  # 导入tqdm，用于显示进度条

# 导入项目内的模块
from CasRel_RE.utils.data_loader import *  # 导入数据加载相关的工具函数
from CasRel_RE.config import Config  # 导入配置类

class CasRel(nn.Module):
    """
    CasRel模型的核心实现。
    这是一个基于PyTorch nn.Module的类，实现了“级联的二元标注框架”（Cascading Binary Tagging Framework）用于关系抽取。
    
    模型结构:
    1. 一个共享的BERT编码器，用于获取文本的上下文表示。
    2. 一个主语标注器（Subject Tagger），包含两个独立的线性层，用于预测每个token作为主语（subject）的开始和结束位置。
    3. 一个关系-宾语标注器（Relation-Specific Object Tagger），包含两个线性层，用于在给定一个主语的条件下，预测所有关系类型下宾语（object）的开始和结束位置。
    """
    def __init__(self, conf):
        """
        模型的构造函数。
        
        参数:
            conf (Config): 包含所有配置（如BERT路径、维度、关系数量等）的配置实例。
        """
        super().__init__()
        # --- 1. 加载共享的BERT编码器 ---
        # from_pretrained会加载预训练好的权重
        self.bert = BertModel.from_pretrained(conf.bert_path)

        # --- 2. 定义主语标注器（Subject Tagger）的线性层 ---
        # 这个线性层将BERT的输出（bert_dim）映射到1维，用于预测每个token是主语头的概率
        self.sub_heads_linear = nn.Linear(conf.bert_dim, 1)
        # 这个线性层同样将BERT输出映射到1维，用于预测每个token是主语尾的概率
        self.sub_tails_linear = nn.Linear(conf.bert_dim, 1)

        # --- 3. 定义关系-宾语标注器（Relation-Specific Object Tagger）的线性层 ---
        # 这个线性层将BERT输出映射到（关系数量）维，用于预测每个token是某个关系下的宾语头的概率
        self.obj_heads_linear = nn.Linear(conf.bert_dim, conf.num_rel)
        # 这个线性层同样将BERT输出映射到（关系数量）维，用于预测每个token是某个关系下的宾语尾的概率
        self.obj_tails_linear = nn.Linear(conf.bert_dim, conf.num_rel)

    def get_encoded_text(self, token_ids, mask):
        """
        通过BERT模型获取输入的编码表示。
        
        参数:
            token_ids (Tensor): 输入文本的token ID张量，形状 `[batch_size, seq_len]`。
            mask (Tensor): 注意力掩码张量，用于标识哪些是真实token，哪些是padding，形状 `[batch_size, seq_len]`。
        
        返回:
            Tensor: BERT的最后一层隐藏状态，形状 `[batch_size, seq_len, bert_dim]`。
        """
        # `[0]` 表示获取 `last_hidden_state`
        encoded_text = self.bert(token_ids, attention_mask=mask)[0]
        return encoded_text

    def get_subs(self, encoded_text):
        """
        从编码后的文本中预测主语的头和尾位置。
        
        参数:
            encoded_text (Tensor): BERT的输出，形状 `[batch_size, seq_len, bert_dim]`。
        
        返回:
            tuple: 包含两个张量的元组 (pred_sub_heads, pred_sub_tails)。
                   - pred_sub_heads: 每个token作为主语头的概率，形状 `[batch_size, seq_len, 1]`。
                   - pred_sub_tails: 每个token作为主语尾的概率，形状 `[batch_size, seq_len, 1]`。
        """
        # 将编码后的文本通过主语头线性层，并用sigmoid激活函数转换为概率
        pred_sub_heads = torch.sigmoid(self.sub_heads_linear(encoded_text))
        # 将编码后的文本通过主语尾线性层，并用sigmoid激活函数转换为概率
        pred_sub_tails = torch.sigmoid(self.sub_tails_linear(encoded_text))
        return pred_sub_heads, pred_sub_tails

    def get_objs_for_specific_sub(self, sub_span_mask, sub_len, encoded_text):
        """
        在给定一个特定主语的条件下，预测所有关系下的宾语头和尾。
        这是CasRel模型的核心步骤。
        
        参数:
            sub_span_mask (Tensor): 标记了特定主语位置的掩码，形状 `[batch_size, 1, seq_len]`。
            sub_len (Tensor): 对应主语的长度，形状 `[batch_size, 1]`。
            encoded_text (Tensor): 原始的BERT输出，形状 `[batch_size, seq_len, bert_dim]`。
        
        返回:
            tuple: 包含两个张量的元组 (pred_obj_heads, pred_obj_tails)。
                   - pred_obj_heads: 宾语头预测，形状 `[batch_size, seq_len, num_rel]`。
                   - pred_obj_tails: 宾语尾预测，形状 `[batch_size, seq_len, num_rel]`。
        """
        # --- 1. 提取主语的向量表示 ---
        # 利用矩阵乘法和掩码，提取出主语span内所有token向量的和
        sub_vec_sum = torch.matmul(sub_span_mask, encoded_text)

        # --- 2. 计算主语的平均向量 ---
        # 为了消除主语长度对向量大小的影响，除以主语的长度得到平均向量
        sub_vec_avg = sub_vec_sum / sub_len.unsqueeze(1)  # sub_len需要扩维以匹配

        # --- 3. 将主语信息融合到文本编码中 ---
        # 将主语的平均向量加到原始文本编码的每一个token上，实现“条件化”
        conditioned_encoded_text = encoded_text + sub_vec_avg

        # --- 4. 预测宾语的头和尾 ---
        # 将融合了主语信息的编码传入宾语标注器
        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(conditioned_encoded_text))
        pred_obj_tails = torch.sigmoid(self.obj_tails_linear(conditioned_encoded_text))
        return pred_obj_heads, pred_obj_tails

    def forward(self, input_ids, mask, sub_head2tail, sub_len):
        """
        模型的前向传播函数。
        
        参数:
            input_ids (Tensor): 输入token ID, `[batch_size, seq_len]`.
            mask (Tensor): 注意力掩码, `[batch_size, seq_len]`.
            sub_head2tail (Tensor): 标记了训练时选定的主语span的掩码, `[batch_size, seq_len]`.
            sub_len (Tensor): 对应主语的长度, `[batch_size, 1]`.
        
        返回:
            dict: 包含所有预测结果的字典。
        """
        # --- 1. 获取文本编码 ---
        encoded_text = self.get_encoded_text(input_ids, mask)

        # --- 2. 预测主语 ---
        # 这一步的预测结果在训练时仅用于计算主语损失，不用于下一步
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)

        # --- 3. 预测宾语（使用真实主语进行条件化，即Teacher Forcing）---
        # 增加一个维度以进行矩阵乘法
        sub_head2tail_mask = sub_head2tail.unsqueeze(1)
        # 调用函数，传入真实的（ground-truth）主语信息来预测宾语
        pred_obj_heads, pred_obj_tails = self.get_objs_for_specific_sub(sub_head2tail_mask, sub_len, encoded_text)

        # --- 4. 组装返回结果 ---
        result_dict = {
            'pred_sub_heads': pred_sub_heads,
            'pred_sub_tails': pred_sub_tails,
            'pred_obj_heads': pred_obj_heads,
            'pred_obj_tails': pred_obj_tails,
            'mask': mask  # 将mask也返回，用于计算损失
        }
        return result_dict

    def compute_loss(self, pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails, mask,
                     sub_heads, sub_tails, obj_heads, obj_tails):
        """
        计算模型的总损失。
        总损失是四个部分损失的和：主语头损失、主语尾损失、宾语头损失、宾语尾损失。
        
        参数:
            (所有参数都是Tensor)
            pred_...: 模型的预测结果。
            sub_.../obj_...: 真实的标签。
            mask: 注意力掩码。
        
        返回:
            Tensor: 一个标量，表示当前批次的总损失。
        """
        # --- 1. 准备宾语损失计算所需的掩码 ---
        # 获取关系类型的数量
        rel_count = obj_heads.shape[-1]
        # 将原始mask `[batch, seq]` 扩展为 `[batch, seq, num_rel]` 以匹配宾语预测的形状
        rel_mask = mask.unsqueeze(-1).repeat(1, 1, rel_count)

        # --- 2. 分别计算四个部分的损失 ---
        # 主语头损失
        loss_sub_heads = self.loss(pred_sub_heads, sub_heads, mask)
        # 主语尾损失
        loss_sub_tails = self.loss(pred_sub_tails, sub_tails, mask)
        # 宾语头损失
        loss_obj_heads = self.loss(pred_obj_heads, obj_heads, rel_mask)
        # 宾语尾损失
        loss_obj_tails = self.loss(pred_obj_tails, obj_tails, rel_mask)

        # --- 3. 将所有损失相加 ---
        total_loss = loss_sub_heads + loss_sub_tails + loss_obj_heads + loss_obj_tails
        return total_loss

    def loss(self, pred, gold, mask):
        """
        一个通用的二元交叉熵损失计算函数。
        
        参数:
            pred (Tensor): 模型的预测概率值。
            gold (Tensor): 真实的0/1标签。
            mask (Tensor): 掩码，用于忽略padding部分带来的损失。
        
        返回:
            Tensor: 计算出的平均损失。
        """
        # 如果预测结果的最后一个维度是1，则压缩掉，使其形状与gold和mask匹配
        pred = pred.squeeze(-1)
        
        # 使用二元交叉熵损失，reduction='none'表示不立即求和或平均，而是返回每个元素的损失
        bce_loss = nn.BCELoss(reduction='none')(pred, gold)
        
        # 只计算非padding部分的损失
        masked_loss = bce_loss * mask
        # 将有效损失相加，并除以有效token的数量，得到平均损失
        return torch.sum(masked_loss) / torch.sum(mask)

# --- 示例代码（用于调试） ---
if __name__ == '__main__':
    # 这部分代码只有在直接运行此文件时才会执行
    print("--- 开始CasRel模型调试 ---")
    conf = Config()
    model = CasRel(conf).to(conf.device)
    
    # 获取一个批次的数据用于测试
    dataloaders = get_all_dataloader()
    dev_loader = dataloaders["dev"]
    
    # 取一个批次
    for inputs, labels in tqdm(dev_loader):
        # 将数据移动到指定设备
        for key in inputs: inputs[key] = inputs[key].to(conf.device)
        for key in labels: labels[key] = labels[key].to(conf.device)

        print("\n--- 正在测试前向传播... ---")
        # **inputs会将字典解包为函数的关键字参数
        logits = model(**inputs)
        print("模型输出（logits）的键:", logits.keys())
        print("主语头预测形状:", logits['pred_sub_heads'].shape)
        print("宾语头预测形状:", logits['pred_obj_heads'].shape)

        print("\n--- 正在测试损失计算... ---")
        # 将模型输出和真实标签传入损失计算函数
        total_loss = model.compute_loss(**logits, **labels)
        print(f"计算出的总损失为: {total_loss.item():.4f}")
        
        # 只测试一个批次就退出
        break 
    print("--- CasRel模型调试结束 ---")

