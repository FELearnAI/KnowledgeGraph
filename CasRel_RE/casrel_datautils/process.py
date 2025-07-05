# 导入必要的库
from .Base_Conf import BaseConfig  # 从同级目录导入BaseConfig
import torch  # PyTorch深度学习框架
from collections import defaultdict  # 导入defaultdict，一个带有默认值的字典
import logging  # 用于日志记录

# --- PyTorch打印选项配置 ---
# 设置打印张量时显示的元素数量阈值为无穷大，确保所有元素都能被显示
torch.set_printoptions(threshold=torch.inf)
# 设置打印时的行宽，以避免在打印长张量时自动换行
torch.set_printoptions(linewidth=200)

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # 创建一个日志记录器


def find_head_idx(source, target):
    """
    在源序列（source）中查找目标子序列（target）的起始索引。
    
    使用说明: 主要用于在分词后的文本ID序列中，定位实体（subject或object）的起始位置。
    
    参数:
        source (list): 源序列，例如 `[101, 2, 3, 4, 5, 102]`。
        target (list): 要查找的目标子序列，例如 `[3, 4]`。
    
    返回:
        int: 目标子序列在源序列中的起始索引。如果未找到，则返回 -1。
    
    鲁棒性: 会处理源或目标序列为空的情况。
    """
    if not source or not target:
        return -1  # 如果任一序列为空，则无法查找，返回-1
    target_len = len(target)
    for i in range(len(source)):
        # 滑动窗口，比较源序列中的子序列是否与目标序列匹配
        if source[i: i + target_len] == target:
            return i  # 如果找到匹配，返回起始索引
    return -1  # 如果遍历完仍未找到，返回-1

def create_label(inner_triples, inner_input_ids, seq_len, baseconf):
    """
    为单个样本创建用于训练的标签张量。
    
    使用说明: 根据样本中的三元组（SPO）信息，生成主语、宾语和关系的one-hot形式的标签张量。
    
    参数:
        inner_triples (list): 一个包含多个三元组字典的列表，例如 `[{'subject': 'A', 'predicate': 'r', 'object': 'B'}, ...]`。
        inner_input_ids (list): 已经分词并转换为ID的输入文本序列。
        seq_len (int): 输入序列的长度（通常是填充或截断后的长度）。
        baseconf (BaseConfig): 全局配置实例。
    
    返回:
        tuple: 一个包含所有标签张量的元组，具体包括：
               (sub_len, sub_head2tail, sub_heads, sub_tails, obj_heads, obj_tails)。
    """
    # 如果没有三元组或输入ID，则返回全零的标签张量
    if not inner_triples or not inner_input_ids:
        return torch.tensor([0], dtype=torch.float), torch.zeros(seq_len), torch.zeros(seq_len), torch.zeros(seq_len), \
               torch.zeros((seq_len, baseconf.num_rel)), torch.zeros((seq_len, baseconf.num_rel))

    # --- 初始化标签张量 ---
    inner_sub_heads = torch.zeros(seq_len)  # 主语头标签，长度为seq_len
    inner_sub_tails = torch.zeros(seq_len)  # 主语尾标签
    inner_obj_heads = torch.zeros((seq_len, baseconf.num_rel))  # 宾语头标签，形状为(seq_len, 关系数量)
    inner_obj_tails = torch.zeros((seq_len, baseconf.num_rel))  # 宾语尾标签
    inner_sub_head2tail = torch.zeros(seq_len)  # 用于标记某个特定主语的span
    inner_sub_len = torch.tensor([1], dtype=torch.float)  # 标记主语的长度
    s2ro_map = defaultdict(list)  # 创建一个字典，用于存储每个主语(S)对应的所有关系(R)和宾语(O)

    # --- 1. 解析三元组并构建 s2ro_map ---
    for inner_triple in inner_triples:
        try:
            # 将S, P, O文本转换为token ID
            sub_token_ids = baseconf.tokenizer(inner_triple.get('subject', ''), add_special_tokens=False)['input_ids']
            rel_id = baseconf.rel2id.get(inner_triple.get('predicate', ''), 0)
            obj_token_ids = baseconf.tokenizer(inner_triple.get('object', ''), add_special_tokens=False)['input_ids']
            
            # 在输入序列中找到S和O的起始位置
            sub_head_idx = find_head_idx(inner_input_ids, sub_token_ids)
            obj_head_idx = find_head_idx(inner_input_ids, obj_token_ids)

            # 如果S和O都被成功找到
            if sub_head_idx != -1 and obj_head_idx != -1 and sub_head_idx < seq_len and obj_head_idx < seq_len:
                sub_tail_idx = sub_head_idx + len(sub_token_ids) - 1
                obj_tail_idx = obj_head_idx + len(obj_token_ids) - 1
                sub_span = (sub_head_idx, sub_tail_idx) # 主语的span (start, end)
                # 将 (宾语头, 宾语尾, 关系ID) 添加到对应主语的列表中
                s2ro_map[sub_span].append((obj_head_idx, obj_tail_idx, rel_id))
        except Exception as e:
            logger.warning(f"处理三元组 {inner_triple} 时发生错误: {str(e)}")
            continue

    # --- 2. 根据 s2ro_map 生成标签张量 ---
    if s2ro_map:
        # 标记所有出现过的主语的头和尾位置
        for s_span in s2ro_map:
            if s_span[0] < seq_len and s_span[1] < seq_len:
                inner_sub_heads[s_span[0]] = 1
                inner_sub_tails[s_span[1]] = 1
        
        # 遍历所有主语，并为每个主语生成其对应的宾语-关系标签
        # 修复了之前随机选择一个主语的问题，现在会为所有主语生成标签
        for sub_head_idx, sub_tail_idx in s2ro_map.keys():
            if sub_head_idx < seq_len and sub_tail_idx < seq_len:
                # 在`inner_sub_head2tail`中标记这个被选中的主语的整个范围
                # 注意：这里仍然只标记一个主语的span，因为模型在预测宾语时是基于一个特定主语的
                # 但在训练时，我们会遍历所有主语，确保每个主语的宾语都能被学习到
                inner_sub_head2tail[sub_head_idx : sub_tail_idx + 1] = 1
                # 计算并保存这个主语的长度
                inner_sub_len = torch.tensor([sub_tail_idx + 1 - sub_head_idx], dtype=torch.float)
                
                # 遍历这个主语对应的所有宾语和关系
                for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                    obj_head_idx, obj_tail_idx, rel_id = ro
                    # 在`inner_obj_heads`和`inner_obj_tails`的相应位置标记为1
                    if obj_head_idx < seq_len and obj_tail_idx < seq_len and rel_id < baseconf.num_rel:
                        inner_obj_heads[obj_head_idx][rel_id] = 1
                        inner_obj_tails[obj_tail_idx][rel_id] = 1

    return inner_sub_len, inner_sub_head2tail, inner_sub_heads, inner_sub_tails, inner_obj_heads, inner_obj_tails

def collate_fn(batch, baseconf):
    """
    自定义的批处理函数，用于DataLoader。
    
    使用说明: 将一个批次（batch）的原始数据（文本和三元组）转换为模型所需的格式化的PyTorch张量。
              它会处理文本的padding和truncation，并调用`create_label`生成标签。
    
    参数:
        batch (list): 一个列表，其中每个元素都是 `(text, spo_list)` 的元组。
        baseconf (BaseConfig): 全局配置实例。
    
    返回:
        tuple: 一个包含两个字典的元组 `(inputs, labels)`。
               `inputs` 包含模型前向传播所需的张量。
               `labels` 包含计算损失所需的真实标签张量。
    """
    if not batch:
        # 处理空批次的情况
        return {'input_ids': torch.tensor([]).to(baseconf.device), 'mask': torch.tensor([]).to(baseconf.device), \
                'sub_head2tail': torch.tensor([]).to(baseconf.device), 'sub_len': torch.tensor([]).to(baseconf.device)}, \
               {'sub_heads': torch.tensor([]).to(baseconf.device), 'sub_tails': torch.tensor([]).to(baseconf.device), \
                'obj_heads': torch.tensor([]).to(baseconf.device), 'obj_tails': torch.tensor([]).to(baseconf.device)}

    # --- 1. 提取文本和三元组 ---
    text_list = [value[0] for value in batch]
    triple_list = [value[1] for value in batch]

    try:
        # --- 2. 文本编码、填充和截断 ---
        bert_max_length = baseconf.bert_model.config.max_position_embeddings  # 获取BERT的最大长度 (512)
        # 计算当前批次中最长句子的长度
        max_seq_len_in_batch = max(len(baseconf.tokenizer.encode(text, add_special_tokens=True)) for text in text_list) if text_list else 0
        # 确定最终的有效最大长度
        effective_max_length = baseconf.max_length if baseconf.max_length and baseconf.max_length <= bert_max_length else min(max_seq_len_in_batch, bert_max_length)

        # 使用tokenizer对整个批次的文本进行编码
        encoded_text = baseconf.tokenizer.batch_encode_plus(
            text_list,
            padding=True,  # 填充到批次中的最长序列
            truncation=True,  # 截断超长序列
            max_length=effective_max_length,
            return_tensors='pt'  # 返回PyTorch张量
        )
        batch_size = len(encoded_text['input_ids'])
        seq_len = len(encoded_text['input_ids'][0])

    except Exception as e:
        logger.error(f"Tokenization失败: {str(e)}")
        # 返回空的张量
        return {'input_ids': torch.tensor([]).to(baseconf.device), 'mask': torch.tensor([]).to(baseconf.device), \
                'sub_head2tail': torch.tensor([]).to(baseconf.device), 'sub_len': torch.tensor([]).to(baseconf.device)}, \
               {'sub_heads': torch.tensor([]).to(baseconf.device), 'sub_tails': torch.tensor([]).to(baseconf.device), \
                'obj_heads': torch.tensor([]).to(baseconf.device), 'obj_tails': torch.tensor([]).to(baseconf.device)}

    # --- 3. 为批次中的每个样本创建标签 ---
    sub_heads_list, sub_tails_list, obj_heads_list, obj_tails_list, sub_len_list, sub_head2tail_list = [], [], [], [], [], []
    for i in range(batch_size):
        input_ids = encoded_text['input_ids'][i].tolist()
        triples = triple_list[i]
        # 调用create_label为当前样本生成标签
        labels = create_label(triples, input_ids, seq_len, baseconf)
        sub_len_list.append(labels[0])
        sub_head2tail_list.append(labels[1])
        sub_heads_list.append(labels[2])
        sub_tails_list.append(labels[3])
        obj_heads_list.append(labels[4])
        obj_tails_list.append(labels[5])

    # --- 4. 将标签列表堆叠成张量，并移动到指定设备 ---
    try:
        input_ids = encoded_text['input_ids'].to(baseconf.device)
        mask = encoded_text['attention_mask'].to(baseconf.device)
        sub_heads = torch.stack(sub_heads_list).to(baseconf.device)
        sub_tails = torch.stack(sub_tails_list).to(baseconf.device)
        sub_len = torch.stack(sub_len_list).to(baseconf.device)
        sub_head2tail = torch.stack(sub_head2tail_list).to(baseconf.device)
        obj_heads = torch.stack(obj_heads_list).to(baseconf.device)
        obj_tails = torch.stack(obj_tails_list).to(baseconf.device)
    except RuntimeError as e:
        logger.error(f"张量堆叠失败: {str(e)}")
        # 返回空的张量
        return {'input_ids': torch.tensor([]).to(baseconf.device), 'mask': torch.tensor([]).to(baseconf.device), \
                'sub_head2tail': torch.tensor([]).to(baseconf.device), 'sub_len': torch.tensor([]).to(baseconf.device)}, \
               {'sub_heads': torch.tensor([]).to(baseconf.device), 'sub_tails': torch.tensor([]).to(baseconf.device), \
                'obj_heads': torch.tensor([]).to(baseconf.device), 'obj_tails': torch.tensor([]).to(baseconf.device)}

    # --- 5. 组织成模型所需的输入和标签字典 ---
    inputs = {'input_ids': input_ids, 'mask': mask, 'sub_head2tail': sub_head2tail, 'sub_len': sub_len}
    labels = {'sub_heads': sub_heads, 'sub_tails': sub_tails, 'obj_heads': obj_heads, 'obj_tails': obj_tails}

    return inputs, labels

def extract_sub(pred_sub_heads, pred_sub_tails):
    """
    从预测的头和尾位置中提取主语实体。
    
    参数:
        pred_sub_heads (Tensor): 预测的主语头概率张量。
        pred_sub_tails (Tensor): 预测的主语尾概率张量。
    
    返回:
        list: 提取到的主语实体列表，每个实体表示为 `(start_index, end_index)` 的元组。
    """
    # 将概率转换为0或1（基于0.5的阈值）
    pred_sub_heads = convert_score_to_zero_one(pred_sub_heads)
    pred_sub_tails = convert_score_to_zero_one(pred_sub_tails)

    subs = []
    # 找到所有被预测为头和尾的位置
    heads = torch.nonzero(pred_sub_heads == 1).squeeze(1)
    tails = torch.nonzero(pred_sub_tails == 1).squeeze(1)

    # 遍历所有可能的头和尾组合，形成主语span
    for head in heads:
        for tail in tails:
            if tail >= head:  # 确保尾部在头部之后或相同位置
                subs.append((head.item(), tail.item()))
    return subs

def extract_obj_and_rel(obj_heads, obj_tails):
    """
    从预测的宾语头和尾位置中提取宾语实体和关系。
    
    参数:
        obj_heads (Tensor): 预测的宾语头概率张量，形状 `[seq_len, num_rel]`。
        obj_tails (Tensor): 预测的宾语尾概率张量，形状 `[seq_len, num_rel]`。
    
    返回:
        list: 提取到的宾语和关系列表，每个元素表示为 `(rel_id, start_index, end_index)` 的元组。
    """
    # 将概率转换为0或1
    obj_heads = convert_score_to_zero_one(obj_heads)
    obj_tails = convert_score_to_zero_one(obj_tails)

    obj_and_rels = []
    # 遍历每种关系类型
    for rel_id in range(obj_heads.shape[1]):
        # 提取当前关系下的宾语头和尾
        current_rel_obj_heads = obj_heads[:, rel_id]
        current_rel_obj_tails = obj_tails[:, rel_id]
        
        # 使用extract_sub函数来提取宾语span
        objs = extract_sub(current_rel_obj_heads, current_rel_obj_tails)
        
        # 如果提取到了宾语，则将其与关系ID组合
        for obj_span in objs:
            obj_and_rels.append((rel_id, obj_span[0], obj_span[1]))
    return obj_and_rels

def convert_score_to_zero_one(tensor):
    """
    将预测的概率分数张量转换为二值（0或1）张量。
    
    使用说明: 以0.5为阈值，大于等于0.5的设置为1，小于0.5的设置为0。
    
    参数:
        tensor (Tensor): 输入的概率分数张量。
    
    返回:
        Tensor: 二值化后的张量。
    """
    # 克隆张量以避免修改原始张量
    tensor = tensor.clone()
    tensor[tensor >= 0.5] = 1
    tensor[tensor < 0.5] = 0
    return tensor

def single_sample_process(baseconf, sample_data):
    """
    处理单条样本，主要用于模型预测阶段。
    
    使用说明: 将单条文本数据编码为模型预测所需的输入张量和注意力掩码。
    
    参数:
        baseconf (BaseConfig): 全局配置实例。
        sample_data (dict): 包含 'text' 键的字典，表示单条样本数据。
    
    返回:
        tuple: (input_tensor, mask_tensor) - 输入张量和注意力掩码张量，形状为 `[1, seq_len]`。
    
    鲁棒性: 处理无效输入和长度超限情况，确保输出格式与训练数据一致。
    """
    if not sample_data or 'text' not in sample_data:
        logger.error("样本数据无效，必须包含 'text' 键。")
        raise ValueError("样本数据无效，必须包含 'text' 键。")

    text = sample_data['text']
    max_char_length = 480  # 字符长度限制，用于截断过长的文本
    try:
        # 检查文本字符长度，如果过长则进行截断
        char_length = len(text)
        logger.info(f"文本字符长度: {char_length}")

        if char_length > max_char_length:
            logger.warning(f"文本字符长度 {char_length} 超过 {max_char_length}，已截断。")
            text = text[:max_char_length]  # 截断文本

        # 使用tokenizer对单条样本进行编码
        encoded = baseconf.tokenizer(
            [text],  # 将单条文本放入列表中，以便使用batch_encode_plus进行批处理编码
            padding=True,  # 填充到批次中的最长序列
            truncation=True,  # 截断超长序列
            max_length=baseconf.max_length,  # 使用配置中定义的最大长度
            return_tensors='pt'  # 返回PyTorch张量
        )
        # 获取编码后的input_ids和attention_mask
        input_tensor = encoded['input_ids']  # 形状: [1, seq_len]
        mask_tensor = encoded['attention_mask']  # 形状: [1, seq_len]

        # 验证长度
        seq_len = input_tensor.size(1)
        if seq_len > 480: # 这里的480可能需要根据实际max_length调整
            logger.warning(f"序列长度 {seq_len} 超过 {baseconf.max_length}，已截断。")
        logger.info(f"单条样本处理完成。输入张量形状: {input_tensor.shape}, 掩码张量形状: {mask_tensor.shape}")
        return input_tensor, mask_tensor
    except Exception as e:
        logger.error(f"处理单条样本失败: {str(e)}")
        raise ValueError(f"处理单条样本失败: {str(e)}")

# --- 示例代码（被注释掉） ---
# if __name__ == '__main__':
#     try:
#         # 这是一个如何直接测试此文件功能的示例
#         baseconf = BaseConfig(bert_path=None, train_data="train.json", test_data="test.json", rel_data="relation.json")
#         a = torch.tensor([0, 1, 0, 0, 0, 0])
#         b = torch.tensor([0, 0, 0, 0, 1, 0])
#         subs = extract_sub(a, b)
#         print("提取的主实体:", subs)
#
#         obj_heads = torch.tensor([[[0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
#         obj_tails = torch.tensor([[[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]]])
#         obj_and_rel = extract_obj_and_rel(obj_heads[0], obj_tails[0])
#         print("提取的客实体和关系:", obj_and_rel)
#     except Exception as e:
#         logger.error(f"主程序执行错误: {str(e)}")