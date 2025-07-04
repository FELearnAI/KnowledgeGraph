from .Base_Conf import BaseConfig
import torch
from random import choice
from collections import defaultdict
import logging
torch.set_printoptions(threshold=torch.inf)  # threshold 设置为无穷大，避免省略
# 设置打印选项
torch.set_printoptions(linewidth=200)  # 设置足够大的行宽，避免换行


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_head_idx(source, target):
    """查找目标序列在源序列中的起始索引。
    使用说明：用于定位实体在输入序列中的起始位置。
    参数：
        source (list): 源序列。
        target (list): 目标序列。
    返回：
        int: 起始索引，若未找到则返回 -1。
    鲁棒性：处理空序列情况。
    """
    if not source or not target:
        return -1
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1

def create_label(inner_triples, inner_input_ids, seq_len, baseconf):
    """创建标签张量。
    使用说明：根据三元组和输入序列生成实体和关系的张量表示。
    参数：
        inner_triples (list): 三元组列表。
        inner_input_ids (list): 输入序列的 token IDs。
        seq_len (int): 序列长度。
        baseconf (BaseConfig): 配置实例，包含 tokenizer 和 num_rel。
    返回：
        tuple: 包含多个张量的元组（sub_len, sub_head2tail, sub_heads, sub_tails, obj_heads, obj_tails）。
    鲁棒性：处理空 triples 和无效索引。
    """
    if not inner_triples or not inner_input_ids:
        return torch.tensor([0], dtype=torch.float), torch.zeros(seq_len), torch.zeros(seq_len), torch.zeros(seq_len), \
               torch.zeros((seq_len, baseconf.num_rel)), torch.zeros((seq_len, baseconf.num_rel))

    inner_sub_heads, inner_sub_tails = torch.zeros(seq_len), torch.zeros(seq_len)
    inner_obj_heads = torch.zeros((seq_len, baseconf.num_rel))
    inner_obj_tails = torch.zeros((seq_len, baseconf.num_rel))
    inner_sub_head2tail = torch.zeros(seq_len)
    inner_sub_len = torch.tensor([1], dtype=torch.float)
    s2ro_map = defaultdict(list)

    for inner_triple in inner_triples:
        try:
            inner_triple = (
                baseconf.tokenizer(inner_triple.get('subject', ''), add_special_tokens=False)['input_ids'],
                baseconf.rel2id.get(inner_triple.get('predicate', ''), 0),
                baseconf.tokenizer(inner_triple.get('object', ''), add_special_tokens=False)['input_ids']
            )
            sub_head_idx = find_head_idx(inner_input_ids, inner_triple[0])
            obj_head_idx = find_head_idx(inner_input_ids, inner_triple[2])
            if sub_head_idx != -1 and obj_head_idx != -1 and sub_head_idx < seq_len and obj_head_idx < seq_len:
                sub = (sub_head_idx, sub_head_idx + len(inner_triple[0]) - 1)
                s2ro_map[sub].append((obj_head_idx, obj_head_idx + len(inner_triple[2]) - 1, inner_triple[1]))
        except Exception as e:
            logger.warning(f"处理三元组 {inner_triple} 失败: {str(e)}")
            continue

    if s2ro_map:
        for s in s2ro_map:
            if s[0] < seq_len and s[1] < seq_len:
                inner_sub_heads[s[0]] = 1
                inner_sub_tails[s[1]] = 1
        try:
            sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
            if sub_head_idx < seq_len and sub_tail_idx < seq_len:
                inner_sub_head2tail[sub_head_idx:sub_tail_idx + 1] = 1
                inner_sub_len = torch.tensor([sub_tail_idx + 1 - sub_head_idx], dtype=torch.float)
                for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                    if ro[0] < seq_len and ro[1] < seq_len and ro[2] < baseconf.num_rel:
                        inner_obj_heads[ro[0]][ro[2]] = 1
                        inner_obj_tails[ro[1]][ro[2]] = 1
        except IndexError:
            pass

    # logger.info("create_label 处理完成")
    return inner_sub_len, inner_sub_head2tail, inner_sub_heads, inner_sub_tails, inner_obj_heads, inner_obj_tails

def collate_fn(batch, baseconf):
    """批次数据整理函数。
    使用说明：将批次数据转换为 PyTorch 张量，供模型使用。根据 baseconf.max_length 或批次中最长句子长度进行 padding，
    若超过 BERT 最大长度，则截断到 max_length 或 BERT 最大长度。
    参数：
        batch (list): 批次数据，包含文本和三元组。
        baseconf (BaseConfig): 配置实例，包含 tokenizer、device 和 max_length。
    返回：
        tuple: 包含 inputs 和 labels 的字典。
    鲁棒性：处理空批次、序列长度不一致和超长序列。
    """
    if not batch:
        return {'input_ids': torch.tensor([]).to(baseconf.device), 'mask': torch.tensor([]).to(baseconf.device),
                'sub_head2tail': torch.tensor([]).to(baseconf.device), 'sub_len': torch.tensor([]).to(baseconf.device)}, \
               {'sub_heads': torch.tensor([]).to(baseconf.device), 'sub_tails': torch.tensor([]).to(baseconf.device),
                'obj_heads': torch.tensor([]).to(baseconf.device), 'obj_tails': torch.tensor([]).to(baseconf.device)}

    text_list = [value[0] for value in batch]
    triple = [value[1] for value in batch]
    try:
        # 获取 BERT 最大序列长度
        bert_max_length = baseconf.bert_model.config.max_position_embeddings  # 通常为 512
        # 计算批次中最长句子的长度
        max_seq_len = max(len(baseconf.tokenizer.encode(text, add_special_tokens=True)) for text in text_list) if text_list else 0
        # 确定 max_length：使用 baseconf.max_length 或批次最大长度，但不超过 BERT 最大长度
        effective_max_length = baseconf.max_length if baseconf.max_length and baseconf.max_length <= bert_max_length else min(max_seq_len, bert_max_length)

        text = baseconf.tokenizer.batch_encode_plus(
            text_list,
            padding=True,
            truncation=True,
            max_length=effective_max_length,
            return_tensors='pt'
        )
        # 使用 len() 获取批次大小和序列长度
        batch_size = len(text['input_ids'])
        seq_len = len(text['input_ids'][0])
        # 验证序列长度一致性
        for i in range(batch_size):
            if len(text['input_ids'][i]) != seq_len:
                logger.warning(f"样本 {i} 的序列长度 {len(text['input_ids'][i])} 与批次长度 {seq_len} 不一致")
                raise ValueError(f"批次中序列长度不一致，请检查数据或调整 max_length")
        if seq_len > effective_max_length:
            logger.warning(f"序列长度 {seq_len} 超过有效最大长度 {effective_max_length}")
            raise ValueError(f"序列长度超过有效最大长度 {effective_max_length}，请调整数据或 max_length")
        # logger.info(f"批次大小: {batch_size}, 序列长度: {seq_len}, 有效最大长度: {effective_max_length}")
    except Exception as e:
        logger.error(f"tokenization 失败: {str(e)}")
        return {'input_ids': torch.tensor([]).to(baseconf.device), 'mask': torch.tensor([]).to(baseconf.device),
                'sub_head2tail': torch.tensor([]).to(baseconf.device), 'sub_len': torch.tensor([]).to(baseconf.device)}, \
               {'sub_heads': torch.tensor([]).to(baseconf.device), 'sub_tails': torch.tensor([]).to(baseconf.device),
                'obj_heads': torch.tensor([]).to(baseconf.device), 'obj_tails': torch.tensor([]).to(baseconf.device)}

    sub_heads, sub_tails, obj_heads, obj_tails, sub_len, sub_head2tail = [], [], [], [], [], []
    for batch_index in range(batch_size):
        inner_input_ids = text['input_ids'][batch_index].tolist()  # 转换为列表以兼容 find_head_idx
        inner_triples = triple[batch_index]
        results = create_label(inner_triples, inner_input_ids, seq_len, baseconf)
        sub_len.append(results[0])
        sub_head2tail.append(results[1])
        sub_heads.append(results[2])
        sub_tails.append(results[3])
        obj_heads.append(results[4])
        obj_tails.append(results[5])

    try:
        input_ids = text['input_ids'].to(baseconf.device)
        mask = text['attention_mask'].to(baseconf.device)
        sub_heads = torch.stack(sub_heads).to(baseconf.device)
        sub_tails = torch.stack(sub_tails).to(baseconf.device)
        sub_len = torch.stack(sub_len).to(baseconf.device)
        sub_head2tail = torch.stack(sub_head2tail).to(baseconf.device)
        obj_heads = torch.stack(obj_heads).to(baseconf.device)
        obj_tails = torch.stack(obj_tails).to(baseconf.device)
    except RuntimeError as e:
        logger.error(f"张量堆叠失败: {str(e)}")
        return {'input_ids': torch.tensor([]).to(baseconf.device), 'mask': torch.tensor([]).to(baseconf.device),
                'sub_head2tail': torch.tensor([]).to(baseconf.device), 'sub_len': torch.tensor([]).to(baseconf.device)}, \
               {'sub_heads': torch.tensor([]).to(baseconf.device), 'sub_tails': torch.tensor([]).to(baseconf.device),
                'obj_heads': torch.tensor([]).to(baseconf.device), 'obj_tails': torch.tensor([]).to(baseconf.device)}

    inputs = {'input_ids': input_ids, 'mask': mask, 'sub_head2tail': sub_head2tail, 'sub_len': sub_len}
    labels = {'sub_heads': sub_heads, 'sub_tails': sub_tails, 'obj_heads': obj_heads, 'obj_tails': obj_tails}

    # logger.info("collate_fn 处理完成")
    # print("casrel模型需要用到的输入inputs==>", inputs)
    # print("input_ids==>", input_ids.shape)
    # print("mask==>", mask.shape)
    # print("sub_head2tail==>", sub_head2tail.shape)
    # print("sub_len==>", sub_len.shape)
    # print("casrel模型需要用来计算loss的label==>", labels)
    # print("sub_heads==>", sub_heads.shape)
    # print("sub_tails==>", sub_tails.shape)
    # print("obj_heads==>", obj_heads.shape)
    # print("obj_tails==>", obj_tails.shape)

    return inputs, labels

def extract_sub(pred_sub_heads, pred_sub_tails, baseconf):
    """提取主实体。
    使用说明：根据预测的头部和尾部位置提取实体范围。
    参数：
        pred_sub_heads (tensor): 预测的主实体头部。
        pred_sub_tails (tensor): 预测的主实体尾部。
        baseconf (BaseConfig): 配置实例，包含 device。
    返回：
        list: 包含 (head, tail) 的实体列表。
    鲁棒性：处理空张量和无效索引。
    """
    if pred_sub_heads is None or pred_sub_tails is None or len(pred_sub_heads) == 0 or len(pred_sub_tails) == 0:
        return []
    subs = []
    try:
        heads = torch.arange(0, len(pred_sub_heads), device=baseconf.device)[pred_sub_heads == 1]
        tails = torch.arange(0, len(pred_sub_tails), device=baseconf.device)[pred_sub_tails == 1]
        for head, tail in zip(heads, tails):
            if tail >= head:
                subs.append((head.item(), tail.item()))
    except Exception as e:
        logger.warning(f"提取主实体失败: {str(e)}")
    return subs

def extract_obj_and_rel(obj_heads, obj_tails, baseconf):
    """提取客实体和关系。
    使用说明：根据预测的头部和尾部位置及关系类型提取三元组。
    参数：
        obj_heads (tensor): 预测的客实体头部及关系。
        obj_tails (tensor): 预测的客实体尾部及关系。
        baseconf (BaseConfig): 配置实例，包含 num_rel。
    返回：
        list: 包含 (rel_index, start_index, end_index) 的列表。
    鲁棒性：处理空张量和维度不匹配。
    """
    if obj_heads is None or obj_tails is None or obj_heads.size(0) != obj_tails.size(0):
        return []
    try:
        obj_heads = obj_heads.T
        obj_tails = obj_tails.T
        rel_count = obj_heads.size(0)
        obj_and_rels = []
        for rel_index in range(rel_count):
            obj_head = obj_heads[rel_index]
            obj_tail = obj_tails[rel_index]
            objs = extract_sub(obj_head, obj_tail, baseconf)
            if objs:
                for obj in objs:
                    start_index, end_index = obj
                    obj_and_rels.append((rel_index, start_index, end_index))
    except Exception as e:
        logger.warning(f"提取客实体和关系失败: {str(e)}")
    return obj_and_rels

def convert_score_to_zero_one(tensor, baseconf):
    """将分数转换为 0 或 1。
    使用说明：以 0.5 为阈值，将张量值转换为二值。
    参数：
        tensor (tensor): 输入张量。
        baseconf (BaseConfig): 配置实例，包含 device。
    返回：
        tensor: 二值化后的张量。
    鲁棒性：处理空张量。
    """
    if tensor is None or len(tensor) == 0:
        return torch.tensor([], device=baseconf.device)
    tensor = tensor.clone()  # 避免修改原始张量
    tensor[tensor >= 0.5] = 1
    tensor[tensor < 0.5] = 0
    return tensor


def single_sample_process(baseconf, sample_data):
    """处理单条样本，生成用于模型预测的 input_tensor 和 mask_tensor。
    使用说明：根据文本字符长度判断，若不超过 360 字符直接编码，若超过则截断至 360 字符后编码，始终以批处理格式返回。
    参数：
        baseconf (BaseConfig): 配置实例，包含 tokenizer 和 max_length。
        sample_data (dict): 包含 'text' 键的字典，表示单条样本数据。
    返回：
        tuple: (input_tensor, mask_tensor) - 输入张量和注意力掩码张量，形状为 [1, seq_len]。
    鲁棒性：处理无效输入和长度超限情况，保持与训练数据一致的批处理格式。
    """
    if not sample_data or 'text' not in sample_data:
        logger.error("样本数据无效，必须包含 'text' 键")
        raise ValueError("样本数据无效，必须包含 'text' 键")

    text = sample_data['text']
    max_char_length = 480  # 字符长度限制
    try:
        # 判断文本字符长度
        char_length = len(text)
        logger.info(f"文本字符长度: {char_length}")

        if char_length > max_char_length:
            logger.warning(f"文本字符长度 {char_length} 超过 {max_char_length}，已截断至 {max_char_length} 字符")
            text = text[:max_char_length]  # 截断至 360 字符

        # 使用 tokenizer 批处理编码单条样本
        encoded = baseconf.tokenizer(
            [text],  # 批处理格式，单条样本作为列表
            padding=True,
            truncation=True,
            max_length=baseconf.max_length,
            return_tensors='pt'
        )
        # 获取 input_ids 和 attention_mask
        input_tensor = encoded['input_ids']  # 形状: [1, seq_len]
        mask_tensor = encoded['attention_mask']  # 形状: [1, seq_len]

        # 验证长度
        seq_len = input_tensor.size(1)
        if seq_len > 480:
            logger.warning(f"序列长度 {seq_len} 超过 文本长度360限制 {baseconf.max_length}，已截断")
        logger.info(f"单条样本处理完成，输入张量形状: {input_tensor.shape}, 掩码张量形状: {mask_tensor.shape}")
        return input_tensor, mask_tensor
    except Exception as e:
        logger.error(f"处理单条样本失败: {str(e)}")
        raise ValueError(f"处理单条样本失败: {str(e)}")


# if __name__ == '__main__':
#     try:
#         baseconf = BaseConfig(bert_path=None, train_data="train.json", test_data="test.json", rel_data="relation.json")
#         a = torch.tensor([0, 1, 0, 0, 0, 0])
#         b = torch.tensor([0, 0, 0, 0, 1, 0])
#         subs = extract_sub(a, b, baseconf)
#         print("提取的主实体:", subs)
#
#         obj_heads = torch.tensor([[[0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
#         obj_tails = torch.tensor([[[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]]])
#         obj_and_rel = extract_obj_and_rel(obj_heads[0], obj_tails[0], baseconf)
#         print("提取的客实体和关系:", obj_and_rel)
#     except Exception as e:
#         logger.error(f"主程序执行错误: {str(e)}")