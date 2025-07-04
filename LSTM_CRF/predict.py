from LSTM_CRF.model.BiLSTM import NERLSTM
from LSTM_CRF.model.BiLSTM_CRF import NERLSTM_CRF
from utils.data_loader import build_data
import torch
import torch.nn as nn
from LSTM_CRF.config import Config

# 加载配置和数据
conf = Config()
datas, word2id = build_data()
# 构建标签ID到标签名的映射
id2tag = {v: k for k, v in conf.tag2id.items()}

# 初始化模型
models = {'BiLSTM': NERLSTM, 'BiLSTM_CRF': NERLSTM_CRF}  # 模型字典
model_path = {
    'BiLSTM': r'/Users/dyq/PycharmProjects/KnowledgeGraph/LSTM_CRF/save_model/bilstm_best.pth',
    'BiLSTM_CRF': r'/Users/dyq/PycharmProjects/KnowledgeGraph/LSTM_CRF/save_model/bilstm_crf_best.pth'
}
# 根据配置选择模型
model = models[conf.model](conf)
# 加载预训练权重
model.load_state_dict(torch.load(model_path[conf.model], map_location=conf.device))
# 设置为评估模式
model.eval()


def model2predict(input_data):
    # 重要知识点：模型预测流程
    try:
        # 检查输入是否为字典并包含 'text' 字段
        if not isinstance(input_data, dict) or 'text' not in input_data:
            raise ValueError("输入必须是一个字典，且包含 'text' 字段")

        # 提取文本
        sample = input_data['text']

        # 准备输入数据
        # 重要知识点：文本预处理
        # 将字符转换为ID，未知字符使用UNK的ID
        x = [word2id.get(char, word2id.get("UNK", 0)) for char in sample]
        x_test = torch.tensor([x])  # 转换为tensor并添加batch维度
        mask = (x_test != 0).long()  # 创建mask，标记非填充位置

        # 预测
        # 重要知识点：不同模型的预测方式
        with torch.no_grad():  # 不计算梯度，节省内存
            if model.name == "BiLSTM":
                # BiLSTM模型输出发射分数矩阵
                out = model(x_test, mask)
                # 取每个位置概率最大的标签
                pred = torch.argmax(out, dim=-1)
                # 将ID转换回标签
                tags = [id2tag[i.item()] for i in pred[0]]
            else:
                # BiLSTM_CRF模型直接输出最优标签序列
                pred = model(x_test, mask)
                # 将ID转换回标签
                tags = [id2tag[i] for i in pred[0]]

            # 检查字符数与标签数是否匹配
            chars = list(sample)
            if len(chars) != len(tags):
                raise ValueError("字符数与标签数不匹配")
                
            # 提取实体
            return extract_ents(chars, tags)
    except Exception as e:
        raise Exception(f"预测出错: {str(e)}")


def extract_ents(chars, tags):
    # 重要知识点：BIO标注的实体提取
    ents = []  # 存储提取的实体
    current_ent = []  # 当前正在处理的实体
    current_type = None  # 当前实体的类型

    # 遍历字符和对应的标签
    for char, tag in zip(chars, tags):
        if tag.startswith("B-"):  # 实体开始
            # 如果之前有未处理完的实体，先添加到结果中
            if current_ent:
                ents.append((current_type, ''.join(current_ent)))
                current_ent = []
            # 开始新实体
            current_type = tag.split('-')[1]  # 提取实体类型
            current_ent.append(char)  # 添加字符到当前实体
        elif tag.startswith("I-") and current_ent:  # 实体内部
            current_ent.append(char)  # 继续添加字符到当前实体
        else:  # O标签或I标签但前面没有B标签
            # 如果之前有未处理完的实体，添加到结果中
            if current_ent:
                ents.append((current_type, ''.join(current_ent)))
                current_ent = []
                current_type = None

    # 处理最后一个实体
    if current_ent:
        ents.append((current_type, ''.join(current_ent)))

    # 返回实体到类型的映射
    return {ent: ent_type for ent_type, ent in ents}


if __name__ == '__main__':
    # 直接传入字典，模拟 API 的输入
    input_data = {"text": "李华的父亲患有冠心病及糖尿病，无手术外伤史。"}
    result = model2predict(input_data)
    print(result)