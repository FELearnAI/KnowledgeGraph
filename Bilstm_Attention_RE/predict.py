import torch
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.data_loader import get_loader, process_single_sample
from model.bilstm_atten import BiLSTM_ATT
from config import Config
from lstmAtten_datautils.process import relation2id, word2id
from tqdm import tqdm
import shutup

shutup.please()

# 1 准备参数
# 本地配置文件config定义数据路径
conf = Config()

# 2 准备数据、模型
model = BiLSTM_ATT(conf)
model.load_state_dict(torch.load(conf.root_path / 'save' / 'test_best__.pth', map_location=conf.device))
# 但保险起见，再次将模型整体放到设备上
model.to(conf.device)
print(model)

# 得到relation
relation2id = relation2id(conf.base_conf)
id2relation = {v: k for k, v in relation2id.items()}


# 3 开始预测
def model2predict(data):
    single_sample = data
    sents_tensor, pos_e1_tensor, pos_e2_tensor, labels_tensor, sents, pos_e1, pos_e2, ents = process_single_sample(
        single_sample, conf.base_conf)

    # 输入转移到模型设备
    sents_tensor = sents_tensor.to(conf.device)
    pos_e1_tensor = pos_e1_tensor.to(conf.device)
    pos_e2_tensor = pos_e2_tensor.to(conf.device)

    model.eval()
    with torch.no_grad():
        output = model(sents_tensor, pos_e1_tensor, pos_e2_tensor)
        pred = torch.argmax(output, dim=1).tolist()
        predict_label = id2relation[pred[0]]
        print('模型预测: ', predict_label)
        return predict_label


def load_test_data(file_path, num_samples=20):
    """从test.txt文件中加载前num_samples条数据"""
    test_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            line = line.strip()
            if line:
                parts = line.split(' ', 3)  # 分割成4部分：作品名、实体1、关系、描述
                if len(parts) >= 4:
                    work_name, ent1, relation, text = parts
                    # 构造数据格式
                    data = {
                        "text": line,  # 使用完整的行作为文本
                        "ent1": ent1,
                        "ent2": work_name
                    }
                    test_data.append((data, relation))  # 保存数据和真实关系
    return test_data


if __name__ == '__main__':
    # 加载测试数据
    test_file_path = '/Users/dyq/PycharmProjects/KnowledgeGraph/Bilstm_Attention_RE/data/test.txt'
    test_samples = load_test_data(test_file_path, 20)
    
    print(f"加载了 {len(test_samples)} 条测试数据")
    print("="*50)
    
    correct_predictions = 0
    total_predictions = len(test_samples)
    

    # 情感高烧 甄妮 歌手 甄妮--情感高烧专辑名称:至爱2000精选
    # 西游记 杨洁 导演 而《西游记》能有如今的成功，有今天的地位，这都是和杨洁导演分不开的
    # data = {"text": "甄妮--情感高烧专辑名称:至爱2000精选", "ent1": "情感高烧", "ent2": "甄妮"}
    # model2predict(data)

    # 逐个测试
    for i, (data, true_relation) in enumerate(test_samples, 1):
        print(f"\n测试样本 {i}:")
        print(f"文本: {data['text']}")
        print(f"实体1: {data['ent1']}")
        print(f"实体2: {data['ent2']}")
        print(f"真实关系: {true_relation}")
        
        # 进行预测
        predicted_relation = model2predict(data)
        
        # 检查预测是否正确
        is_correct = predicted_relation == true_relation
        if is_correct:
            correct_predictions += 1
            print(f"✓ 预测正确")
        else:
            print(f"✗ 预测错误")
        
        print("-" * 30)
    
    # 计算准确率
    accuracy = correct_predictions / total_predictions * 100
    print(f"\n测试结果统计:")
    print(f"总样本数: {total_predictions}")
    print(f"正确预测: {correct_predictions}")
    print(f"错误预测: {total_predictions - correct_predictions}")
    print(f"准确率: {accuracy:.2f}%")
