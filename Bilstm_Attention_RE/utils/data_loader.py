from lstmAtten_datautils.data_loader import get_loader, process_single_sample
from lstmAtten_datautils.base_Conf import BaseConfig
from config import Config


# 本地配置文件config定于数据路径
locConf = Config()
base_conf = BaseConfig(
    train_data_path=locConf.train_data_path,
    test_data_path=locConf.test_data_path,
    vocab_path=locConf.vocab_data_path,
    rel_path=locConf.rel_data_path,
    max_length=locConf.max_length, #70
    batch_size=locConf.batch_size #32
)

# 加载数据
def get_all_loader():
    loaders = get_loader(base_conf)  # 基于BseConfig配置，返回dataloader
    return loaders


loaders = get_all_loader()
print(loaders.items())
# 遍历并打印一个批次
for dataset_type, loader in loaders.items():
    for sents_tensor, pos_e1_tensor, pos_e2_tensor, labels_tensor, sents, labels, ents in loader:
        print(f"{dataset_type}句子张量:", sents_tensor.shape)
        print(f"{dataset_type}实体1位置张量:", pos_e1_tensor.shape)
        print(f"{dataset_type}实体2位置张量:", pos_e2_tensor.shape)
        print(f"{dataset_type}标签张量:", labels_tensor.shape)
        print(f"{dataset_type}原始句子:", sents)
        print(f"{dataset_type}原始标签:", labels)
        print(f"{dataset_type}实体对:", ents)
        break


#单条样本的数据处理,用于单条样本预测。
#labels_tensor默认为None，这里为了保持与其他数据处理一致性
single_sample = {"text": "温暖的家歌曲《温暖的家》由余致迪作词，曲作者未知，蔡国庆演唱", "ent1": "温暖的家", "ent2": "余致迪"}
sents_tensor, pos_e1_tensor, pos_e2_tensor, labels_tensor, sents, pos_e1, pos_e2, ents = process_single_sample(single_sample, base_conf)

print("pos_e1_tensor",pos_e1_tensor)
print("单条样本张量:", sents_tensor)