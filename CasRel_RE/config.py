import torch
from transformers import BertTokenizer
import json

class Config(object):
    def __init__(self):
        super().__init__()
        # 1 设备 cuda mps cpu
        # 由于MPS设备兼容性问题，暂时使用CPU
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = torch.device('cpu')
        # 2 bert相关：路径、tokenizer、输出维度
        self.bert_path = "/Users/dyq/PycharmProjects/KnowledgeGraph/CasRel_RE/bert-base-chinese"
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_dim = 768
        # 3 数据集路径 绝对路径不会出错
        self.train_path = "/Users/dyq/PycharmProjects/KnowledgeGraph/CasRel_RE/data/train.json"
        self.dev_path = "/Users/dyq/PycharmProjects/KnowledgeGraph/CasRel_RE/data/dev.json"
        self.test_path = "/Users/dyq/PycharmProjects/KnowledgeGraph/CasRel_RE/data/test.json"
        self.rel_data = "/Users/dyq/PycharmProjects/KnowledgeGraph/CasRel_RE/data/relation_raw.json"
        self.id2rel = json.load(open(self.rel_data, encoding="utf-8"))
        self.rel2id = {rel: int(id) for id, rel in self.id2rel.items()}
        self.num_rel = len(self.id2rel)  # 关系类型数量
        # 4 超参数
        self.epochs = 5
        self.learning_rate = 1e-5
        self.batch_size = 2
        self.bert_dim = 768


if __name__ == "__main__":
    config = Config()
    print(config.id2rel)
    print(config.rel2id)
