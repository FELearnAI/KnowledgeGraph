import torch
import torch.nn as nn
import sys
import os
import time
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import torch.optim as optim
from model.bilstm_atten import BiLSTM_ATT
from config import Config
from utils.data_loader import get_all_loader
from lstmAtten_datautils.process import relation2id, word2id
import shutup

shutup.please()
conf = Config()

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def model2train():
    # 1 准备物料
    loaders = get_all_loader()
    train_loader = loaders["train"]
    test_loader = loaders["test"]
    
    model = BiLSTM_ATT(conf).to(conf.device)
    optimizer = optim.AdamW(model.parameters(), lr=conf.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    """
    当loss函数（如CrossEntropyLoss）设置reduction='mean'时,当前batch内所有样本loss的平均值
    epoch_loss = total_loss / total_samples
    """

    # 2. 开始训练
    start__time = time.time()
    best_f1 = -1000
    patience = 2
    # 3. 开启训练
    for epoch in range(conf.epochs):
        # 3.1 训练一个epoch
        for idx, (sent, pos1, pos2, label, _, _, _) in enumerate(tqdm(train_loader, desc='BiLSTM_ATT Training')):
            model.train()
            # 3.1.1 将数据移动到指定设备
            sent = sent.to(conf.device)
            pos1 = pos1.to(conf.device)
            pos2 = pos2.to(conf.device)
            label = label.to(conf.device)
            # 3.1.2 前向传播，计算损失
            output = model(sent, pos1, pos2)
            loss = loss_fn(output, label)
            # 3.1.3 反向传播，更新参数
            optimizer.zero_grad()
            loss.backward()
            # 3.1.4 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # 3.1.5 每20个batch打印训练日志
            if idx % 20 == 0 and idx:
                print(f"Epoch {epoch + 1}, Batch {idx}, 训练集loss: {loss.item():.4f}")
                precision, recall, f1, report, loss = model2dev(test_loader, model, loss_fn)
                print(f"验证集loss: {loss:.4f}, 验证集F1: {f1:.4f}")
                torch.save(model.state_dict(), conf.root_path / 'save' / 'test_best__.pth')
                print("F1提升！已保存最佳模型！")

        # 3.2 在每个epoch结束后，评估模型
        precision, recall, f1, report, loss = model2dev(test_loader, model, loss_fn)
        print(f"Epoch {epoch + 1}, 验证集loss: {loss:.4f}, 验证集F1: {f1:.4f}")
        print(report)

        # 3.3 检查是否需要保存模型或触发早停
        if f1 > best_f1:
            # 3.3.1 F1提升，更新最佳F1并保存模型
            best_f1 = f1
            no_improve_count = 0  # 重置未提升计数
            torch.save(model.state_dict(), conf.root_path / 'save' / 'test_best__.pth')
            print("F1提升！已保存最佳模型！")
            model.train()
        else:
            # 3.3.2 F1未提升，增加未提升计数
            no_improve_count += 1
            print(f"F1未提升，当前未提升轮次: {no_improve_count}/{patience}")

        if no_improve_count >= patience:
            # 3.3.3 达到早停条件，结束训练
            print(f"早停触发！连续 {patience} 轮次F1未提升，终止训练！")
            break

    # 3.4 保存最后一个轮次的模型
    torch.save(model.state_dict(), conf.root_path / 'save' / 'last__.pth')
    print(f"训练结束，总耗时：{time.time() - start__time:.2f}s")


def model2dev(test_loader, model, loss_fn):
    # 4 验证模型
    golds = []
    preds = []
    test_loss = 0
    model.eval()
    with torch.no_grad():  # 禁用梯度计算
        for sent, pos1, pos2, label, _, _, _ in tqdm(test_loader, desc="BiLSTM_Attention Model Testing"):
            # 4.1 将数据移到设备
            sent = sent.to(conf.device)
            pos1 = pos1.to(conf.device)
            pos2 = pos2.to(conf.device)
            label = label.to(conf.device)

            # 4.2 前向传播，计算损失
            output = model(sent, pos1, pos2)
            loss = loss_fn(output, label)
            test_loss += loss.item()

            # 4.3 获取预测结果
            output = torch.argmax(output, dim=-1)
            preds.extend(output.tolist())
            golds.extend(label.tolist())

    # 4.4 计算评估指标
    loss = test_loss / len(test_loader)
    precision = precision_score(golds, preds, average='micro')
    recall = recall_score(golds, preds, average='micro')
    f1 = f1_score(golds, preds, average='micro')
    report = classification_report(golds, preds)
    return precision, recall, f1, report, loss


if __name__ == '__main__':
    # 5 程序入口
    # text_list = open(conf.vocab_data_path, 'r', encoding='utf-8').read().strip().split('\n')
    # vocab_size = len(word2id(conf.base_conf))
    # print(f"词汇表大小: {vocab_size}")
    # tag_size = len(relation2id(conf.base_conf))
    # print(f"标签数量: {tag_size}")
    model2train()
