import torch
import torch.nn as nn
import torch.optim as optim
import time
from model.BiLSTM import *
from model.BiLSTM_CRF import *
from utils.data_loader import *
from tqdm import tqdm
# classification_report可以导出字典格式，修改参数：output_dict=True，可以将字典在保存为csv格式输出
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from config import *

conf = Config()

def model2train():
    # 重要知识点：模型训练流程
    # 1. 准备物料：训练和验证的dataloader、model、loss_fn、optimizer
    train_loader, dev_loader = get_data()  # 获取数据加载器
    models = {'BiLSTM_CRF': NERLSTM_CRF}  # 模型字典
    model = models[conf.model](conf)  # 根据配置选择模型
    model = model.to(conf.device)  # 将模型移动到指定设备(CPU/GPU/MPS)
    loss_fn = nn.CrossEntropyLoss()  # 定义损失函数(仅用于BiLSTM模型)
    optimizer = optim.AdamW(model.parameters(), lr=conf.lr)  # 定义优化器

    # 2. 开始训练：设置初始时间、最佳F1分数
    start_time = time.time()
    best_f1 = -1000  # 初始化最佳F1分数
    
    # 重要知识点：不同模型的训练流程差异
    # BiLSTM使用交叉熵损失，BiLSTM_CRF使用CRF的负对数似然损失
    if conf.model == 'BiLSTM':
        # 2.1 循环训练指定轮次
        for epoch in range(conf.epochs):

            # 2.2 逐批次训练
            for idx, (inputs, labels, mask) in enumerate(tqdm(train_loader, desc='BiLSTM Training')):
                model.train()  # 设置为训练模式
                # 将数据移动到指定设备
                x = inputs.to(conf.device)
                tags = labels.to(conf.device)
                mask = mask.to(conf.device)
                
                # 前向传播
                pred = model(x, mask)
                # 重要知识点：维度处理
                # 预测值是三维的[batch_size, seq_len, tag_size]，CrossEntropyLoss需要二维输入
                # 需要view(-1, len(conf.tag2id))将其变形为[batch_size*seq_len, tag_size]
                # 同时将标签变形为一维[batch_size*seq_len]
                pred = pred.view(-1, len(conf.tag2id))
                loss = loss_fn(pred, tags.view(-1))
                
                # 反向传播和优化
                loss.backward()  # 计算梯度
                optimizer.step()  # 更新参数
                optimizer.zero_grad()  # 清空梯度
                
                # 定期打印损失
                if idx % 50 == 0:
                    print('Epoch:%d\t\tLoss:%.4f' % (epoch, loss.item()))
                    
            # 2.3 每轮结束后在验证集上评估模型
            precision, recall, f1, report = model2dev(dev_loader, model, loss_fn)
            # 如果当前F1分数更好，保存模型
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), 'save_model/bilstm_best1.pth')
                print(report)  # 打印分类报告
                
        # 2.4 训练结束，打印总耗时
        end_time = time.time()
        print(f'训练总耗时：{end_time - start_time}')

    elif conf.model == 'BiLSTM_CRF':
        # 2.1 循环训练指定轮次
        for epoch in range(conf.epochs):

            # 2.2 逐批次训练
            for idx, (inputs, labels, mask) in enumerate(tqdm(train_loader, desc='BiLSTM_CRF Training')):
                model.train()  # 设置为训练模式
                # 将数据移动到指定设备
                x = inputs.to(conf.device)
                tags = labels.to(conf.device)
                mask = mask.to(conf.device)

                # 重要知识点：CRF模型的损失计算
                # 使用模型的log_likelihood方法计算负对数似然损失
                loss = model.log_likelihood(x, tags, mask)
                loss.backward()  # 计算梯度
                
                # 重要知识点：梯度裁剪
                # 防止梯度爆炸，将梯度值限制在一定范围内
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
                optimizer.step()  # 更新参数

                # 定期打印损失
                if idx % 1 == 0:
                    print('Epoch:%d\tBatch:%d\tLoss:%.4f' % (epoch, idx, loss.item()))
                    
            # 2.3 每轮结束后在验证集上评估模型
            precision, recall, f1, report = model2dev(dev_loader, model, loss_fn)
            # 如果当前F1分数更好，保存模型
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), 'save_model/bilstm_crf_best1.pth')
                print(report)  # 打印分类报告
                
        # 2.4 训练结束，打印总耗时
        end_time = time.time()
        print(f'训练总计耗时：{end_time - start_time}')


def model2dev(dev_loader, model, loss_fn=None):  # 模型验证函数
    # 重要知识点：模型评估流程
    # 1. 初始化评估指标
    aver_loss = 0  # 平均损失
    preds, golds = [], []  # 预测结果和真实标签

    print("Unique gold labels:", set(golds))
    print("Unique predicted labels:", set(preds))

    model.eval()  # 设置为评估模式

    # 2. 在验证集上进行预测
    for idx, (inputs, labels, mask) in enumerate(tqdm(dev_loader, desc="Model Validation")):
        print(f"idx=>{idx}")
        print(f"len(inputs)=>{len(inputs)}")
        # 将数据移动到指定设备
        val_x = inputs.to(conf.device)
        val_y = labels.to(conf.device)
        mask = mask.to(conf.device)
        
        # 3. 根据模型类型获取预测结果
        # 重要知识点：不同模型的预测方式
        predict = []
        if model.name == "BiLSTM":
            # BiLSTM模型输出是发射分数矩阵
            pred = model(val_x, mask)
            # print("===model2dev========BiLSTM预测=========")
            # print(f"pred=>{pred.shape}")
            # 取每个位置概率最大的标签作为预测结果
            predict = torch.argmax(pred, dim=-1).tolist()
            # print(f"predict=torch.argmax(pred, dim=-1)=>{torch.argmax(pred, dim=-1).shape}")
            # print(f"predict=torch.argmax(pred, dim=-1).tolist() =>{predict}")
            # 计算损失
            pred = pred.view(-1, len(conf.tag2id))
            val_loss = loss_fn(pred, val_y.view(-1))
            aver_loss += val_loss.item()
        elif model.name == "BiLSTM_CRF":
            # BiLSTM_CRF模型直接输出最优标签序列
            predict = model(val_x, mask)  # CRF前向传播的结果本身就是list
            # 计算损失
            loss = model.log_likelihood(val_x, val_y, mask)
            aver_loss += loss.item()

        # 4. 处理预测结果，去除填充部分
        # 重要知识点：处理填充标记
        # 需要去除标签中的填充部分(值为11)，只评估实际内容
        for one_pred, one_true in zip(predict, val_y.tolist()):
            # print("处理 预测值 predict 和 val_y 中的 pad 部分 ======>：")
            # print(f"predict[2]==>{predict[:2]}")
            # print(f"val_y.tolist()==>{val_y.tolist()[:2]}")
            # 计算填充长度和有效长度
            pad_len = one_true.count(11)  # 填充标记的数量
            no_pad_len = len(one_true) - pad_len  # 有效内容的长度
            # 只保留有效部分的预测和真实标签
            preds.extend(one_pred[:no_pad_len])
            golds.extend(one_true[:no_pad_len])

    # 5. 计算评估指标
    # 重要知识点：NER评估指标
    aver_loss /= (len(dev_loader) * conf.batch_size)  # 计算平均损失
    # 使用宏平均计算精确率、召回率和F1分数
    # macro表示宏平均，平等对待每个类别，适合类别不平衡的情况
    precision = precision_score(golds, preds, average='macro')
    recall = recall_score(golds, preds, average='macro')
    f1 = f1_score(golds, preds, average='macro')
    # 生成详细的分类报告
    report = classification_report(golds, preds)
    return precision, recall, f1, report


if __name__ == '__main__':
    model2train()