import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CasRel_RE.model.CasrelModel import *
from CasRel_RE.config import *
import time
import pandas as pd
from tqdm import tqdm
from CasRel_RE.utils.data_loader import *
import shutup
from CasRel_RE.model.CasrelModel import CasRel
from torch.optim import AdamW
shutup.please()

conf=Config()


def model2train():
    #实例化一个模型
    model=CasRel(conf)
    #获取数据dataloader
    dataloaders=get_all_dataloader()
    train_loader=dataloaders["train"]
    test_loader = dataloaders["test"]
    dev_loader = dataloaders["dev"]

    #实例化优化器
    optimizer=AdamW(model.parameters(),lr=conf.learning_rate)

    # 创建保存模型的绝对路径
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save_model')
    os.makedirs(save_dir, exist_ok=True)
    
    start_time = time.time()
    best_f1 = 0
    print("开始训练=============================")
    for epoch in range(conf.epochs):

        for idx, (inputs, labels) in enumerate(tqdm(train_loader,desc='CasReL模型训练......')):
            model.train()
            logits = model(**inputs)
            loss = model.compute_loss(**logits, **labels)

            #梯度清零
            optimizer.zero_grad()

            #损失反向传播
            loss.backward()

            #参数更新
            optimizer.step()
            # 遍历dataloader数据
            if idx % (len(train_loader) // 1000) == 0:
                print("Epoch:%d\tStep:%d\tLoss:%.4f" % (epoch, idx, loss.item()))
                torch.save(model.state_dict(), os.path.join(save_dir, 'casrel_current_model.pth'))
                sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1, df = model2dev(model, dev_loader)
                print(f"验证指标df---------------\n{df}")
        # 模型验证
        result = model2dev(model, dev_loader)
        sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1, df = result
        print('''Epoch:%d\tSub_precision:%.4f\tSub_recall:%.4f\tSub_f1:%.4f\t
        Triple_precision:%.4f\tTriple_recall:%.4f\tTriple_f1:%.4f''' %
              (epoch, sub_precision, sub_recall, sub_f1,
               triple_precision, triple_recall, triple_f1))

        if triple_f1 > best_f1:
            best_f1 = triple_f1
            torch.save(model.state_dict(), os.path.join(save_dir, 'dt_best_f1.pth'))

    torch.save(model.state_dict(), os.path.join(save_dir, 'dt_last_model.pth'))
    print('训练总耗时：', time.time() - start_time)


def model2dev(model, dev_loader):
    model.eval()

    # 使用字典组织指标字典
    metrics = {
        'sub': {'TP': 0, 'PRED': 0, 'REAL': 0},
        'triple': {'TP': 0, 'PRED': 0, 'REAL': 0}
    }

    # 数据收集阶段
    for inputs, labels in tqdm(dev_loader,desc='CasReL模型验证......'):
        logits = model(**inputs)

        # 将预测值转化为 0 和 1
        pred_sub_heads = convert_score_to_zero_one(logits['pred_sub_heads'])
        pred_sub_tails = convert_score_to_zero_one(logits['pred_sub_tails'])
        pred_obj_heads = convert_score_to_zero_one(logits['pred_obj_heads'])
        pred_obj_tails = convert_score_to_zero_one(logits['pred_obj_tails'])

        # 标签值处理
        sub_heads = convert_score_to_zero_one(labels['sub_heads'])
        sub_tails = convert_score_to_zero_one(labels['sub_tails'])
        obj_heads = convert_score_to_zero_one(labels['obj_heads'])
        obj_tails = convert_score_to_zero_one(labels['obj_tails'])

        batch_size = inputs['input_ids'].shape[0]
        for batch_index in range(batch_size):
            # 提取主体
            pred_subs = extract_sub(pred_sub_heads[batch_index].squeeze(1),
                                    pred_sub_tails[batch_index].squeeze(1))
            true_subs = extract_sub(sub_heads[batch_index].squeeze(),
                                    sub_tails[batch_index].squeeze())

            # 提取客体-关系
            pred_objs = extract_obj_and_rel(pred_obj_heads[batch_index],
                                            pred_obj_tails[batch_index])
            true_objs = extract_obj_and_rel(obj_heads[batch_index],
                                            obj_tails[batch_index])

            # 更新指标
            metrics['sub']['PRED'] += len(pred_subs)
            metrics['sub']['REAL'] += len(true_subs)
            for true_sub in true_subs:
                if true_sub in pred_subs:
                    metrics['sub']['TP'] += 1

            metrics['triple']['PRED'] += len(pred_objs)
            metrics['triple']['REAL'] += len(true_objs)
            for true_obj in true_objs:
                if true_obj in pred_objs:
                    metrics['triple']['TP'] += 1

    # 计算指标
    sub_precision = metrics['sub']['TP'] / (metrics['sub']['PRED'] + 1e-9)
    sub_recall = metrics['sub']['TP'] / (metrics['sub']['REAL'] + 1e-9)
    sub_f1 = 2 * sub_precision * sub_recall / (sub_precision + sub_recall + 1e-9)

    triple_precision = metrics['triple']['TP'] / (metrics['triple']['PRED'] + 1e-9)
    triple_recall = metrics['triple']['TP'] / (metrics['triple']['REAL'] + 1e-9)
    triple_f1 = 2 * triple_precision * triple_recall / (triple_precision + triple_recall + 1e-9)

    # 创建最终的DataFrame
    df = pd.DataFrame({
        'TP': [metrics['sub']['TP'], metrics['triple']['TP']],
        'PRED': [metrics['sub']['PRED'], metrics['triple']['PRED']],
        'REAL': [metrics['sub']['REAL'], metrics['triple']['REAL']],
        'p': [sub_precision, triple_precision],
        'r': [sub_recall, triple_recall],
        'f1': [sub_f1, triple_f1]
    }, index=['sub', 'triple'])

    return sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1, df


if __name__ == '__main__':
    model2train()


