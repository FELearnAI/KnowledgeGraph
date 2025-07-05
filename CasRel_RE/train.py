# 导入必要的库
import os  # 用于与操作系统交互，如创建目录
import sys  # 用于与Python解释器交互
import time  # 用于计时
import pandas as pd  # 用于数据处理和分析，如此处的评估结果展示
from tqdm import tqdm  # 用于显示美观的进度条
import shutup  # 一个可以抑制第三方库警告的工具
from torch.optim import AdamW  # 导入AdamW优化器，它是Adam的一个改进版本，常用于Transformer模型

# 导入项目内的模块
from path_utils import get_model_save_path  # 导入路径管理工具，用于获取模型保存路径
from CasRel_RE.model.CasrelModel import *  # 从模型文件导入所有内容
from CasRel_RE.config import *  # 从配置文件导入所有内容
from CasRel_RE.utils.data_loader import *  # 从数据加载器工具导入所有内容
from CasRel_RE.model.CasrelModel import CasRel  # 明确地再次导入CasRel模型类

# 使用shutup抑制不必要的警告信息，让输出更整洁
shutup.please()

# 实例化配置类，创建一个全局配置对象
conf = Config()

def model2train():
    """
    模型训练的主函数。
    该函数会完成模型的初始化、数据加载、训练循环、验证、模型保存和早停等所有操作。
    """
    # --- 1. 初始化 ---
    print("--- 1. 初始化模型、数据加载器和优化器 ---")
    # 实例化CasRel模型
    model = CasRel(conf)
    # 将模型移动到在配置中指定的设备（CPU或GPU）
    model.to(conf.device)
    
    # 获取所有数据加载器（训练、验证、测试）
    dataloaders = get_all_dataloader()
    train_loader = dataloaders["train"]
    dev_loader = dataloaders["dev"]

    # 实例化AdamW优化器，传入模型的所有可训练参数和学习率
    optimizer = AdamW(model.parameters(), lr=conf.learning_rate)

    # --- 2. 设置模型保存路径 ---
    # 使用路径管理工具获取模型应该保存的目录
    save_dir = get_model_save_path('CasRel_RE')
    # 如果目录不存在，则创建它
    os.makedirs(save_dir, exist_ok=True)
    print(f"模型将保存在: {save_dir}")

    # --- 3. 初始化训练状态变量 ---
    start_time = time.time()  # 记录训练开始时间
    best_f1 = 0  # 用于记录验证集上最好的F1分数
    patience_counter = 0  # 用于早停机制的计数器
    print("--- 2. 开始训练 ---")

    # --- 4. 开始训练循环 ---
    for epoch in range(conf.epochs):
        # 将模型设置为训练模式
        model.train()
        print(f"\nEpoch {epoch + 1}/{conf.epochs}")
        
        # 遍历训练数据加载器，每个批次进行一次训练
        for inputs, labels in tqdm(train_loader, desc=f'训练中...'):
            # 将输入和标签数据移动到指定设备
            for key in inputs: inputs[key] = inputs[key].to(conf.device)
            for key in labels: labels[key] = labels[key].to(conf.device)

            # --- 前向传播 ---
            # 将输入数据传递给模型，获取预测结果
            logits = model(**inputs)
            # 根据预测结果和真实标签计算损失
            loss = model.compute_loss(**logits, **labels)

            # --- 反向传播和优化 ---
            # 在进行新的梯度计算之前，清除旧的梯度
            optimizer.zero_grad()
            # 反向传播，计算损失相对于模型参数的梯度
            loss.backward()
            # 根据梯度更新模型参数
            optimizer.step()

        # --- 5. 每个epoch结束后进行验证 ---
        print(f"Epoch {epoch + 1} 训练完成. 开始在验证集上评估...")
        # 调用评估函数，获取验证集上的所有指标
        sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1, df = model2dev(model, dev_loader)
        print(f"--- Epoch {epoch + 1} 验证结果 ---")
        print(df)
        print(f"三元组 F1: {triple_f1:.4f}, 最佳 F1: {best_f1:.4f}")

        # --- 6. 检查是否保存最佳模型及执行早停 ---
        if triple_f1 > best_f1:
            # 如果当前F1分数更高，则更新最佳分数并重置耐心计数器
            best_f1 = triple_f1
            patience_counter = 0
            # 保存当前最好的模型权重
            torch.save(model.state_dict(), os.path.join(save_dir, 'casrel_best.pth'))
            print(f"发现新的最佳模型！已保存至 'casrel_best.pth'")
        else:
            # 如果F1分数没有提升，则增加耐心计数器
            patience_counter += 1
            print(f"验证集F1分数未提升，耐心计数: {patience_counter}/{conf.patience}")

        # 如果连续多个epoch性能都没有提升，则触发早停
        if patience_counter >= conf.patience:
            print(f"连续 {conf.patience} 个epoch性能未提升，触发早停机制。")
            break

    # --- 7. 训练结束 ---
    # 保存最后一个epoch的模型状态
    torch.save(model.state_dict(), os.path.join(save_dir, 'casrel_last.pth'))
    print(f"训练结束。最终模型已保存至 'casrel_last.pth'")
    print(f'总训练耗时: {(time.time() - start_time) / 60:.2f} 分钟')

def model2dev(model, dev_loader):
    """
    在验证集或测试集上评估模型性能。
    
    参数:
        model (nn.Module): 要评估的CasRel模型。
        dev_loader (DataLoader): 验证集的数据加载器。
        
    返回:
        tuple: 包含所有评估指标的元组。
    """
    # 将模型设置为评估模式（这会关闭dropout等层）
    model.eval()

    # 初始化一个字典来存储主语和三元组抽取的TP, PRED, REAL数量
    metrics = {
        'sub': {'TP': 0, 'PRED': 0, 'REAL': 0},  # TP: True Positives, PRED: Predicted, REAL: Ground Truth
        'triple': {'TP': 0, 'PRED': 0, 'REAL': 0}
    }

    # 在评估时不需要计算梯度，以节省计算资源和内存
    with torch.no_grad():
        # 遍历验证集数据
        for inputs, labels in tqdm(dev_loader, desc="评估中..."):
            # 将数据移动到设备
            for key in inputs: inputs[key] = inputs[key].to(conf.device)
            for key in labels: labels[key] = labels[key].to(conf.device)

            # 获取模型预测结果
            logits = model(**inputs)

            # --- 从预测和标签中提取实体 ---
            # 遍历批次中的每个样本
            batch_size = inputs['input_ids'].shape[0]
            for i in range(batch_size):
                # 从预测的概率中提取主语
                pred_subs = extract_sub(logits['pred_sub_heads'][i], logits['pred_sub_tails'][i])
                # 从真实标签中提取主语
                true_subs = extract_sub(labels['sub_heads'][i], labels['sub_tails'][i])
                
                # 从预测的概率中提取宾语和关系
                pred_objs = extract_obj_and_rel(logits['pred_obj_heads'][i], logits['pred_obj_tails'][i])
                # 从真实标签中提取宾语和关系
                true_objs = extract_obj_and_rel(labels['obj_heads'][i], labels['obj_tails'][i])

                # --- 更新统计指标 ---
                metrics['sub']['PRED'] += len(pred_subs)
                metrics['sub']['REAL'] += len(true_subs)
                metrics['sub']['TP'] += len(set(pred_subs) & set(true_subs))

                metrics['triple']['PRED'] += len(pred_objs)
                metrics['triple']['REAL'] += len(true_objs)
                metrics['triple']['TP'] += len(set(pred_objs) & set(true_objs))

    # --- 计算精确率、召回率和F1分数 ---
    # 为避免除以零，在分母上加上一个极小值epsilon
    epsilon = 1e-9
    sub_p = metrics['sub']['TP'] / (metrics['sub']['PRED'] + epsilon)
    sub_r = metrics['sub']['TP'] / (metrics['sub']['REAL'] + epsilon)
    sub_f1 = 2 * sub_p * sub_r / (sub_p + sub_r + epsilon)

    triple_p = metrics['triple']['TP'] / (metrics['triple']['PRED'] + epsilon)
    triple_r = metrics['triple']['TP'] / (metrics['triple']['REAL'] + epsilon)
    triple_f1 = 2 * triple_p * triple_r / (triple_p + triple_r + epsilon)

    # 使用pandas DataFrame来美观地展示结果
    df = pd.DataFrame({
        'TP': [metrics['sub']['TP'], metrics['triple']['TP']],
        'PRED': [metrics['sub']['PRED'], metrics['triple']['PRED']],
        'REAL': [metrics['sub']['REAL'], metrics['triple']['REAL']],
        'Precision': [sub_p, triple_p],
        'Recall': [sub_r, triple_r],
        'F1': [sub_f1, triple_f1]
    }, index=['Subject', 'Triple'])

    return sub_p, sub_r, sub_f1, triple_p, triple_r, triple_f1, df

# --- 主程序入口 ---
if __name__ == '__main__':
    # 如果此脚本是作为主程序运行，则直接调用训练函数
    model2train()