import torch
from torch.utils.data import DataLoader
from random import choice
from collections import defaultdict



# 导入必要的模块
from CasRel_RE.casrel_datautils.Base_Conf import BaseConfig
from CasRel_RE.casrel_datautils.data_loader import get_dataloader
from CasRel_RE.casrel_datautils.process import single_sample_process
from CasRel_RE.config import Config

locConf = Config()

def baseconfig():
    baseconf = BaseConfig(bert_path=locConf.bert_path,
                          train_data=locConf.train_path,
                          test_data=locConf.test_path,
                          dev_data=locConf.dev_path,
                          rel_data=locConf.rel_data,
                          batch_size=2)
    return baseconf

def get_all_dataloader():
    baseconf = baseconfig()
    dataloaders = get_dataloader(baseconf)
    return dataloaders


def extract_sub(pred_sub_heads, pred_sub_tails):
    '''
    :param pred_sub_heads: 模型预测出的主实体开头位置
    :param pred_sub_tails: 模型预测出的主实体尾部位置
    :return: subs列表里面对应的所有实体【head, tail】
    '''
    subs = []
    # 统计预测出所有值为1的元素索引位置
    heads = torch.arange(0, len(pred_sub_heads), device=locConf.device)[pred_sub_heads == 1]
    tails = torch.arange(0, len(pred_sub_tails), device=locConf.device)[pred_sub_tails == 1]
    for head, tail in zip(heads, tails):
        if tail >= head:
            subs.append((head.item(), tail.item()))
    return subs


def extract_obj_and_rel(obj_heads, obj_tails):
    '''

    :param obj_heads:  模型预测出的从实体开头位置以及关系类型
    :param obj_tails:  模型预测出的从实体尾部位置以及关系类型
    :return: obj_and_rels：元素形状：(rel_index, start_index, end_index)
    '''
    obj_heads = obj_heads.T
    obj_tails = obj_tails.T
    rel_count = obj_heads.shape[0]
    obj_and_rels = []

    for rel_index in range(rel_count):
        obj_head = obj_heads[rel_index]
        obj_tail = obj_tails[rel_index]
        objs = extract_sub(obj_head, obj_tail)
        if objs:
            for obj in objs:
                start_index, end_index = obj
                obj_and_rels.append((rel_index, start_index, end_index))
    return obj_and_rels


def convert_score_to_zero_one(tensor):
    '''
    以0.5为阈值，大于0.5的设置为1，小于0.5的设置为0
    '''
    tensor[tensor >= 0.5] = 1
    tensor[tensor < 0.5] = 0
    return tensor
