import os
# 导入路径管理工具
from path_utils import get_model_save_path

from CasRel_RE.model.CasrelModel import *
from CasRel_RE.casrel_datautils.process import single_sample_process
from CasRel_RE.utils.data_loader import baseconfig
from CasRel_RE.config import Config

#
baseconf=baseconfig()
conf=Config()
model = CasRel(conf)
# 使用路径管理工具加载模型
model_path = get_model_save_path('CasRel_RE', 'best_f1.pth')
model.load_state_dict(torch.load(model_path, map_location=conf.device))
def model2predict(sample):
    # 读取关系字典 id2rel
    id2rel=conf.id2rel
    # 保存结果
    spo_list = []

    model.eval()
    with torch.no_grad():
        #获取单条样本的输入
        input_ids,mask=single_sample_process(baseconf,sample)
        #获取编码结果
        encoded_text = model.get_encoded_text(input_ids, mask)
        sub_heads, sub_tails = model.get_subs(encoded_text)
        pred_sub_heads = convert_score_to_zero_one(sub_heads)
        pred_sub_tails = convert_score_to_zero_one(sub_tails)
        pred_subs = extract_sub(pred_sub_heads.squeeze(), pred_sub_tails.squeeze())

        # 是否有 sub
        if len(pred_subs) != 0:
            for sub in pred_subs:
                # sub.shape 要与 pred_objs 保持一致
                sub = [sub]
                sub_head_idx = sub[0][0]
                sub_tail_idx = sub[0][1]

                # 初始化 model.get_objs_for_specific_sub() 的输入
                seq_len = len(input_ids[0])

                # 用来保存 单个sub 信息，预测客体关系的输入
                inner_sub_head2tail = torch.zeros(seq_len)

                # 获取输入主体位置信息，主体位置全部赋值为 1
                inner_sub_head2tail[sub_head_idx: sub_tail_idx + 1] = 1
                # sub_head2tail = inner_sub_head2tail.unsqueeze(0).to(conf.device)  # [None,None,:]等价于两次unsqueeze()
                sub_head2tail = inner_sub_head2tail[None, None, :].to(conf.device)

                # 获取主体长度
                inner_sub_len = torch.tensor([sub_tail_idx + 1 - sub_head_idx], dtype=torch.float)
                sub_len = inner_sub_len.unsqueeze(0).to(conf.device)

                # 预测 客体obj_rel 索引
                pred_obj_heads, pred_obj_tails = model.get_objs_for_specific_sub(sub_head2tail, sub_len, encoded_text)
                pred_obj_heads = convert_score_to_zero_one(pred_obj_heads)
                # print(f"pred_obj_heads 0与1分值转换之后=>{pred_obj_heads}")

                pred_obj_tails = convert_score_to_zero_one(pred_obj_tails)
                pred_objs = extract_obj_and_rel(pred_obj_heads[0], pred_obj_tails[0])

                # 要解码的原文本，有特殊符号
                text_list = conf.tokenizer.convert_ids_to_tokens(input_ids[0])

                # 如果 sub、obj 有一方不存在
                if len(sub) == 0 or len(pred_objs) == 0:
                    # print('没有识别出结果')
                    sample['predict'] = '没有识别出SPO结果'
                    return sample

                # 如果一个 sub 对应多个 obj
                if len(pred_objs) > 1:
                    sub = sub * len(pred_objs)

                # 组建 spo
                for same_sub, rel_obj in zip(sub, pred_objs):
                    # 初始化 1 个 spo
                    sub_spo = {}

                    # 拿到 sub 文本
                    sub_head, sub_tail = same_sub
                    sub_text = ''.join(text_list[sub_head: sub_tail + 1])
                    if '[PAD]' in sub_text:
                        continue
                    sub_spo['subject'] = sub_text

                    # 拿到 关系 文本
                    relation = id2rel[str(rel_obj[0])]
                    sub_spo['predicate'] = relation

                    # 拿到 obj 文本
                    obj_head, obj_tail = rel_obj[1], rel_obj[2]
                    obj_text = ''.join(text_list[obj_head: obj_tail + 1])
                    if '[PAD]' in obj_text:
                        continue
                    sub_spo['object'] = obj_text

                    # 每个 sub_spo三元组 都要加入 spo_list
                    spo_list.append(sub_spo)

    sample['predict'] = spo_list
    return sample


if __name__ == '__main__':
    # sample = '王菲深情演唱了歌曲《清风徐来》'
    sample = '《秋天的眼泪》是孟庭苇演唱的歌曲，收录于孟庭苇于1992年3月发行的专辑《谁的眼泪在飞》'
    # sample = '《今晚会在哪里醒来》是黄家强的一首粤语歌曲，由何启弘作词，黄家强作曲编曲并演唱，收录于2007年08月01日发行的专辑《她他》中'
    # sample = '刘万亭，男，汉族，1934年10月生，莱阳市沐浴店镇南旺村人'
    data={"text":sample}
    # sample = '白百何的处女座是《与青春有关的日子》，合作的演员是佟大为、陈羽凡'

    new_dict = model2predict(data)
    print("========================================================================================")
    print(new_dict)