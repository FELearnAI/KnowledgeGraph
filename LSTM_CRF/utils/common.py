from LSTM_CRF.config import *

conf = Config()


# 构造数据集
def build_data():
    # 重要知识点：数据预处理和词汇表构建
    # 1. 初始化容器
    # 总样本datas、一个句子的sample_x sample_y
    # 词表vocab_list初始化包含特殊标记["PAD", 'UNK']
    datas = []  # 存储所有处理后的样本
    sample_x = []  # 临时存储一个句子的字符
    sample_y = []  # 临时存储一个句子的标签
    vocab_list = ["PAD", 'UNK']  # 词汇表，PAD用于填充，UNK用于未知字符
    
    # 2. 遍历训练集，将句子分段，保存在 datas
    # 重要知识点：逐行读取文件
    # for line in open() 这种方式：处理大文件时，逐行读取会节省内存
    for line in open(conf.train_path, 'r', encoding='utf-8'):
        # 2.1 切割 line，并获得 ['咳','B-SIGNS']，添加到 sample_x、sample_y
        line = line.strip().split('\t')  # 去除空白并按制表符分割
        
        # 如果 len(line) != 2 直接跳过（可能是空行或格式不正确）
        if len(line) != 2:
            continue
            
        char = line[0]  # 获取字符
        cate = line[-1]  # 获取BIO标签
        sample_x.append(char)  # 添加到字符列表
        sample_y.append(cate)  # 添加到标签列表
        
        # 2.2 构建词汇表
        if char not in vocab_list:
            vocab_list.append(char)  # 将新字符添加到词汇表
            
        # 2.3 句子截断处理
        # 重要知识点：基于标点符号的句子分割
        # 如果字符在标点符号列表中，将当前积累的样本添加到数据集并重置
        if char in ['?', '!', '。', '？', '！']:  # 句子结束标记
            datas.append([sample_x, sample_y])  # 将当前句子添加到数据集
            sample_x = []  # 重置字符列表
            sample_y = []  # 重置标签列表
            
    # 3. 构建 word2id 映射
    # 重要知识点：词汇表到ID的映射
    word2id = {word: idx for idx, word in enumerate(vocab_list)}
    
    # 4. 保存词表文件 vocab.txt
    write_file(vocab_list, conf.vocab_path)
    return datas, word2id


def write_file(vocab_list, file_path):  # 保存词表文件 vocab.txt
    # 换行拼接并写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab_list))


if __name__ == '__main__':
    datas, word2id = build_data()
    print('word2id:', word2id)
    print(len(datas))
    print(datas[:2])
    print(f"datas[0][0]即X =>{datas[0][0]}")
    print(f"datas[0][1]即Y=>{datas[0][1]}")
