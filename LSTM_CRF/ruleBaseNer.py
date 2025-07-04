# 步骤1：导入库并忽略警告
import jieba, jieba.posseg as pseg, re, warnings

warnings.filterwarnings('ignore')

# 步骤2：定义组织机构标识列表
# 变量名：org_tag 作用：存储机构后缀（如“公司”“总局”），用于识别组织机构名称的结束标志
org_tag = ['公司', '有限公司', '大学', '政府', '人民政府', '总局']


# 步骤3：定义提取组织机构函数
# 函数名：extract_org, 参数：text（输入文本） 作用：通过 jieba 分词和正则表达式提取组织机构名称

def extract_org(text):
    # 步骤3.1：使用 jieba 词性标注分词
    # 变量名：words_flags  作用：存储分词和词性结果，格式为 [(word, flag), ...]
    words_flags = pseg.lcut(text)
    print(words_flags)

    # 步骤3.2：初始化词和标签列表
    # 变量名：列表words, lables_list 作用：words 存词语，lables_list 存 BIO 标签（B: 地名, E: 机构后缀, O: 其他）
    words = []
    labels_list = []
    # 步骤3.3：生成 BIO 标签
    # 变量名：word, flag 作用：word 为词，flag 为词性，生成 B（地名）、E（机构后缀）或 O 标签
    for word, flag in words_flags:
        words.append(word)
        if word in org_tag:
            labels_list.append('E')
        elif flag == 'ns':
            labels_list.append('B')
        else:
            labels_list.append('O')

    print(words, labels_list)
    # 步骤3.4：将lables_list标签列表拼接为字符串
    # 变量名：labels  作用：将 lables_list 拼接为 BIO 标签序列（如 "BBOOE"）
    labels = "".join(labels_list)
    # 步骤3.5：使用正则表达式匹配组织机构
    # 变量名：pattern, match_label 作用：pattern 匹配 B+O*E+ 模式，提取地名开头、机构后缀结尾的序列
    pattern = re.compile("B+O*E+")
    match_label = re.finditer(pattern, labels)
    # 步骤3.6：提取匹配的组织机构名称
    # 变量名：match_list=[] 作用：根据匹配的索引范围，从 words 拼接组织机构名称
    match_list = []
    print("words_flags：", len(words_flags))
    for match in match_label:
        print('match:', match)
        match_list.append(''.join(words[int(match.start()):int(match.end())]))

    # 步骤3.7：返回提取结果
    # 输出：match_list（组织机构名称列表，如 ['中国国家市场监督管理总局']）
    print(match_list)
    return match_list


# 步骤4：测试函数
# 变量名：text
# 作用：输入测试文本，调用 extract_org，打印提取的组织机构
text = "可在接到本决定书之日起六十日内向中国国家市场监督管理总局申请行政复议,杭州海康威视数字技术股份有限公司."
print(extract_org(text))  # 输出：['中国国家市场监督管理总局', '杭州海康威视数字技术股份有限公司']
