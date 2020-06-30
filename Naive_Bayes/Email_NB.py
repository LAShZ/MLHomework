from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from bs4 import BeautifulSoup

import re
import os
import sys
import stat
import email
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_file(filename):
    '''
    读出邮件内容，需要使用 email 解析
    :param filename: 邮件文件路径
    :return: 返回邮件主题和邮件内容字符串, 可能带 html 格式
    '''
    with open(filename, encoding='latin-1') as fp:
        msg = email.message_from_file(fp)
        payload = msg.get_payload()
        if type(payload) == type(list()):
            payload = payload[0]
        if type(payload) != type(''):
            payload = str(payload)

        sub = msg.get('subject')
        sub = str(sub)
        return sub + payload


def clean_html(raw_html):
    '''
    清除邮件内容中的 html 标签
    :param raw_html: 带 html 标签的文本内容
    :return: 不带 html 标签的文本内容
    '''
    return BeautifulSoup(raw_html, 'html.parser').text


def label_from_file(filename):
    '''
    从文件名中读取需要，如 'TRAIN_1234.eml' 的文件序号为 1234
    :param filename: 文件名
    :return: 文件序号
    '''
    for s in re.findall(r'\d+', filename):
        return int(s)
    raise ValueError('filename error : ' + filename)


def calc_tf_idf(tf, idf, text, ignore=3):
    '''
    计算一份邮件内容的词频和逆文档频率（仅计数）
    :param tf: 词频计数
    :param idf: 逆文档频率计数
    :return: 文档的单词数
    '''
    words = re.findall('\w+', text)
    count = 0
    word_set = set()
    for word in words:
        # 过滤无效的单词
        if len(word) < ignore or len(word) > 20:
            continue
        word = word.lower()

        # 统计逆文档频率, 一篇文章只加一次
        if not (word in word_set):
            idf[word] = idf.get(word, 0) + 1
            word_set.add(word)

        # 统计词频
        tf[word] = tf.get(word, 0) + 1

        # 计算一篇文档的单词总数
        count = count + 1

    return count


def get_label(labels, index):
    '''
    获取邮件的标签
    :param labels: 全部标签数据(Id 和 Prediction 两列)
    :return: 1 表示正常邮件，0 表示垃圾邮件
    '''
    return labels.Prediction[labels.Id == index].iloc[0]


def train_data():
    '''
    读取训练目录下到所有邮件和邮件分类标签
    :return: 所有词频和逆文档频率和邮件数量信息
    '''
    pathname = 'emails/TR'
    labels = pd.read_csv('emails/spam-mail.tr.label')

    ham_tf = dict()
    spam_tf = dict()
    word_idf = dict()
    ham_word_count = 0
    spam_word_count = 0
    file_count = 0
    spam_file_count = 0
    ham_file_count = 0

    # 遍历所有邮件文件
    for file in os.listdir(pathname):
        fpath = os.path.join(pathname, file)
        info = os.stat(fpath)
        if stat.S_ISREG(info.st_mode) and file.endswith('.eml'):
            '''
            1. 从邮件文件出读出所有文本
            2. 根据邮件标签，分别计算垃圾邮件的词频和逆文档频率
            '''
            text = clean_html(read_file(fpath))
            index = label_from_file(file)
            file_count = file_count + 1
            if get_label(labels, index) == 1:
                ham_file_count = ham_file_count + 1
                ham_word_count = ham_word_count + \
                    calc_tf_idf(ham_tf, word_idf, text)
            else:
                spam_file_count = spam_file_count + 1
                spam_word_count = spam_word_count + \
                    calc_tf_idf(spam_tf, word_idf, text)

    info = {}
    info['ham_word_count'] = ham_word_count
    info['spam_word_count'] = spam_word_count
    info['file_count'] = file_count
    info['ham_file_count'] = ham_file_count
    info['spam_file_count'] = spam_file_count
    print('train email info : ', info)

    word_df = pd.DataFrame([ham_tf, spam_tf, word_idf]).T
    word_df.columns = ['ham_tf', 'spam_tf', 'word_idf']
    return (word_df, info)


'''
读取所有训练集中的邮件，范围正常邮件和垃圾邮件对应每个词出现的次数以及训练集邮件的计数信息
'''
email_df, email_info = train_data()
# 拷贝数据，可重复运行这段代码
word_df = email_df.copy()
word_df.fillna(1, inplace=True)

# P(Y=S) : 垃圾邮件的概率
p_y_s = email_info['spam_file_count'] / email_info['file_count']

# P(Y=H) : 正常邮件的概率
p_y_h = 1 - p_y_s

# P(W|Y=H) : 正常邮件时，出现单词 W 的概率
word_df['ham_tf'] = word_df['ham_tf'] / email_info['ham_word_count']

# P(W|Y=S) : 垃圾邮件时，出现单词 W 的概率
word_df['spam_tf'] = word_df['spam_tf'] / email_info['spam_word_count']

# 根据公式计算 P(Y=S|W)
word_df['spam_sp'] = (word_df['spam_tf'] * p_y_s) / \
    (word_df['ham_tf'] * p_y_h + word_df['spam_tf'] * p_y_s)

# 选择 P(Y=S|W) >= 0.9 的单词作为识别关键词，节省计算
word_df = word_df.loc[(word_df['spam_sp'] >= 0.9)]

# 从大到小排序
word_df = word_df.sort_values(by=['spam_sp'], ascending=[False])

print(word_df)


def is_spam_email(filename, word_df, info, ignore=3):
    text = clean_html(read_file(filename))
    words = re.findall('[A-Za-z]+', text)
    word_set = set()
    p_s_w = info['spam_file_count'] / info['file_count']
    p_h_w = 1 - p_s_w

    for word in words:
        # 过滤无效的单词
        if len(word) < ignore or len(word) > 20:
            continue

        word = word.lower()

        # 属于垃圾邮件关键词 且 未参与计算过, 分子分母都乘以1000，防止小数点过小导致计算结果为0
        if (word in word_df.index) and not (word in word_set):
            word_set.add(word)
            p_s_w = 1000 * p_s_w * (word_df.loc[word].spam_tf)
            p_h_w = 1000 * p_h_w * (word_df.loc[word].ham_tf)

    # 没有垃圾邮件关键词则认为是正常邮件
    if len(word_set) == 0:
        return (False, 0)

    result = p_s_w / (p_s_w + p_h_w)
    if result > 0.9:
        return (True, result)
    return (False, result)


def test_data():
    pathname = 'emails/TT'
    spam_count = 0
    ham_count = 0
    Id = []
    Prediction = []

    # 遍历所有邮件文件
    for file in os.listdir(pathname):
        fpath = os.path.join(pathname, file)
        info = os.stat(fpath)
        if stat.S_ISREG(info.st_mode) and file.endswith('.eml'):
            spam, p = is_spam_email(fpath, word_df, email_info)
            value = 0 if spam else 1
            index = label_from_file(fpath)
            Id.append(index)
            Prediction.append(value)
            if spam:
                spam_count = spam_count + 1
            else:
                ham_count = ham_count + 1

    print('emal count ham %d spam %d' % (ham_count, spam_count))
    return (Id, Prediction)


Id, Prediction = test_data()
'''
根据返回的 Id 和 预测结果，生成 DataFrame 并排序后写到 csv 文件中
'''
test_df = pd.DataFrame(data=np.array([Id, Prediction])).T
test_df.columns = ['Id', 'Prediction']
test_df = test_df.sort_values(by=['Id'], ascending=[True])
test_df.to_csv('emails/submission.csv', sep=',', index=False, encoding='utf-8')
