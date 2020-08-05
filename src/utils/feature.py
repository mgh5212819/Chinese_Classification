

import numpy as np
import copy
from __init__ import *
from src.utils.tools import wam, format_data
from src.utils import config
import pandas as pd
import joblib
import json
import string
import jieba.posseg as pseg
from PIL import Image
import torchvision.transforms as transforms
#pd.set_option('display.max_columns',10000)
#pd.set_option('display.width', 10000)
#pd.set_option('display.max_colwidth',10000)



def get_lda_features(lda_model, document):

    topic = lda_model.get_document_topics(document, minimum_probability=0)
    topic = np.array(topic)

    print(topic[:,1])

    return topic[:,1]


def get_pretrain_embedding(text, tokenizer, model):

    # 1. 返回bert embedding
    # hint  返回需要转换成cpu模式
    #text = '我是真的喜欢你，没道理。'
    print(tokenizer.tokenize(text))
    text_dict = tokenizer.encode_plus(text, add_special_tokens=True, max_length=400, ad_to_max_length=True, return_attention_mask=True, return_tensors='pt',)
    input_ids, attention_mask, token_type_ids =text_dict['input_ids'], text_dict['attention_mask'], text_dict['token_type_ids']
    _, res = model(input_ids.to(config.device), 
                   attention_mask=attention_mask.to(config.device),
                   token_type_ids=token_type_ids.to(config.device))
    print(res.size())
    
    return res.detach().cpu().numpy()[0]


def get_transforms():

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.46777044, 0.44531429, 0.40661017],
            std=[0.12221994, 0.12145835, 0.14380469],
        ),
    ])


def get_img_embedding(cover, model):

    transforms = get_transforms()

    # 1. 读取封面， 返回modal embedding
    # 返回需要转换成cpu模式
    if str(cover)[-3:] != 'jpg':
        print("warning")
        return np.zeros((1,1000))[0]
    print(cover)          
    image = Image.open(cover).convert("RGB")
    image = transforms(image).to(config.device)
    print (model(image.unsqueeze(0)).size())          
    return model(image.unsqueeze(0)).detach().cpu().numpy()[0]


def get_embedding_feature(mldata):

    mldata.train["queryCutRMStopWords"] = mldata.train[
        "queryCutRMStopWord"].apply(lambda x: " ".join(x))
    mldata.dev["queryCutRMStopWords"] = mldata.dev[
        "queryCutRMStopWord"].apply(lambda x: " ".join(x))
    train_tfidf = pd.DataFrame(
        mldata.em.tfidf.transform(
            mldata.train["queryCutRMStopWords"].tolist()).toarray())
    train_tfidf.columns = [
        'tfidf' + str(i) for i in range(train_tfidf.shape[1])
    ]
    test_tfidf = pd.DataFrame(
        mldata.em.tfidf.transform(
            mldata.dev["queryCutRMStopWords"].tolist()).toarray())
    test_tfidf.columns = [
        'tfidf' + str(i) for i in range(train_tfidf.shape[1])
    ]


    print("transform w2v")
    mldata.train['w2v'] = mldata.train[
        "queryCutRMStopWord"].apply(
            lambda x: wam(x, mldata.em.w2v, aggregate=False))
    mldata.dev['w2v'] = mldata.dev["queryCutRMStopWord"].apply(
        lambda x: wam(x, mldata.em.w2v, aggregate=False))
    

    train = copy.deepcopy(mldata.train)   ########################################################
    test = copy.deepcopy(mldata.dev)
    labelNameToIndex = json.load(open(config.root_path +
                                      '/data/label2id.json'))
    labelIndexToName = {v: k for k, v in labelNameToIndex.items()}
    w2v_label_embedding = np.array([
        mldata.em.w2v.wv.get_vector(labelIndexToName[key])
        for key in labelIndexToName
        if labelIndexToName[key] in mldata.em.w2v.wv.vocab.keys()
    ])

    joblib.dump(w2v_label_embedding,
                config.root_path + '/data/w2v_label_embedding.pkl')
    train = generate_feature(train, w2v_label_embedding, model_name='w2v')
    test = generate_feature(test, w2v_label_embedding, model_name='w2v')

    return train_tfidf, test_tfidf, train, test


ch2en = {
    '！': '!',
    '？': '?',
    '｡': '.',
    '（': '(',
    '）': ')',
    '，': ',',
    '：': ':',
    '；': ';',
    '｀': ','
}


def tag_part_of_speech(data):
    '''
    @description: tag part of speech, then calculate the num of noun, adj and verb
    @param {type}
    data, input data
    @return:
    noun_count,num of noun
    adjective_count, num of adj
    verb_count, num of verb
    '''

    # 1. 计算名词个数
    # 1. 计算形容词个数
    # 1. 计算动词个数
    words = [tuple(x) for x in list(pseg.cut(data))]

    noun_count = len([w for w in words if w[1] in ('n', 'nr', 'ns', 'nt', 'nz', 'Ng')])
    adjective_count = len([w for w in words if w[1] in ('ag', 'a', 'ad', 'an')])
    verb_count = len([w for w in words if w[1] in ('vg', 'v', 'vd', 'vn')])

    return noun_count, adjective_count, verb_count


def get_basic_feature(df):

    
    df['text'] = df['title'] + df['desc']
    # 分词
    df['queryCut'] = df['queryCut'].progress_apply(
        lambda x: [i if i not in ch2en.keys() else ch2en[i] for i in x])
    # 文本的长度
    df['length'] = df['queryCut'].progress_apply(lambda x: len(x))
    # 大写的个数
    df['capitals'] = df['queryCut'].progress_apply(
        lambda x: sum(1 for c in x if c.isupper()))
    # 大写 与 文本长度的占比
    df['caps_vs_length'] = df.progress_apply(
        lambda row: float(row['capitals']) / float(row['length']), axis=1)
    # 感叹号的个数
    df['num_exclamation_marks'] = df['queryCut'].progress_apply(
        lambda x: x.count('!'))
    # 问号个数
    df['num_question_marks'] = df['queryCut'].progress_apply(
        lambda x: x.count('?'))
    # 标点符号个数
    df['num_punctuation'] = df['queryCut'].progress_apply(
        lambda x: sum(x.count(w) for w in string.punctuation))
    # *&$%字符的个数
    df['num_symbols'] = df['queryCut'].progress_apply(
        lambda x: sum(x.count(w) for w in '*&$%'))
    # 词的个数
    df['num_words'] = df['queryCut'].progress_apply(lambda x: len(x))
    # 唯一词的个数
    df['num_unique_words'] = df['queryCut'].progress_apply(
        lambda x: len(set(w for w in x)))
    # 唯一词 与总词数的比例
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    # 获取名词， 形容词， 动词的个数， 使用tag_part_of_speech函数
    df['nouns'], df['adjectives'], df['verbs'] = zip(
        *df['text'].progress_apply(lambda x: tag_part_of_speech(x)))
    # 名词占总长度的比率
    df['nouns_vs_length'] = df['nouns'] / df['length']
    # 形容词占总长度的比率
    df['adjectives_vs_length'] = df['adjectives'] / df['length']
    # 动词占总长度的比率
    df['verbs_vs_length'] = df['verbs'] / df['length']
    # 名词占总词数的比率
    df['nouns_vs_words'] = df['nouns'] / df['num_words']
    # 形容词占总词数的比率
    df['adjectives_vs_words'] = df['adjectives'] / df['num_words']
    # 动词占总词数的比率
    df['verbs_vs_words'] = df['verbs'] / df['num_words']
    # 首字母大写其他小写的个数
    df["count_words_title"] = df["queryCut"].progress_apply(
        lambda x: len([w for w in x if w.istitle()]))
    # 平均词的个数
    df["mean_word_len"] = df["queryCut"].progress_apply(
        lambda x: np.mean([len(w) for w in x]))
    # 标点符号的占比
    df['punct_percent'] = df['num_punctuation'] * 100 / df['num_words']
    
    return df


def generate_feature(data, label_embedding, model_name='w2v'):
    '''
    @description: word2vec -> max/mean, word2vec n-gram(2, 3, 4) -> max/mean, label embedding->max/mean
    @param {type}
    data， input data, DataFrame
    label_embedding, all label embedding
    model_name, w2v means word2vec
    @return: data, DataFrame
    '''
    print('generate w2v & fast label max/mean')
    # 首先在预训练的词向量中获取标签的词向量句子,每一行表示一个标签表示
    # 每一行表示一个标签的embedding

    data[model_name + '_label_mean'] = data[model_name].progress_apply(
        lambda x: Find_Label_embedding(x, label_embedding, method='mean'))
    data[model_name + '_label_max'] = data[model_name].progress_apply(
        lambda x: Find_Label_embedding(x, label_embedding, method='max'))

    print('generate embedding max/mean')
    data[model_name + '_mean'] = data[model_name].progress_apply(
        lambda x: np.mean(np.array(x), axis=0))
    data[model_name + '_max'] = data[model_name].progress_apply(
        lambda x: np.max(np.array(x), axis=0))

    print('generate embedding window max/mean')
    data[model_name + '_win_2_mean'] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 2, method='mean'))
    data[model_name + '_win_3_mean'] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 3, method='mean'))
    data[model_name + '_win_4_mean'] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 4, method='mean'))
    data[model_name + '_win_2_max'] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 2, method='max'))
    data[model_name + '_win_3_max'] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 3, method='max'))
    data[model_name + '_win_4_max'] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 4, method='max'))
    
    return data


def Find_embedding_with_windows(embedding_matrix, window_size=2,
                                method='mean'):
    '''
    @description: generate embedding use window
    @param {type}
    embedding_matrix, input sentence's embedding
    window_size, 2, 3, 4
    method, max/ mean
    @return: ndarray of embedding
    '''
    # 最终的词向量
    result_list = []
    for k1 in range(len(embedding_matrix)):
        if int(k1 + window_size) > len(embedding_matrix):
            result_list.extend(embedding_matrix[k1:])
        else:
            result_list.extend(embedding_matrix[k1:k1 + window_size])
    if method == 'mean':
        return np.mean(result_list, axis=0)
    else:
        return np.max(result_list, axis=0)


def softmax(x):
    '''
    @description: calculate softmax
    @param {type}
    x, ndarray of embedding
    @return: softmax result
    '''
    return np.exp(x) / np.exp(x).sum(axis=0)


def Find_Label_embedding(example_matrix, label_embedding, method='mean'):
    '''
    @description: 根据论文《Joint embedding of words and labels》获取标签空间的词嵌入
    '''

    # 根据矩阵乘法来计算label与word之间的相似度
    similarity_matrix = np.dot(example_matrix, label_embedding.T) / (
        np.linalg.norm(example_matrix) * (np.linalg.norm(label_embedding)))

    # 然后对相似矩阵进行均值池化，则得到了“类别-词语”的注意力机制
    # 这里可以使用max-pooling和mean-pooling
    attention = similarity_matrix.max()
    attention = softmax(attention)
    # 将样本的词嵌入与注意力机制相乘得到
    attention_embedding = example_matrix * attention
    if method == 'mean':
        return np.mean(attention_embedding, axis=0)
    else:
        return np.max(attention_embedding, axis=0)
