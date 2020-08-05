'''
@Author: xiaoyao jiang
@Date: 2020-04-08 15:35:24
@LastEditTime: 2020-07-06 20:52:29
@LastEditors: xiaoyao jiang
@Description: tools
@FilePath: /bookClassification(ToDo)/src/utils/tools.py
'''

import logging
import re
import time
from datetime import timedelta
from logging import handlers

import jieba
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from bayes_opt import BayesianOptimization
from sklearn import metrics
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from skopt import BayesSearchCV
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from src.utils import config

tqdm.pandas()


def get_score(Train_label, Test_label, Train_predict_label,
              Test_predict_label):
    '''
    @description: get model score
    @param {type}
    Train_label, ground truth label of train data set
    Test_label, ground truth label of test dataset
    @return:acc, f1_score
    '''
    return metrics.accuracy_score(
        Train_label, Train_predict_label), metrics.accuracy_score(
            Test_label, Test_predict_label), metrics.recall_score(
                Test_label, Test_predict_label,
                average='micro'), metrics.f1_score(Test_label,
                                                   Test_predict_label,
                                                   average='weighted')


def query_cut(query):
    '''
    @description: word segment
    @param {type} query: input data
    @return:
    list of cut word
    '''
    return list(jieba.cut(query))


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def create_logger(log_path):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    logger = logging.getLogger(log_path)
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)  # 设置日志格式
    logger.setLevel(level_relations.get('info'))  # 设置日志级别
    sh = logging.StreamHandler()  # 往屏幕上输出
    sh.setFormatter(format_str)  # 设置屏幕上显示的格式
    th = handlers.TimedRotatingFileHandler(
        filename=log_path, when='D', backupCount=3,
        encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
    th.setFormatter(format_str)  # 设置文件里写入的格式
    logger.addHandler(sh)  # 把对象加到logger里
    logger.addHandler(th)

    return logger


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\s+", "", string)
    # string = re.sub(r"[^\u4e00-\u9fff]", "", string)
    string = re.sub(r"[^\u4e00-\u9fa5^.^,^!^?^:^;^、^a-z^A-Z^0-9]", "", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    return string.strip()


def strQ2B(ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281
                  and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return "".join(ss)


def wam(sentence, w2v_model, method='mean', aggregate=True):
    '''
    @description: 通过word average model 生成句向量
    @param {type}
    sentence: 以空格分割的句子
    w2v_model: word2vec模型
    @return:
    '''
    arr = np.array([
        w2v_model.wv.get_vector(s) for s in sentence
        if s in w2v_model.wv.vocab.keys()
    ])
    if not aggregate:
        return arr
    if len(arr) > 0:
        # 第一种方法对一条样本中的词求平均
        if method == 'mean':
            return np.mean(np.array(arr), axis=0)
        # 第二种方法返回一条样本中的最大值
        elif method == 'max':
            return np.max(np.array(arr), axis=0)
        else:
            raise NotImplementedError
    else:
        return np.zeros(300)


def Grid_Train_model(model, Train_features, Test_features, Train_label,
                     Test_label):
    # 构建训练模型并训练及预测
    # 网格搜索
    parameters = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [1000, 2000],
        'subsample': [0.6, 0.75, 0.9],
        'colsample_bytree': [0.6, 0.75, 0.9],
        'reg_alpha': [5, 10],
        'reg_lambda': [10, 30, 50]
    }
    # 有了gridsearch我们便不需要fit函数
    gsearch = GridSearchCV(model,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=3,
                           verbose=True)
    gsearch.fit(Train_features, Train_label)
    # 输出最好的参数
    print("Best parameters set found on development set:{}".format(
        gsearch.best_params_))
    return gsearch


def bayes_parameter_opt_lgb(trn_data,
                            init_round=15,
                            opt_round=25,
                            n_folds=5,
                            random_seed=6,
                            n_estimators=10000,
                            learning_rate=0.05,
                            output_process=False):
    # parameters
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth,
                 lambda_l1, lambda_l2, min_split_gain, min_child_weight):
        params = {
            'application': 'multiclass',
            'num_iterations': n_estimators,
            'learning_rate': learning_rate,
            'early_stopping_round': 100,
            'num_class': len([x.strip() for x in open(config.root_path + '/data/class.txt').readlines()]),
            'metric': 'multi_logloss'
        }
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        cv_result = lgb.cv(params,
                           trn_data,
                           nfold=n_folds,
                           seed=random_seed,
                           stratified=True,
                           verbose_eval=200)
        return max(cv_result['multi_logloss-mean'])
        # range

    lgbBO = BayesianOptimization(lgb_eval, {
        'num_leaves': (24, 31),
        'feature_fraction': (0.5, 0.9),
        'bagging_fraction': (0.6, 1),
        'max_depth': (2, 5),
        'lambda_l1': (1, 30),
        'lambda_l2': (1, 50),
    },
                                 random_state=0)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    # output optimization process
    if output_process:
        lgbBO.points_to_csv("bayes_opt_result.csv")
    # return best parameters
    print(lgbBO.max)
    return lgbBO.max


bayes_cv_tuner = BayesSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt',
                                 application='multiclass',
                                 n_jobs=-1,
                                 verbose=1),
    search_spaces={
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'num_leaves': (2, 500),
        'max_depth': (0, 500),
        'min_child_samples': (0, 200),
        'max_bin': (100, 100000),
        'subsample': (0.01, 1.0, 'uniform'),
        'subsample_freq': (0, 10),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'min_child_weight': (0, 10),
        'subsample_for_bin': (100000, 500000),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'scale_pos_weight': (1e-6, 500, 'log-uniform'),
        'n_estimators': (10, 10000),
    },
    scoring='f1_macro',
    cv=StratifiedKFold(n_splits=2),
    n_iter=30,
    verbose=1,
    refit=True)


def rfecv_opt(model, n_jobs, X, y, cv=StratifiedKFold(2)):
    rfecv = RFECV(estimator=model,
                  step=1,
                  cv=cv,
                  n_jobs=n_jobs,
                  scoring='f1_macro',
                  verbose=1)
    rfecv.fit(X.values, y.values.ravel())
    print('Optimal number of features : %d', rfecv.n_features_)
    print('Max score with current model :', round(np.max(rfecv.grid_scores_),
                                                  3))
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel('Number of features selected')
    plt.ylabel('Cross validation score (f1_macro)')
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    important_columns = []
    n = 0
    for i in rfecv.support_:
        if i:
            important_columns.append(X.columns[n])
        n += 1
    return important_columns, np.max(rfecv.grid_scores_), rfecv


def routine(X, y, n_iter_max, n_jobs):
    list_models = []
    list_scores_max = []
    list_features = []
    for i in range(n_iter_max):
        print('Currently on iteration', i + 1, 'of', n_iter_max, '.')
        if i == 0:
            model = lgb.LGBMClassifier(max_depth=-1,
                                       learning_rate=0.1,
                                       objective='multiclass',
                                       silent=False,
                                       metric='None',
                                       num_class=len([x.strip() for x in open(config.root_path + '/data/class.txt').readlines()]),
                                       n_jobs=n_jobs,
                                       n_estimators=8000,
                                       class_weight='unbalanced')
        else:
            print('Adjusting model.')
            X_provi = X[imp_columns]
            # Get current parameters and the best parameters
            result = bayes_cv_tuner.fit(X_provi.values, y.values.ravel())
            best_params = pd.Series(result.best_params_)
            param_dict = pd.Series.to_dict(best_params)
            model = lgb.LGBMClassifier(
                colsample_bytree=param_dict['colsample_bytree'],
                learning_rate=param_dict['learning_rate'],
                max_bin=int(param_dict['max_bin']),
                max_depth=int(param_dict['max_depth']),
                min_child_samples=int(param_dict['min_child_samples']),
                min_child_weight=param_dict['min_child_weight'],
                n_estimators=int(param_dict['n_estimators']),
                num_leaves=int(param_dict['num_leaves']),
                reg_alpha=param_dict['reg_alpha'],
                reg_lambda=param_dict['reg_lambda'],
                scale_pos_weight=param_dict['scale_pos_weight'],
                subsample=param_dict['subsample'],
                subsample_for_bin=int(param_dict['subsample_for_bin']),
                subsample_freq=int(param_dict['subsample_freq']),
                n_jobs=n_jobs,
                class_weight='unbalanced',
                objective='multiclass')
        imp_columns, max_score, rfecv = rfecv_opt(model, n_jobs, X, y)
        list_models.append(model)
        list_scores_max.append(max_score)
        list_features.append(imp_columns)
    return list_models, list_scores_max, list_features


def formate_data(train, test, train_tfidf, test_tfidf, train_ae, test_ae):
    Train = pd.concat([
        train[[
            'labelIndex', 'length', 'capitals', 'caps_vs_length',
            'num_exclamation_marks', 'num_question_marks', 'num_punctuation',
            'num_symbols', 'num_words', 'num_unique_words', 'words_vs_unique',
            'nouns', 'adjectives', 'verbs', 'nouns_vs_length',
            'adjectives_vs_length', 'verbs_vs_length', 'nouns_vs_words',
            'adjectives_vs_words', 'verbs_vs_words', 'count_words_title',
            'mean_word_len', 'punct_percent'
        ]], train_tfidf, train_ae
    ] + [
        pd.DataFrame(
            train[i].tolist(),
            columns=[i + str(x) for x in range(train[i].iloc[0].shape[0])])
        for i in [
            'w2v_label_mean', 'w2v_label_max', 'w2v_mean', 'w2v_max',
            'w2v_win_2_mean', 'w2v_win_3_mean', 'w2v_win_4_mean',
            'w2v_win_2_max', 'w2v_win_3_max', 'w2v_win_4_max', 'res_embedding',
            'resnext_embedding', 'wide_embedding', 'bert_embedding', 'lda'
        ]
    ],
                      axis=1).fillna(0.0)

    Test = pd.concat([
        test[[
            'labelIndex', 'length', 'capitals', 'caps_vs_length',
            'num_exclamation_marks', 'num_question_marks', 'num_punctuation',
            'num_symbols', 'num_words', 'num_unique_words', 'words_vs_unique',
            'nouns', 'adjectives', 'verbs', 'nouns_vs_length',
            'adjectives_vs_length', 'verbs_vs_length', 'nouns_vs_words',
            'adjectives_vs_words', 'verbs_vs_words', 'count_words_title',
            'mean_word_len', 'punct_percent'
        ]], test_tfidf, test_ae
    ] + [
        pd.DataFrame(
            test[i].tolist(),
            columns=[i + str(x) for x in range(test[i].iloc[0].shape[0])])
        for i in [
            'w2v_label_mean', 'w2v_label_max', 'w2v_mean', 'w2v_max',
            'w2v_win_2_mean', 'w2v_win_3_mean', 'w2v_win_4_mean',
            'w2v_win_2_max', 'w2v_win_3_max', 'w2v_win_4_max', 'res_embedding',
            'resnext_embedding', 'wide_embedding', 'bert_embedding', 'lda'
        ]
    ],
                     axis=1).fillna(0.0)
    return Train, Test


def format_data(data, max_features, maxlen, tokenizer=None, shuffle=False):
    '''
    @description: use for autoencoder
    @param {type}
    @return:
    '''
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)

    data['text'] = data['text'].apply(lambda x: str(x).lower())

    X = data['text']

    if not tokenizer:
        filters = "\"#$%&()*+./<=>@[\\]^_`{|}~\t\n"
        tokenizer = Tokenizer(num_words=max_features, filters=filters)
        tokenizer.fit_on_texts(list(X))

    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=maxlen)

    return X, tokenizer
