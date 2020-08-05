'''
@Description: There are two options. One is using pretrained embedding as feature to compare common ML models.
              The other is using feature engineering + param search tech + imbanlance to train a liaghtgbm model.
'''
import os
import torch
import lightgbm as lgb
import torchvision
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from transformers import BertModel, BertTokenizer

from src.data.mlData import MLData
from src.utils import config
from src.utils.config import root_path
from src.utils.tools import (Grid_Train_model, bayes_parameter_opt_lgb,
                             create_logger, formate_data, get_score)
from src.utils.feature import (get_embedding_feature, get_img_embedding,
                               get_lda_features, get_pretrain_embedding,
                               get_autoencoder_feature, get_basic_feature)

logger = create_logger(config.log_dir + 'model.log')


class Models(object):
    def __init__(self, feature_engineer=False):
        '''
        @description: initlize Class, EX: model
        @param {type} :
        feature_engineer: whether using feature engineering, if `False`, then compare common ML models
        res_model: res network model
        resnext_model: resnext network model
        wide_model: wide res network model
        bert: bert model
        ml_data: new mldata class
        @return: No return
        '''
        # 1. 使用torchvision 初始化resnet152模型
        # 2. 使用torchvision 初始化 resnext101_32x8d 模型
        # 3. 使用torchvision 初始化  wide_resnet101_2 模型
        # 4. 加载bert 模型
        print("load")
        self.res_model =torchvision.models.resnet152(pretrained=False)
        self.res_model.load_state_dict(torch.load(config.root_path + '/model/resnet150/resnet152-b121ed2d.pth'))
        self.res_model = self.res_model.to(config.device)
        self.resnext_model =torchvision.models.resnext101_32x8d(pretrained=True)
        self.resnext_model = self.resnext_model.to(config.device)
        self.wide_model = torchvision.models.wide_resnet101_2(pretrained=True)
        self.wide_model = self.wide_model.to(config.device)
        
        self.bert_tonkenizer = BertTokenizer.from_pretrained(config.root_path + '/model/bert')
        self.bert = BertModel.from_pretrained(config.root_path + '/model/bert')
        self.bert = self.bert.to(config.device)
        self.ml_data = MLData(debug_mode=True)
        if feature_engineer:
            self.model = lgb.LGBMClassifier(objective='multiclass',
                                            device = 'gpu',
                                            n_jobs=10,
                                            num_class=33,
                                            num_leaves=30,
                                            reg_alpha=10,
                                            reg_lambda=200,
                                            max_depth=3,
                                            learning_rate=0.05,
                                            n_estimators=2000,
                                            bagging_freq=1,
                                            bagging_fraction=0.9,
                                            feature_fraction=0.8,
                                            seed=1440)
        else:
            self.models = [
                RandomForestClassifier(n_estimators=500,
                                       max_depth=5,
                                       random_state=0),
                LogisticRegression(solver='liblinear', random_state=0),
                MultinomialNB(),
                SVC(),
                lgb.LGBMClassifier(objective='multiclass',
                                   n_jobs=10,
                                   num_class=33,
                                   num_leaves=30,
                                   reg_alpha=10,
                                   reg_lambda=200,
                                   max_depth=3,
                                   learning_rate=0.05,
                                   n_estimators=2000,
                                   bagging_freq=1,
                                   bagging_fraction=0.8,
                                   feature_fraction=0.8),
            ]

    def feature_engineer(self):
        '''
        @description: This function is building all kings of features
        @param {type} None
        @return:
        X_train, feature of train set
        X_test, feature of test set
        y_train, label of train set
        y_test, label of test set
        '''
        logger.info("generate embedding feature ")
        train_tfidf, test_tfidf, train, test = get_embedding_feature(
            self.ml_data)

        logger.info("generate basic feature ")

        
        # 1. 获取 基本的 NLP feature
        train =get_basic_feature(train)
        test =get_basic_feature(test)
        print(test.loc[0])

        logger.info("generate modal feature ")
        cover = os.listdir(config.root_path + '/data/book_cover/')
        train['cover'] = train.title.progress_apply(
            lambda x: config.root_path + '/data/book_cover/' + x + '.jpg'
            if x + '.jpg' in cover else '')
        test['cover'] = test.title.progress_apply(
            lambda x: config.root_path + '/data/book_cover/' + x + '.jpg'
            if x + '.jpg' in cover else '')


        # 1. 获取 三大CV模型的 modal embedding
        train['res_embedding'] = train['cover'].progress_apply(lambda x: get_img_embedding(x,self.res_model))
        test['res_embedding'] = test['cover'].progress_apply(lambda x: get_img_embedding(x,self.res_model))  
        print(len(test.loc[0,'res_embedding']))
        
        #train['resnext_embedding'] = test['cover'].progress_apply(lambda x: get_img_embedding(x,self.resnext_model))
        #test['resnext_embedding'] = test['cover'].progress_apply(lambda x: get_img_embedding(x,self.resnext_model))

        #train['wide_embedding'] = test['cover'].progress_apply(lambda x: get_img_embedding(x,self.wide_model))
        #test['wide_embedding'] = test['cover'].progress_apply(lambda x: get_img_embedding(x,self.wide_model))
 
        logger.info("generate bert feature ")

        # 1. 获取bert embedding
        train['bert_embedding'] = train['text'].progress_apply(lambda x: get_pretrain_embedding(x, self.bert_tonkenizer, self.bert))
        test['bert_embedding'] = test['text'].progress_apply(lambda x: get_pretrain_embedding(x, self.bert_tonkenizer, self.bert))
        
        print(test.loc[0])

        logger.info("generate lda feature ")


        # 1. 获取 lda feature

        
        train['bow'] = train['queryCutRMStopWord'].apply(lambda x: self.ml_data.em.lda.id2word.doc2bow(x))
        test['bow'] = test['queryCutRMStopWord'].apply(lambda x: self.ml_data.em.lda.id2word.doc2bow(x))
        print(test['queryCutRMStopWord'])
        print(test['bow'])
        # 在bag of word 基础上得到lda的embedding
        train['lda'] = list(map(lambda doc: get_lda_features(self.ml_data.em.lda, doc),train['bow']))
        test['lda'] = list(map(lambda doc: get_lda_features(self.ml_data.em.lda, doc),test['bow']))        
        print(test['lda'])
        print(test.loc[0])


        logger.info("formate data")
        print(test)
        print(test_tfidf)
        train, test = formate_data(train, test, train_tfidf, test_tfidf)
        print(test)
        print(test.loc[0])
        
        cols = [x for x in train.columns if str(x) not in ['labelIndex']]
        print(cols)
        X_train = train[cols]
        X_test = test[cols]
        print(X_test)
        train["labelIndex"] = train["labelIndex"].astype(int)
        test["labelIndex"] = test["labelIndex"].astype(int)
        y_train = train["labelIndex"]
        y_test = test["labelIndex"]
        print(y_test)
        return X_train, X_test, y_train, y_test

    def param_search(self, search_method='grid'):
        '''
        @description: use param search tech to find best param
        @param {type}
        search_method: two options. grid or bayesian optimization
        @return: None
        '''
        if search_method == 'grid':
            logger.info("use grid search")
            self.model = Grid_Train_model(self.model, self.X_train,
                                          self.X_test, self.y_train,
                                          self.y_test)
        elif search_method == 'bayesian':
            logger.info("use bayesian optimization")
            trn_data = lgb.Dataset(data=self.X_train,
                                   label=self.y_train,
                                   free_raw_data=False)
            param = bayes_parameter_opt_lgb(trn_data)
            logger.info("best param", param)
            return param


    def unbalance_helper(self,
                         imbalance_method='under_sampling',
                         search_method='grid'):
        '''
        @description: handle unbalance data, then search best param
        @param {type}
        imbalance_method,  three option, under_sampling for ClusterCentroids, SMOTE for over_sampling, ensemble for BalancedBaggingClassifier
        search_method: two options. grid or bayesian optimization
        @return: None
        '''
        logger.info("get all freature")
        self.X_train, self.X_test, self.y_train, self.y_test = self.feature_engineer(
        )
        model_name = None
        if imbalance_method == 'over_sampling':
            logger.info("Use SMOTE deal with unbalance data ")

            # 1. 使用over_sampling 处理样本不平衡问题
            print(self.y_train)
            self.X_train, self.y_train = SMOTE().fit_resample(
                self.X_train, self.y_train)
            print(self.y_train)
            self.X_test, self.y_test = SMOTE().fit_resample(
                self.X_train, self.y_train)
            model_name = 'lgb_over_sampling'
            
        elif imbalance_method == 'under_sampling':
            logger.info("Use ClusterCentroids deal with unbalance data ")

            # 1. 使用 under_sampling 处理样本不平衡问题
            print(self.X_train)
            #print(self.y_train)
            self.X_train, self.y_train = ClusterCentroids(
                random_state=0).fit_resample(self.X_train, self.y_train)
            print(self.X_train)
            #print(self.y_train)
            self.X_test, self.y_test = ClusterCentroids(
                random_state=0).fit_resample(self.X_test, self.y_test)
            model_name = 'lgb_under_sampling'
            
        elif imbalance_method == 'ensemble':
            self.model = BalancedBaggingClassifier(
                base_estimator=DecisionTreeClassifier(),
                sampling_strategy='auto',
                replacement=False,
                random_state=0)
            model_name = 'ensemble'
        logger.info('search best param')
        
        if imbalance_method != 'ensemble':
            param = self.param_search(search_method=search_method)
            param['params']['num_leaves'] = int(param['params']['num_leaves'])
            param['params']['max_depth'] = int(param['params']['max_depth'])
            self.model = self.model.set_params(**param['params'])        
        

        logger.info('fit model ')
        self.model.fit(self.X_train, self.y_train)
        
        

        # 1. 预测测试集的label
        # 2. 预测训练机的label
        # 3. 计算percision , accuracy, recall, fi_score

        Test_predict_label = self.model.predict(self.X_test)
        Train_predict_label = self.model.predict(self.X_train)
        per, acc, recall, f1 = get_score(self.y_train, self.y_test,
                                         Train_predict_label,
                                         Test_predict_label)
        
        # 输出训练集的准确率
        logger.info('Train accuracy %s' % per)
        # 输出测试集的准确率
        logger.info('test accuracy %s' % acc)
        # 输出recall
        logger.info('test recall %s' % recall)
        # 输出F1-score
        logger.info('test F1_score %s' % f1)
        self.save(model_name)

    def model_select(self,
                     X_train,
                     X_test,
                     y_train,
                     y_test,
                     feature_method='tf-idf'):
        '''
        @description: using different embedding feature to train common ML models
        @param {type}
        X_train, feature of train 
        X_test, feature of test set
        y_train, label of train set
        y_test, label of test set
        feature_method, three options , tfidf, word2vec and fasttext
        @return: None
        '''
        for model in self.models:
            model_name = model.__class__.__name__
            print(model_name)
            clf = model.fit(X_train, y_train)
            Test_predict_label = clf.predict(X_test)
            Train_predict_label = clf.predict(X_train)
            per, acc, recall, f1 = get_score(y_train, y_test,
                                             Train_predict_label,
                                             Test_predict_label)
            # 输出训练集的准确率
            logger.info(model_name + '_' + 'Train accuracy %s' % per)

            # 输出测试集的准确率
            logger.info(model_name + '_' + ' test accuracy %s' % acc)

            # 输出recall
            logger.info(model_name + '_' + 'test recall %s' % recall)

            # 输出F1-score
            logger.info(model_name + '_' + 'test F1_score %s' % f1)
            


    def predict(self, title, desc):

        inputs = self.process(title, desc)
        label = self.ix2label[self.model.predict(inputs)[0]]
        proba = np.max(self.model.predict_proba(inputs))
        return label, proba

    def save(self, model_name):

        joblib.dump(self.model, root_path + '/model/ml_model/' + model_name)

    def load(self, path):

        self.model = joblib.load(path)