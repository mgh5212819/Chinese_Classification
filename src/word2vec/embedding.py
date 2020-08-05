
import pandas as pd
from gensim import models
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from gensim.models import LdaMulticore
from gensim.models.ldamodel import LdaModel
import gensim

from __init__ import *
from src.utils.config import root_path
from src.utils.tools import create_logger, query_cut
from src.word2vec.autoencoder import AutoEncoder
logger = create_logger(root_path + '/logs/embedding.log')


class SingletonMetaclass(type):
    '''
    @description: singleton
    '''
    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super(SingletonMetaclass,
                                    self).__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance


class Embedding(metaclass=SingletonMetaclass):
    def __init__(self):
        '''
        @description: This is embedding class. Maybe call so many times. we need use singleton model.
        In this class, we can use tfidf, word2vec, fasttext, autoencoder word embedding
        @param {type} None
        @return: None
        '''
        self.stopWords = open(root_path + '/data/stopwords.txt').readlines()
        self.ae = AutoEncoder()

    def load_data(self):
        '''
        @description:Load all data, then do word segment
        @param {type} None
        @return:None
        '''
        logger.info('load data')
        self.data = pd.concat([
            pd.read_csv(root_path + '/data/train.csv', sep='\t'),
            pd.read_csv(root_path + '/data/dev.csv', sep='\t'),
            pd.read_csv(root_path + '/data/test.csv', sep='\t')
        ])
        self.data["text"] = self.data['title'] + self.data['desc']
        self.data["text"] = self.data["text"].apply(query_cut)
        self.data['text'] = self.data.text.apply(lambda x: " ".join(x))

    def trainer(self):
        '''
        @description: Train tfidf,  word2vec, fasttext and autoencoder
        @param {type} None
        @return: None
        '''
        
        logger.info('train tfidf')
        # 1. 训练tfidf模型
        count_vect = TfidfVectorizer(stop_words=self.stopWords,
                                     max_df=0.4,
                                     min_df=0.001,
                                     ngram_range=(1, 2))
        self.tfidf = count_vect.fit(self.data["text"])
        
        
        
        logger.info('train word2vec')
        
        logger.info('train word2vec')
        
        self.data['text'] = self.data.text.apply(lambda x: x.split(' '))
        
        # 1. 训练 word2vec
        self.w2v = models.Word2Vec(min_count=2,
                                window=3,
                                size=300,
                                sample=6e-5,
                                alpha=0.03,
                                min_alpha=0.0007,
                                negative=15,
                                workers=4,
                                iter=10,
                                max_vocab_size=50000) 
        
        self.w2v.build_vocab(self.data['text'])
        self.w2v.train(self.data['text'],
              total_examples=self.w2v.corpus_count,
              epochs=15, 
              report_delay=1)
        

        
        
        logger.info('train fast')
        # 训练fast的词向量
        # 1. 训练fasttest 模型
        self.fast = models.FastText(min_count=2,
                                window=3,
                                size=300,
                                sample=6e-5,
                                alpha=0.03,
                                min_alpha=0.0007,
                                negative=15,
                                workers=4,
                                iter=10,
                                max_vocab_size=50000) 
    
        self.fast.build_vocab(self.data['text'])
        self.fast.train(self.data['text'],
              total_examples=self.fast.corpus_count,
              epochs=15,
              report_delay=1)
               
        
        
        

        logger.info('train lda')
        # 1. 训练 LDA 模型
        # hint 使用gensim
        self.id2word = gensim.corpora.Dictionary(self.data.text)
        corpus = [self.id2word.doc2bow(text) for text in self.data.text]
        self.LDAmodel = LdaMulticore(corpus=corpus, id2word=self.id2word, num_topics=30)

        
  
        
        

    def saver(self):
        '''
        @description: save all model 
        @param {type} None
        @return: None
        '''


        logger.info('save tfidf model')
        joblib.dump(self.tfidf, root_path + '/model/embedding/tfidf')

        logger.info('save w2v model')
        self.w2v.wv.save_word2vec_format(root_path +
                                         '/model/embedding/w2v.bin',
                                         binary=False)

        logger.info('save fast model')
        self.fast.wv.save_word2vec_format(root_path +
                                          '/model/embedding/fast.bin',
                                          binary=False)

        logger.info('save lda model')
        self.LDAmodel.save(root_path + '/model/embedding/lda')

    def load(self):
        '''
        @description: Load all embedding model
        @param {type} None
        @return: None
        '''
        logger.info('load tfidf model')
        self.tfidf = joblib.load(root_path + '/model/embedding/tfidf')

        logger.info('load w2v model')
        self.w2v = models.KeyedVectors.load_word2vec_format(
            root_path + '/model/embedding/w2v.bin', binary=False)

        logger.info('load fast model')
        self.fast = models.KeyedVectors.load_word2vec_format(root_path + '/model/embedding/fast.bin', binary=False)

        logger.info('load lda model')
        self.lda = LdaModel.load(root_path + '/model/embedding/lda')




if __name__ == "__main__":
    em = Embedding()
    em.load_data()
    em.trainer()
    em.saver()
    #em.load()
