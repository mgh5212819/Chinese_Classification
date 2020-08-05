# 文件介绍与运行说明


任务是基于图书的相关描述和图书的封面图片，自动给一个图书做类目的分类。这种任务是一个中文文本分类任务，也可以看作一个多模态分类任务。



## 具体给出的实现方式有三种：

### 1. 特征工程（图片特征、Tfidf特征、LDA特征、窗口词向量、包括label交互词向量、bert预训练句向量、基本NLP特征）+ GBDT

### 2. 机器学习模型（包括RandomForestClassifier随机森林，LogisticRegression逻辑回归，MultinomialNB朴素贝叶斯，SVC支持向量机，LightGBM梯度提升决策树等等）

### 3. 深度学习模型（包括RNN、CNN、RCNN、RNN_ATT、Transformer、BERT、XLNet、Roberta等等）



## 运行说明：

### 第一步，运行 `src/word2vec/embedding.py` 去生成各种各样的词嵌入（包括word2v3c，fasttext，tfidf以及lda）

### 第二步，运行 `src/ML/main.py`参数选择 'feature_engineerning' 去进行特征工程 + GBDT

### 第三步，运行 `src/ML/main.py`参数不选择 'feature_engineerning'可以尝试各种机器学习模型

### 第四步，运行`src/DL/train.py`指定不同的model可以尝试不同的深度学习模型，例如`python3 train,py --model bert`  尝试bert模型。



## 代码结构介绍
`data`: 数据存放目录

`model` : 模型存放目录

`logs` : 日志存放目录

`src` : 核心代码部分

`app.py` : 代码部署部分

`src/data` : 数据处理部分

`src/data/dataset.py` : 主要用于深度学习的数据处理

`src/data/mlData.py` : 主要用于机器学习的数据处理

`src/DL/` : 包含各类深度学习模型， 运行主入口为`src/DL/train.py`

`src/ML/` : 包含各类机器学习模型， 运行主入口为`src/ML/main.py`

`src/utils/` : 包含配置文件，特征工程函数，以及通用函数

`src/word2vec/` : 包含各类embedding的训练，保存加载。运行主入口为`src/word2vec/embedding.py`

