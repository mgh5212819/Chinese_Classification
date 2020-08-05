'''
@Author: your name
@Date: 2020-04-08 16:05:19
@LastEditTime: 2020-04-08 17:05:59
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /textClassification/src/data/__init__.py
'''
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
# sys.path.append('/Users/leonjiang/Downloads/textClassification/')
# ys.path.append('/home/user10000281/notespace/textClassification/')