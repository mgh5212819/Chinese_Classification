
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
# sys.path.append('/Users/leonjiang/Downloads/textClassification/')
# ys.path.append('/home/user10000281/notespace/textClassification/')