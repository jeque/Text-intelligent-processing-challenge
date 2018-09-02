# -*- coding: utf-8 -*-
"""
@brief : 将原始数据数字化为tfidf特征，并将结果保存至本地
@author: jeque
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import time

t_start = time.time()

"""=====================================================================================================================
1 数据预处理
"""
df_train = pd.read_csv('../data/train_set.csv')
df_train.drop(columns='article', inplace=True)
df_test = pd.read_csv('../data/test_set.csv')
df_test.drop(columns='article', inplace=True)
f_all = pd.concat(objs=[df_train, df_test], axis=0, sort=True)
y_train = (df_train['class'] - 1).values

"""=====================================================================================================================
2 特征工程
"""
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
vectorizer.fit(df_train['word_seg'])
x_train = vectorizer.transform(df_train['word_seg'])
x_test = vectorizer.transform(df_test['word_seg'])

"""=====================================================================================================================
3 保存至本地
"""
data = (x_train, y_train, x_test)
fp = open('./data_tfidf.pkl', 'wb')
pickle.dump(data, fp)
fp.close()

t_end = time.time()
print("已将原始数据数字化为tfidf特征，共耗时：{}min".format((t_end-t_start)/60))
