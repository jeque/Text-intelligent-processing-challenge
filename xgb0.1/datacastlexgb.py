print("开始............")
import time
import pandas as pd
import xgboost as xgb
import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn import svm

start_time = time.time()

print("[1] Loading Data...")
df_train = pd.read_csv('./train_set.csv')
df_test = pd.read_csv('./test_set.csv')
df_train.drop(columns=['article', 'id'], inplace=True)
df_test.drop(columns=['article'], inplace=True)

print("特征分析...")
vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=6, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
tfidftransformer = TfidfTransformer()
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(df_train['word_seg']))

x_train = tfidf.transform(df_train['word_seg'])
#weight = tfidf.toarray()
x_test = tfidf.transform(df_test['word_seg'])
#test_weight = test_tfidf.toarray()
y_train = (df_train['class']-1).astype(int)

print("[2] xgboost分析...")
param = {'max_depth': 6, 'eta': 0.5, 'eval_metric': 'merror', 'silent': 1, 'objective': 'multi:softmax', 'num_class': 11}
#evallist = [(x_train, 'train')]
num_round = 100
cxg = xgb.train(param, x_train, num_round)
#cxg.fit(x_train, y_train)
preds = cxg.predict(x_test)

#lg = DecisionTreeClassifier(splitter = 'best', max_depth = None, presort = False)
#lg.fit(x_train, y_train)
#y_test = lg.predict(x_test)

print("[3] save............")
df_test['class'] = preds.tolist()
df_test['class'] = df_test['class'] +1
df_result = df_test.loc[:, ['id', 'class']]
df_result.to_csv('./result_svm.csv', index=False)

print("完成............")

end_time = time.time()
duration = end_time - start_time
print(duration)