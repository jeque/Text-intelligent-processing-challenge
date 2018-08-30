print("开始............")
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
start_time = time.time()

print("[1] Loading Data...")
df_train = pd.read_csv('./train_set.csv')
df_test = pd.read_csv('./test_set.csv')
df_train.drop(columns=['article', 'id'], inplace=True)
df_test.drop(columns=['article'], inplace=True)

print("特征分析...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=100000)
vectorizer.fit(df_train['word_seg'])
x_train = vectorizer.transform(df_train['word_seg'])
x_test = vectorizer.transform(df_test['word_seg'])
y_train = df_train['class']-1

print("[2] 逻辑回归分析...")
lg = LogisticRegression(penalty='l2', C=1.0, multi_class='ovr', dual=True)
lg.fit(x_train, y_train)

y_test = lg.predict(x_test)

print("[3] save............")
df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] +1
df_result = df_test.loc[:, ['id', 'class']]
df_result.to_csv('./result.csv', index=False)

print("完成............")

print('Score: %.2f' % lg.score(x_test, y_test))
end_time = time.time()
duration = end_time - start_time
print(duration)