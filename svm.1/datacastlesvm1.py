print("开始............")
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

start_time = time.time()

print("[1] Loading Data...")
df_train = pd.read_csv('./train_set.csv')
df_test = pd.read_csv('./test_set.csv')
df_train.drop(columns=['article', 'id'], inplace=True)
df_test.drop(columns=['article'], inplace=True)

print("特征分析...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=6, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
vectorizer.fit(df_train['word_seg'])
x_train = vectorizer.transform(df_train['word_seg'])
x_test = vectorizer.transform(df_test['word_seg'])
y_train = (df_train['class']-1).astype(int)

print("[2] SVM分析...")
lin_clf = svm.LinearSVC(multi_class='ovr', fit_intercept=True, intercept_scaling=1)
lin_clf.fit(x_train, y_train)
preds = lin_clf.predict(x_test)

print("[3] save............")
df_test['class'] = preds.tolist()
df_test['class'] = df_test['class'] +1
df_result = df_test.loc[:, ['id', 'class']]
df_result.to_csv('./result_svm.csv', index=False)

print("完成............")

end_time = time.time()
duration = end_time - start_time
print(duration)