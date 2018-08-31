简介
=========================
['达观杯'文本智能处理挑战赛官网](http://www.dcjingsai.com/common/cmpt/%E2%80%9C%E8%BE%BE%E8%A7%82%E6%9D%AF%E2%80%9D%E6%96%87%E6%9C%AC%E6%99%BA%E8%83%BD%E5%A4%84%E7%90%86%E6%8C%91%E6%88%98%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)<br>
该库用于达观杯比赛任务的代码实现研究，主要利用机器学习sklearn包实现，运用了特征工程和分类器。 特征工程部分主要针对文本分类任务的 hash/lsa/lda/doc2vec特征提取/特征选择/特征组合/特征构造进行了实现，而分类器部分主要有逻辑回归/SVM/随机森林/Bagging/Adaboost/GBDT/Xgboost/LightGBM等。该库需要经过调参以达到更优。<br>
# 1 特征工程 (位于features文件夹）
- 生成tf特征<br>
  * 运行`tf.py`生成tf特征；<br>
- 生成lda特征<br>
  * 运行`lda.py`将tf特征降维为lda特征；<br>
________________________________
- 生成tfidf特征<br>
  * 存放于features文件夹，运行里面的`tfidf.py`生成tfidf特征；<br>
  >差异代码如下：<br>
     ```Python
     vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
     ```
  * 运行features文件夹中的`tfidfpro.py`生成新的tfidf特征；（这两种选择其中一种作为下一步的特征基础）<br>
  >差异代码如下：<br>
     ```Python
     vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=6, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
     ```
- 生成lsa特征<br>
  * 运行`lsa.py`将tfidf特征降维为lsa特征；<br>
________________________________
- 构造特征<br>
  * 运行`feature_construct.py`,根据已有的特征，使用多项式方法构造出更多特征；<br>


# 2 分类器算法
- 使用逻辑回归分类器进行训练<br>

- 使用lgb进行训练<br>


