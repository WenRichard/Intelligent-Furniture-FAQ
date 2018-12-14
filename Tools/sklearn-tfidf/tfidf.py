# -*- coding: utf-8 -*-
# @Time    : 2018/12/14 14:55
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : data4_process.py
# @Software: PyCharm

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer

corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?',
 ]

# 1.统计词频
vectorizer = CountVectorizer()   # 初始化字典
X = vectorizer.fit_transform(corpus)
count = X.toarray()
print(vectorizer.get_feature_names())
print(count)

# 2.将词频矩阵转化为TFIDF
transformer = TfidfTransformer()
tfidf_matrix = transformer.fit_transform(count)
print(tfidf_matrix.toarray())  # 生成每句话的tfidf


