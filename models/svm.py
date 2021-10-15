from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# 支持向量机 多标签多分类

def svm():
    clf = OneVsRestClassifier(SVC())
    return "SVM", clf

