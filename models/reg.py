from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn import svm
from sklearn import ensemble

# 回归
# 1.决策树回归
def decisionTreeRegressor():
    model = tree.DecisionTreeRegressor()
    return "decisionTreeRegressor", model
 
# 2.线性回归
def linearRegression():
    model = LinearRegression()
    return "linearRegression", model
 
# 3.SVM回归
def svr():
    model = svm.SVR()
    return "svr", model
 
# 4.kNN回归
def kNeighborsRegressor():
    model = neighbors.KNeighborsRegressor()
    return "kNeighborsRegressor", model
# 5.随机森林回归
def randomForestRegressor():
    model = ensemble.RandomForestRegressor(n_estimators=20)  # 使用20个决策树
    return "randomForestRegressor", model
 
# 6.Adaboost回归
def adaBoostRegressor():
    model = ensemble.AdaBoostRegressor(n_estimators=200)  # 这里使用200个决策树
    return "adaBoostRegressor", model
 
# 7.GBRT回归
def gradientBoostingRegressor():
    model = ensemble.GradientBoostingRegressor(n_estimators=200)  # 这里使用200个决策树
    return "gradientBoostingRegressor", model
