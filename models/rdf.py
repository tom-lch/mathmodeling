from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
# 随机森林
def rforest():
    tfc = RandomForestClassifier(max_depth=2, random_state=0)
    return"rforest",tfc



# 决策树
from sklearn.tree import DecisionTreeClassifier

def dtreec():
    dtc =  DecisionTreeClassifier()
    return "DTree", dtc
