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


from sklearn.naive_bayes import BernoulliNB

def nbbnb():
    bnb = OneVsRestClassifier(BernoulliNB())
    return "BernoulliNB", bnb


from sklearn.naive_bayes import GaussianNB

def nbgnb():
    gnb = OneVsRestClassifier(BernoulliNB())
    return "GernoulliNB", gnb


from sklearn.neighbors import KNeighborsClassifier

def nknc():
    knc = OneVsRestClassifier(KNeighborsClassifier())
    return "KNeighbors", knc


from sklearn.neighbors import RadiusNeighborsClassifier

def nbrnc():
    rnc =  OneVsRestClassifier(RadiusNeighborsClassifier())
    return "RadiusNeighbors", rnc
