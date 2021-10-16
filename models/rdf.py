from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# 随机森林
def rforest():
    tfc = RandomForestClassifier(max_depth=2, random_state=0)
    return"rforest",tfc

def rforest_single():
    tfc = RandomForestClassifier(max_depth=2, random_state=0)
    return"rforest_single",tfc

# 决策树
def dtreec():
    dtc =  DecisionTreeClassifier()
    return "DTree", dtc

def dtreec_single():
    dtc =  DecisionTreeClassifier()
    return "DTree_single", dtc

def nbbnb():
    bnb = OneVsRestClassifier(BernoulliNB())
    return "BernoulliNB", bnb

def nbbnb_single():
    bnb = BernoulliNB()
    return "BernoulliNB_single", bnb




def nbgnb():
    gnb = OneVsRestClassifier(BernoulliNB())
    return "GernoulliNB", gnb

def nbgnb_single():
    gnb = BernoulliNB()
    return "GernoulliNB_single", gnb



def nknc():
    knc = OneVsRestClassifier(KNeighborsClassifier())
    return "KNeighbors", knc

def nknc_single():
    knc = KNeighborsClassifier()
    return "KNeighbors_single", knc



def nbrnc():
    rnc =  OneVsRestClassifier(RadiusNeighborsClassifier())
    return "RadiusNeighbors", rnc

def nbrnc_single():
    rnc =  RadiusNeighborsClassifier()
    return "RadiusNeighbors_single", rnc



