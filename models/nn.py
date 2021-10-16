from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier


def nnmlpc():
    mlpc = OneVsRestClassifier(MLPClassifier())
    return "MLP", mlpc

def nnmlpc_single():
    mlpc = MLPClassifier()
    return "MLP_single", mlpc