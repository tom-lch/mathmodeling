from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
def nnmlpc():
    model = OneVsRestClassifier(MLPClassifier())
    return "MLP", model

def nnmlpc_single():
    model = MLPClassifier()
    return "MLP_single", model



# 使用神经网络拟合多项式，来求解
def nnmplr():
    model = MLPRegressor(
        random_state=1, 
        max_iter=500, 
        hidden_layer_sizes=(500, ), 
        activation='relu', 
        solver='lbfgs', alpha=0.0001, 
        batch_size='auto', 
        learning_rate='constant')
    return "regr", model

