from models import ms, regmodels
from sklearn import metrics
import numpy as np
import time
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



class DataModel:
    def __init__(self, Traindata: dict, Testdata:dict, res: dict, model: str):
        self.Models = {}
        self.modelname = model
        self.Traindata = Traindata
        self.Testdata = Testdata
        self.res = res
        self.TrainX = None
        self.TrainY = None
        self.TestX = None
        self.TestY = None
        self.PredY = None
        self.model = None
        self.modelSingles = {}
        self.__gettrainval__()
        self.__gettestval__()
        self.register_models()

    def __gettrainval__(self):
        x = []
        y = []
        for k, v in self.Traindata.items():
            x.append(v)
            y.append(self.res[k])

        self.TrainX = np.array(x)
        self.TrainY = np.array(y)

    def __gettestval__(self):
        x = []
        y = []
        for k, v in self.Testdata.items():
            x.append(v)
            y.append(self.res[k])
        self.TestX = np.array(x)
        self.TestY = np.array(y)

    def register_models(self):
        for model in ms:
            k, v = model()
            self.Models[k] = v
    
    def Run(self):
        print("X shape:", self.TrainX.shape)
        print("Y shape:", self.TrainY.shape)
        self.model = self.Models[self.modelname]
        self.model.fit(self.TrainX, self.TrainY)
        self.PredY = self.model.predict(self.TestX)

    def F1Score(self, average="macro"):
        # average: "macro" "micro" weighted" "samples"
        return metrics.f1_score(self.TestY, self.PredY, average=average)
    
    def AccuracyScore(self):
        return metrics.accuracy_score(self.TestY, self.PredY)
    
    def AveragePrecisionSscore(self):
        return metrics.average_precision_score(self.TestY, self.PredY)

    def ClassificationReport(self):
        return metrics.classification_report(self.TestY, self.PredY)

    def Predict(self, TestData:dict)->dict:
        res = {}
        for k, v in TestData.items():
            res[k] = self.model.predict(v)[0]
        return res
    
    def PrefictOne(self, val: np.array):
        res = self.model.predict(val)
        return res

    def SimpleRun(self, arr:slice):
        # 将self.TrainY 按照列进行拆分
        self.PredY = []
        for i in range(len(self.TrainY[0])):
            model = self.Models[self.modelname]
            model.fit(self.TrainX, self.TrainY[:,i])
            predY = model.predict(self.TestX)
            self.PredY.append(predY)
            self.modelSingles[i] = model
        print(len(self.PredY), len(self.PredY[0]))

    def ClassificationReportSingle(self):
        res = {}
        for i in range(len(self.TrainY[0])):
            print(self.TestY[:,i], self.PredY[i])
            c = metrics.classification_report(self.TestY[:,i], self.PredY[i])
            print(c)
            res[i] = c
        return res
    
    def PredictSingle(self, TestData:dict):
        #arr = np.zeros(len(TestData), len(self.TestY[0]))
        res = {}
        for i in range(len(self.TrainY[0])):
            model = self.modelSingles[i]
            for k, v in TestData.items():
                if k not in res.keys() :
                    res[k] = np.zeros(len(self.TestY[0]))
                r = model.predict([v])[0]
                print("********************")
                print("单个模型预测结果", r)
                print(res[k], i)
                res[k][i] = r
                print(res[k], i)
                print("$$$$$$$$$$$$$$$$$$$$")
        print(res)
        return res


class Q2Model:
    def __init__(self, MoleDescData: dict, res: dict, model: str):
        self.Models = {}
        self.modelname = model
        self.Traindata = MoleDescData
        self.res = res
        self.TrainX = None
        self.TrainY = None
        self.ValidX = None
        self.ValidY = None
        self.PredY = None
        self.model = None
        self.Scaler = None
        self.modelSingles = {}
        self.__getval__()
        self.register_models()
    

    def __getval__(self):
        X = []
        Y = []
        for k, v in self.Traindata.items():
            X.append(v)
            Y.append(self.res[k][1])
        
        # 此时X shape 1974, 20  Y 1974,
        self.TrainX = np.array(X)
        self.TrainY = np.array(Y)
        print(self.TrainX.shape, self.TrainY.shape)
        # self.TrainX = mmnorm(self.TrainX)
        # 数据标准化
        scaler = StandardScaler()
        scaler.fit(self.TrainX)
        self.TrainX = scaler.transform(self.TrainX)
        self.Scaler = scaler
        # 将数据划分成 训练集和验证集
        self.TrainX, self.ValidX, self.TrainY, self.ValidY = train_test_split(self.TrainX, self.TrainY, test_size=0.2, random_state=2)
        

    def register_models(self):
        for model in regmodels:
            k, v = model()
            self.Models[k] = v
    def Predict(self, val: np.array):
        val = self.Scaler.transform(val)
        return self.model.predict(val)

    def Run(self):
        print("X shape:", self.TrainX.shape)
        print("Y shape:", self.TrainY.shape)
        self.model = self.Models[self.modelname]
        self.model.fit(self.TrainX, self.TrainY)
        self.PredY = self.model.predict(self.ValidX)
        

    def R2Score(self):
        return r2_score(self.ValidY, self.PredY)
    
    def MeanSquaredError(self):
        return mean_squared_error(self.ValidY, self.PredY)

    def MeanAbsoluteError(self):
        return mean_absolute_error(self.ValidY, self.PredY)



  
 

        

# 定义一个节点用于存储回溯值

class HTree:
    def __init__(self, arr:slice, deep:int):
        self.Parames = arr[:]
        self.index = deep
        self.isKeep = False




