import pandas as pd
from pkg import mmnorm, split_train_test_dict
# 根据问题一 预选择出来的分子标识符
moleculars = ["MDEC-23","MLogP","LipoaffinityIndex","maxsOH","minsOH","nC","nT6Ring","n6Ring","minsssN","BCUTp-1h","C2SP2","AMR","SwHBa","maxsssN","MDEC-22","SP-5","SaaCH","CrippenLogP","maxHsOH","nHaaCH"]

# 使用预测模型，根据 ax1 + ... + ax20 = pIC 已知 x 和 pIC ，预测出若干个 X 求出 最高的pIC， 

# 根据传入的数据集来获取最好的生物活性值 : 
# MoleDescPd 从excel获取的初始DataFrame数据
# ERActiveData key: SMILES，value: IC50_nM和pIC50
def GetMEData(MoleDescPd:pd.DataFrame, ERActiveData:dict):
    # 从原始的数据中提取出预选的分子标识符数据，然后使用该数据进行字典化
    MoleDescData = MoleDescPd.loc[::,moleculars]
    # MoleDescData key: SMILES, value: 化学描述符组成的向量
    MoleDescData = MoleDescData.T.to_dict('list')
    # 构建 X 矩阵
    return MoleDescData, ERActiveData

def GetMEDataSplit(MoleDescPd:pd.DataFrame):
    # 从原始的数据中提取出预选的分子标识符数据，然后使用该数据进行字典化
    MoleDescData = MoleDescPd.loc[::,moleculars]
    trainD, validD = split_train_test_dict(MoleDescData)
    #划分训练集个验证集
    # 构建 X 矩阵
    return trainD, validD


    









    


