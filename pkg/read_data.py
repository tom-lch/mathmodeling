import pandas as pd
import numpy as np

def get_data(file: str, sheet: str, index_col='SMILES') -> pd.DataFrame:
    # 将所有训练数据读入 mddata 变量
    mddata = pd.read_excel(file, [sheet], index_col=index_col)
    # 去掉首行标题栏
    # 返回数据
    return mddata[sheet]


# 将数据按照8：2 划分成
# 返回 lebel 对应的729个参数（列向量）{smiles: [nAcid	ALogP	ALogp2	AMR	apol	naAromAtom	nAromBond ...]}
def split_train_test_dict(mddata:pd.DataFrame):
    np.random.seed(10)
    # 随机打乱
    sampler      = np.random.permutation(len(mddata))
    sampler_data = mddata.take(sampler)

    # 随机打乱 8:2 划分训练集和测试集
    datalen    = len(mddata)
    train_len  = int(datalen * 0.8)
    train_data = sampler_data.iloc[:train_len,:]
    test_data  = sampler_data.iloc[train_len:,:]

    # 转置
    ntrain_data = train_data.T
    ntest_data  = test_data.T
    # 转成字典方便后序训练
    train_data_dict =  ntrain_data.to_dict('list')
    test_data_dict  =  ntest_data.to_dict('list')

    return train_data_dict, test_data_dict

def get_data_to_dict(file: str, sheet: str) -> pd.DataFrame:
    # 将所有训练数据读入 mddata 变量
    mddata = get_data(file, sheet)
    return mddata.T.to_dict('list')
    


# 使用最大最小值进行归一化 
def mmnorm(array: np.array):
    maxcols=array.max(axis=0)
    mincols=array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in xrange(data_cols):
        t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
    return t

    

def ValidScore(arr:slice):
    res = [1, 0, 0, 1, 0]
    if len(arr) != 5 :
        print("出错了！")
        return -1
    score = 0
    for i in range(len(arr)) :
        if arr[i] == res[i] :
            score+=1
    return score

def GetBestSMILES_pICAndADMET(ERADict: dict, ADMET:dict)->str:
    ERADict_order=sorted(ERADict.items(),key=lambda x:x[1][1], reverse=True)
    for value in ERADict_order:
        smiles = value[0]
        admet = ADMET[smiles]
        s = ValidScore(admet)
        print(smiles, s)
        if s  >= 3 :
            break
    return value, admet