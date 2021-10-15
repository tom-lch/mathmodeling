from pkg import get_data, split_train_test_dict, get_data_to_dict
from base import DataModel
import pandas as pd
def run():
    # 程序入口函数

    data = get_data('./Molecular_Descriptor.xlsx', 'training')
    train_data, test_data = split_train_test_dict(data)

    # 将结果入进来
    resdata = get_data_to_dict('./ADMET.xlsx', 'training')
    # 模型 ["SVM" "adaboost", "DTree", "rforest" "BernoulliNB", "GernoulliNB", "KNeighbors" "MLP", "RadiusNeighbors"] 

    model = DataModel(train_data, test_data, resdata, "adaboost")
    model.Run()
    print(model.F1Score())
    print(model.AccuracyScore())
    print(model.AveragePrecisionSscore())

    # 读入真实测试数据
    # TestData = get_data_to_dict('./Molecular_Descriptor.xlsx', 'test')
    # result = model.Predict(TestData)

    # # 将result 写入到 ADMET.xlsx test
    # olddata = get_data('./ADMET.xlsx', 'test')
    # for i in range(len(olddata)):
    #     index = olddata.index[i]
    #     olddata.iloc[i,:] = pd.Series(result[index])
    # print(olddata)
    # olddata.to_excel("RadiusNeighbors_Molecular_Descriptor.xlsx", 'test')


if __name__ == '__main__':
    run()