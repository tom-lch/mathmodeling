from pkg import get_data, split_train_test_dict, get_data_to_dict
from base import DataModel
import pandas as pd
def run():
    # 程序入口函数
    data = get_data('./Molecular_Descriptor.xlsx', 'training')
    train_data, test_data = split_train_test_dict(data)

    # 将结果入进来
    resdata = get_data_to_dict('./ADMET.xlsx', 'training')
    # 模型 
    models = ["SVM", "adaboost", "DTree", "rforest", "BernoulliNB", "GernoulliNB", "KNeighbors", "MLP"] 
    #models = ["adaboost"]
    for i in range(len(models)):
        print(models[i])
        model = DataModel(train_data, test_data, resdata, models[i])
        model.Run()
        f1 = model.F1Score()
        a = model.AccuracyScore()
        ap = model.AveragePrecisionSscore()
        print("平均F1:", f1, ", 正确率:", a , ", 平均精度:", ap)
        c = model.ClassificationReport()
        print(c)
        with open(models[i] + ".txt", 'a+') as f:
            f.writelines("f1: ", f1, "\t", "a:", a, "\t", "ap:", ap)
            f.writelines(c)

    # 读入真实测试数据
        TestData = get_data_to_dict('./Molecular_Descriptor.xlsx', 'test')
        result = model.Predict(TestData)
        # 将result 写入到 ADMET.xlsx test
        olddata = get_data('./ADMET.xlsx', 'test')
        for i in range(len(olddata)):
            index = olddata.index[i]
            olddata.iloc[i,:] = pd.Series(result[index])
        print(olddata)
        olddata.to_excel(models[i] + "_Molecular_Descriptor.xlsx", 'test')


if __name__ == '__main__':
    run()