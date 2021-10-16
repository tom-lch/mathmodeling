from pkg import get_data, split_train_test_dict, get_data_to_dict
from base import DataModel
import pandas as pd
import json
def run():
    # 程序入口函数
    data = get_data('./Molecular_Descriptor.xlsx', 'training')
    train_data, test_data = split_train_test_dict(data)

    # 将结果入进来
    resdata = get_data_to_dict('./ADMET.xlsx', 'training')
    # 模型 
    models = ["SVM", "adaboost", "DTree", "rforest", "BernoulliNB", "GernoulliNB", "KNeighbors", "MLP"] 
    for i in range(len(models)):
        model_name = models[i]
        model = DataModel(train_data, test_data, resdata, model_name)
        model.Run()
        f1 = model.F1Score()
        a = model.AccuracyScore()
        ap = model.AveragePrecisionSscore()
        print("平均F1:", f1, ", 正确率:", a , ", 平均精度:", ap)
        c = model.ClassificationReport()
        with open("./1016/" + models[i] + ".txt", 'w+') as f:
            f.write(f"f1: {f1},\t a: {a},\t ap:{ap} \n")
            f.write(f"{c}")

    # 读入真实测试数据
        TestData = get_data_to_dict('./Molecular_Descriptor.xlsx', 'test')
        result = model.Predict(TestData)
        # 将result 写入到 ADMET.xlsx test
        olddata = get_data('./ADMET.xlsx', 'test')
        for i in range(len(olddata)):
            index = olddata.index[i]
            olddata.iloc[i,:] = pd.Series(result[index])
        olddata.to_excel(f"./{model_name}_Molecular_Descriptor.xlsx", 'test')
    #models_singles = ["SVM_single", "adaboost_single", "DTree_single", "rforest_single", "BernoulliNB_single", "GernoulliNB_single", "KNeighbors_single", "MLP_single"] 
    # models_singles = ["adaboost_single"]
    # for i in range(len(models_singles)):
    #     model_name = models_singles[i]
    #     model = DataModel(train_data, test_data, resdata, model_name)
    #     model.SimpleRun()
       
       
    #     c = model.ClassificationReportSingle()
    #     print(c)
    #     # with open("./1016/" + model_name + ".txt", 'w+') as f:
    #     #     jsonobj = json.dumps(c)
    #     #     f.write(jsonobj)


    # # 读入真实测试数据
    #     TestData = get_data_to_dict('./Molecular_Descriptor.xlsx', 'test')
    #     result = model.PredictSingle(TestData)
    #     # 将result 写入到 ADMET.xlsx test
    #     olddata = get_data('./ADMET.xlsx', 'test')
    #     for i in range(len(olddata)):
    #         index = olddata.index[i]
    #         olddata.iloc[i,:] = pd.Series(result[index])
    #     #olddata.to_excel(f"./1016/{model_name}_Molecular_Descriptor.xlsx", 'test')


if __name__ == '__main__':
    run()