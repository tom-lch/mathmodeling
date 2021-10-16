from pkg import get_data, split_train_test_dict, get_data_to_dict
from question4 import GetMEData, GetMEDataSplit
from base import DataModel, Q2Model
import pandas as pd
import json

def run():
    # 程序入口函数
    data = get_data('./Molecular_Descriptor.xlsx', 'training')
    train_data, test_data = split_train_test_dict(data)

    # 将结果输入进来
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


def run_question():
    resdata = get_data_to_dict('./ADMET.xlsx', 'training')
    # 问题2
    print("开始问题2")
    data = get_data('./Molecular_Descriptor.xlsx', 'training')
    ipc = get_data_to_dict('./ERα_activity.xlsx', 'training')
    MoleDescData, ERActiveData = GetMEData(data, ipc)
    model_reg = "svr"
    modelreg = Q2Model(MoleDescData, ERActiveData, model_reg)
    modelreg.Run()
    print("r2:",modelreg.R2Score())
    print("MeanSquaredError:",modelreg.MeanSquaredError())
    print("MeanAbsoluteError:",modelreg.MeanAbsoluteError())

    
    # 问题3
    print("开始问题3")
    train_data = get_data('./Molecular_Descriptor.xlsx', 'training')
    train_data, test_data  = GetMEDataSplit(train_data)
    model_cls = "adaboost"
    modelcls = DataModel(train_data, test_data, resdata, model_cls)
    modelcls.Run()
    c = modelcls.ClassificationReport()
    print(c)



if __name__ == '__main__':
    # question3
    #run()

    # question4
    run_question()