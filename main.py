from pkg import get_data, split_train_test_dict, get_data_to_dict, GetBestSMILES_pICAndADMET, ValidScore
from question4 import GetMEData, GetMEDataSplit
from base import DataModel, Q2Model, HTree
import pandas as pd
import numpy as np
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
    ipcs = get_data_to_dict('./ERα_activity.xlsx', 'training')

    MoleDescData, ERActiveData = GetMEData(data, ipcs)
    model_reg = "adaBoostRegressor"
    modelreg = Q2Model(MoleDescData, ERActiveData, model_reg)
    modelreg.Run()
    # print("r2:",modelreg.R2Score())
    # print("MeanSquaredError:",modelreg.MeanSquaredError())
    # print("MeanAbsoluteError:",modelreg.MeanAbsoluteError())
    
    # 问题3
    print("开始问题3")
    train_data = get_data('./Molecular_Descriptor.xlsx', 'training')
    train_data, test_data  = GetMEDataSplit(train_data)
    model_cls = "adaboost"
    modelcls = DataModel(train_data, test_data, resdata, model_cls)
    modelcls.Run()
    #c = modelcls.ClassificationReport()

    # 获取最优值，查询文件可以发现 pIC 较高的同时 ADMET 满足>3个   ADMET 最好的是 1 0 0 1 0
    # SMILES="Oc1ccc2c(C(=O)c3ccc(OCCN4CCCCC4)cc3)c(sc2c1)c5ccc(Cl)cc5" 时生物活性最好 但是 ADMET 不符合
    # 需要使用 SMILES 将 ERA 排序后测试ADMET 找到后以此为基础构建分支界限，为了防止出现问题，需要将当前的数据写入文件
    # 使用 ERActiveData 和 resdata 
    value, admet = GetBestSMILES_pICAndADMET(ERActiveData, resdata)
    betterSMILES, pIC = value[0], value[1][1]
    # 获取较好值的20个参数
    arr = MoleDescData[betterSMILES]
    print(betterSMILES, arr, pIC, admet) # CC\C(=C(\CC)/c1ccc(O)cc1)\c2ccc(O)cc2 [21.1690060155951, 3.21999999999999, 7.33074756308966, 9.41067968842725, 9.41067968842725, 18.0, 2.0, 2.0, 0.0, 11.5337783674166, 10.0, 91.2618, 20.0534666183583, 0.0, 12.1977012899777, 4.36243931476262, 14.6640226337448, 4.90758, 0.570504419647412, 8.0] 9.48148606012211 [1, 0, 0, 0, 1]
    res = []
    for i in arr:
        res.append([i, i])
    print("初始值：", res)

    newvalpIC = modelreg.Predict(np.array([arr]))[0]
    print("初始值pIC：",newvalpIC)

    # 根据 arr 构建回溯
    weights = [0.75, 0.80, 0.85,0.90, 0.95, 1.05, 1.1, 1.15, 1.20, 1.25, 1.30]
    root = HTree(arr, 0)
    nodes = []
    maxRes = None
    maxpIC = 0
    bestAMTLE = []
    nodes.append(root)
    while len(nodes) > 0:
        node = nodes.pop()
        if node.index >= 20:
            break
        for w in weights:
            parames = node.Parames
            indexW = root.Parames[node.index] * w
            if node.Parames[node.index] == 0:
                indexW += 0.1
            print("开始第i层：", node.index)
            parames[node.index] = indexW
            newNode = HTree(parames, node.index+1)
            # 验证 parames 是否有效
            valpIC = modelreg.Predict(np.array([parames]))[0]
            valadmet = modelcls.PrefictOne(np.array([parames]))[0]
            score = ValidScore(valadmet)
            nodes.append(newNode)
            print("valpIC: ", valpIC, ", valadmet: ", score)
            if valpIC >= newvalpIC and  score>= 3:
                print("节点有效，修改新节点")
                if indexW < res[node.index][0] :
                    res[node.index][0] = indexW
                if indexW > res[node.index][1]:
                    res[node.index][1] = indexW
                if valpIC > maxpIC:
                    maxRes = parames
                if score == 5:
                    with open('bestAMTLRno.txt', 'a+') as f:
                        f.write(f"{valpIC}, {parames}\n")
    
    print(res, maxRes)

    # arr = [[]]
    # val = modelreg.Predict(np.array(arr))
    # print(val[0])

    # val = modelcls.PrefictOne(np.array(arr))
    # print(val[0])


if __name__ == '__main__':
    # question3
    #run()

    # question4
    run_question()


# 剪枝
# [[9.230745073100247, 23.908437429523538], [1.4040809999999957, 3.63669264708749], [3.196572474885246, 8.27940241004434], [4.103526878138702, 10.628493672981076], [4.103526878138702, 10.628493672981076], [7.8489, 20.329337778750006], [0.8721, 2.25881530875], [1.39004019, 2.25881530875], [0.0, 0.15058768725], [10.020259670788539, 13.026337572025101], [4.3605, 11.29407654375], [39.79470789, 103.0717754720404], [8.744314118935137, 22.64853869552741], [0.0, 0.15058768725], [5.3188076474947765, 13.776177202680627], [1.9022416632022403, 4.926972353839332], [6.39424706944442, 16.561659406479624], [2.1399502590000004, 5.542658416457663], [0.2612068747966168, 0.6443320584045545], [3.4884, 9.035261235]] 