import numpy as np
from utils import save_var,load_var
import torch
import math
from tqdm import tqdm
import time

#手动改
# out_file_path = "augmented-2-10000-1"
# data_name = "augmented-10000"
###

def task_dependancies(out_file_path, data_name):
    T = []
    T.append(time.time())#开始时间
    
#     Q6(out_file_path,data_name)
    quickQ6(out_file_path,data_name)
    
    T.append(time.time())
    costT = T[-1]-T[-2]
    return [["Q6",costT]]

def Q6(out_file_path,data_name):
    q6 = []
    feature_array = load_var(out_file_path,"high_feature_array_"+data_name+".pkl")
    test_feature_array = load_var(out_file_path,"test_feature_array_"+data_name+".pkl")

    class_num = len(feature_array)
    feature_similarity_all = []
    ans_sum = []
    ans_mean = []
    for c in range(class_num):
        s = 0
        #相似度矩阵计算
        feature_similarity = np.zeros([len(feature_array[c]),len(test_feature_array[c])])
        for i in tqdm(range(len(feature_array[c]))):
            for j in range(len(test_feature_array[c])):
                feature_similarity[i][j]=torch.cosine_similarity(torch.unsqueeze(feature_array[c][i],0), torch.unsqueeze(test_feature_array[c][j],0), dim=1)
                s += feature_similarity[i][j]
        ans_sum.append(s)
        #平均相似情况
        num = feature_similarity.shape[0]*feature_similarity.shape[1]
        ans_mean.append(s/num)
        feature_similarity_all.append(feature_similarity)
        q6 =[feature_similarity_all,ans_mean]
    print("Q6:",q6[1])
    save_var(q6,out_file_path,"Q6_"+data_name+".pkl")
    
    del q6, feature_array, test_feature_array, feature_similarity_all#删除大对象
    
def quickQ6(out_file_path,data_name):#聚类加速
    q6 = []
    feature_array = load_var(out_file_path,"dimension_reduction_"+data_name+".pkl")#降维后的
#     feature_array = load_var(out_file_path,"high_feature_array_"+data_name+".pkl")
    test_feature_array = load_var(out_file_path,"test_feature_array_"+data_name+".pkl")

    class_num = len(feature_array)
    feature_similarity_all = []
    ans_sum = []
    ans_mean = []
    for c in range(class_num):
        s = 0
        #相似度矩阵计算
        feature_similarity = np.zeros([len(feature_array[c]),len(test_feature_array[c])])
        print("样本数：",len(feature_array[c]))
        for i in tqdm(range(len(feature_array[c]))):
            for j in range(len(test_feature_array[c])):
                feature_similarity[i][j]=torch.cosine_similarity(torch.unsqueeze(torch.from_numpy(feature_array[c][i]),0), torch.unsqueeze(test_feature_array[c][j],0), dim=1)
                s += feature_similarity[i][j]
        ans_sum.append(s)
        #平均相似情况
        num = feature_similarity.shape[0]*feature_similarity.shape[1]
        ans_mean.append(s/num)
        feature_similarity_all.append(feature_similarity)
        q6 =[feature_similarity_all,ans_mean]
    print("Q6:",q6[1])
    save_var(q6,out_file_path,"quickQ6_"+data_name+".pkl")
    