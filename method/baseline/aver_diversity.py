import numpy as np
from utils import save_var,load_var
import torch,random
import math,time
from tqdm import tqdm
from finch import FINCH

#手动改
# data_name = "augmented-10000"
# out_file_path = "augmented-3-10000-0"
###
#总

def diversity():

    for i in range(0,7):
        out_file_path = "CIFAR10-augmented/quick-augmented-"+str(i)
        data_name = "augmented-"+str(i)
        aver_Q1(out_file_path, data_name) 
        aver_Q6(out_file_path, data_name)
    out_file_path = "CIFAR10-origin"
    data_name = "cifar10"
    aver_Q1(out_file_path, data_name) 
    aver_Q6(out_file_path, data_name)


def aver_Q1(out_file_path, data_name):
    q1 = []
    feature_array = load_var(out_file_path,"high_feature_array_"+data_name+".pkl")
    class_num = len(feature_array)
    
    ans_all = []
    
    feature_similarity_all = [i[0] for i in load_var(out_file_path,"quickQ1_"+data_name+".pkl")]
    for cnt in range(class_num):
        
        feature_similarity = feature_similarity_all[cnt]
        
        #特征值计算
        ans = np.mean(feature_similarity)
        print("it:",cnt,"diversity:",ans)
        q1.append(ans)
        ans_all.append([feature_similarity,ans])
    print("averQ1:",q1)
    save_var(ans_all,out_file_path,"averQ1_"+data_name+".pkl")
    


#低维特征多样性
def aver_Q6(out_file_path, data_name):
    q6 = []
    feature_array = load_var(out_file_path,"dimension_reduction_"+data_name+".pkl")#降维后的
#     feature_array = load_var(out_file_path,"high_feature_array_"+data_name+".pkl")
    test_feature_array = load_var(out_file_path,"test_feature_array_"+data_name+".pkl")

    class_num = len(feature_array)
    feature_similarity_all = load_var(out_file_path,"quickQ6_"+data_name+".pkl")[0]

    ans_mean = []
    for c in range(class_num):
        s = 0
        #相似度矩阵计算
        feature_similarity = feature_similarity_all[c]
        
        #平均相似情况
        ans_mean.append(np.mean(feature_similarity))
        feature_similarity_all.append(feature_similarity)
        q6 =[feature_similarity_all,ans_mean]
    print("Q6:",q6[1])
    save_var(q6,out_file_path,"averQ6_"+data_name+".pkl")
    
# diversity("augmented-3-10000-0","augmented-10000")
diversity()