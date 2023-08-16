import numpy as np
from utils import save_var,load_var
import torch
import math,random,time

import h5py
from finch import FINCH
from sklearn.metrics import normalized_mutual_info_score as nmi

#手动改
# out_file_path = "augmented-2-10000-1"
# data_name = "augmented-10000"
# q5k = 
###

def class_balance(out_file_path, data_name):
    T = []
    T.append(time.time())#开始时间
    
    feature_array = load_var(out_file_path,"high_feature_array_"+data_name+".pkl")
    Q4(out_file_path, data_name, feature_array)
    
    T.append(time.time())
    Q4costT = T[-1]-T[-2]
    
    Q5(out_file_path, data_name, feature_array)
    
    T.append(time.time())
    Q5costT = T[-1]-T[-2]
    
    del feature_array#删除大对象
    
    return [["Q4",Q4costT],["Q5",Q5costT]]


def Q4(out_file_path, data_name, feature_array):#数量平衡
    class_num = [ len(feature_array[i]) for i in range(len(feature_array))]
    q4 = np.var(class_num)
    print("Q4:",q4)
    save_var(q4,out_file_path,"Q4_"+data_name+".pkl")
    
def Q5(out_file_path, data_name, feature_array):#特征平衡
    samples = []
    labels = []
    for i in range(len(feature_array)):
        for j in feature_array[i]:
            samples.append(j.numpy())
            labels.append(i)
    data = np.array(samples)
    gt = np.squeeze(np.array(labels))
    print(data.shape)
    c, num_clust, req_c = FINCH(data, verbose=False)
    print(c.shape)
    print(num_clust)
    print(req_c)
    
    q5 = 0
    f=-1
    for i in range(c.shape[1]):
        score = nmi(gt, c[:,i])
        if q5<score:
            q5 = score
            f = i
        print('NMI Score'+str(num_clust[i])+': {:.2f}'.format(score * 100))
    print('Q5: {:.2f}'.format(q5 * 100))
    
#     #对类内元素根据聚类结果去重
#     if f != c.shape[1]:
#         newc = c[:,f]
#     else:
#         newc = req_c
    
#     samples = []
#     cnt=0
#     for i in range(len(feature_array)):
#         for j in feature_array[i]:
#             samples.append([j.numpy(),i,newc[cnt]])
#             cnt+=1
#     data = np.array(samples)
    
#     newclass = []
#     for i in range(len(feature_array)):
#         t = []
#         for j in range(len(np.unique(newc))):
#             t1 = [data[k][0] for k in range(len(data)) if data[k][1] == i and data[k][2] == j]
#             if len(t1)!=0:
#                 t.append([random.sample(t1,k=1),len(t1)])
#         newclass.append(t)
    
    save_var(q5,out_file_path,"Q5_"+data_name+".pkl")
#     save_var([gt, c, num_clust, req_c, newclass],out_file_path,"cluster_result_"+data_name+".pkl")
    save_var([gt, c, num_clust, req_c],out_file_path,"cluster_result_"+data_name+".pkl")