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
def diversity(out_file_path, data_name):
    
    T = []
    T.append(time.time())#开始时间
    
#     Q1(out_file_path, data_name)
    quickQ1(out_file_path, data_name) 
    
    T.append(time.time())
    Q1costT = T[-1]-T[-2]

    Q2Q3(out_file_path, data_name)
    
    T.append(time.time())
    Q2Q3costT = T[-1]-T[-2]
    return [["Q1",Q1costT],["Q2",Q2Q3costT],["Q3",Q2Q3costT]]

#特征向量多样性
def Q1(out_file_path, data_name):
    q1 = []
    feature_array = load_var(out_file_path,"high_feature_array_"+data_name+".pkl")
    class_num = len(feature_array)
    feature_similarity_all = []
    for c in range(class_num):
        #相似度矩阵计算
        feature_similarity = np.zeros([len(feature_array[c]),len(feature_array[c])])
        print("样本数：",len(feature_array[c]))
        for i in tqdm(range(len(feature_array[c]))):
            feature_similarity[i][i]=1
            for j in range(i):
                feature_similarity[i][j]=torch.cosine_similarity(torch.unsqueeze(feature_array[c][i],0), torch.unsqueeze(feature_array[c][j],0), dim=1)
                feature_similarity[j][i]=feature_similarity[i][j]
        #特征值计算
        val,vecs = np.linalg.eig(np.divide(feature_similarity,len(feature_similarity)))
#         print("val:",val)
        ans=0
        #计算多样性熵
        for i in val:
            if i > 1e-6:
                # print(i)
                ans-=i*math.log(i)
        ans = math.exp(ans)
        print("it:",c,"diversity:",ans)
        q1.append(ans)
        feature_similarity_all.append([feature_similarity,ans])

        del feature_similarity
            
    print("Q1:",q1)
    save_var(feature_similarity_all,out_file_path,"Q1_"+data_name+".pkl")
    
    del feature_array,feature_similarity_all

def quickQ1(out_file_path, data_name):
    q1 = []
    feature_array = load_var(out_file_path,"high_feature_array_"+data_name+".pkl")
    class_num = len(feature_array)
    
    all_cluster = []
    
    feature_similarity_all = []
    for cnt in range(class_num):
        samples = []
        labels=[]
        
        for j in feature_array[cnt]:
            samples.append(j.numpy())
            labels.append(cnt)
        data = np.array(samples)
        c, num_clust, req_c = FINCH(data, verbose=False)#抽1/10个典型样本
        req_c = c[:,0]
        cluster_num = len(np.unique(req_c))
        print(cluster_num)

        #抽聚类个数个样本
        cluster_array = []
        for i in range(cluster_num):
            id_index=[h for h in range(len(req_c)) if req_c[h] == i]
            index = random.sample(id_index,k=1)
            cluster_array.append(samples[index[0]])
        #相似度矩阵计算
        all_cluster.append(cluster_array)
        feature_similarity = np.zeros([len(cluster_array),len(cluster_array)])
        for i in tqdm(range(len(cluster_array))):
            feature_similarity[i][i]=1
            for j in range(i):
                feature_similarity[i][j]=torch.cosine_similarity(torch.unsqueeze(torch.from_numpy(cluster_array[i]),0), torch.unsqueeze(torch.from_numpy(cluster_array[j]),0), dim=1)
                feature_similarity[j][i]=feature_similarity[i][j]
    
        #特征值计算
        val,vecs = np.linalg.eig(np.divide(feature_similarity,len(feature_similarity)))
#         print("val:",val)
        ans=0
        #计算多样性熵
        for i in val:
            if i > 1e-6:
                # print(i)
                ans-=i*math.log(i)
        ans = math.exp(ans)
        print("it:",cnt,"diversity:",ans)
        q1.append(ans)
        feature_similarity_all.append([feature_similarity,ans])
    print("quickQ1:",q1)
    save_var(feature_similarity_all,out_file_path,"quickQ1_"+data_name+".pkl")
    save_var(all_cluster,out_file_path,"dimension_reduction_"+data_name+".pkl")
    
    del feature_similarity_all,all_cluster

#低维特征多样性
def Q2Q3(out_file_path, data_name):
    feature_array = load_var(out_file_path,"low_feature_array_"+data_name+".pkl")
    print("Q2:",feature_array[0][1])
    save_var(feature_array[0][1],out_file_path,"Q2_"+data_name+".pkl")
    print("Q3:",feature_array[1][1])
    save_var(feature_array[1][1],out_file_path,"Q3_"+data_name+".pkl")
    
    del feature_array#删除大对象
    
# diversity("augmented-3-10000-0","augmented-10000")