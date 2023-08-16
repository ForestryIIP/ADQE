from utils import save_var,load_var
import numpy as np


def qualitysort(origin_path,origin_name,augmented_path, data_name):
    feature_array = load_var(augmented_path,"high_feature_array_"+data_name+".pkl")
    num=0
    for i in range(len(feature_array)):
        num+=len(feature_array[i])
    Q = getscore(origin_path, origin_name)
    q = getscore(augmented_path, data_name)
    show(Q)
    show(q)
    print(num)
    ans = calculate(Q,q,num)
    print("ans:",ans)
    q.append(ans)
    save_var(q,augmented_path,"Qans_"+data_name+".pkl")
    
    del feature_array #删除大对象

    
    return ans



def getscore(path,data_name):
#     q1 = load_var(path,"Q1_"+data_name+".pkl")
    q1 = load_var(path,"quickQ1_"+data_name+".pkl")
    q2 = load_var(path,"Q2_"+data_name+".pkl")
    q3 = load_var(path,"Q3_"+data_name+".pkl")
    q4 = load_var(path,"Q4_"+data_name+".pkl")
    q5 = load_var(path,"Q5_"+data_name+".pkl")
    q6 = load_var(path,"quickQ6_"+data_name+".pkl")
#     q6 = load_var(path,"Q6_"+data_name+".pkl")
    q1_score = np.mean([i[1] for i in q1]) #特征多样性
    q2_score = q2 #纹理多样性
    q3_score = q3 #亮度多样性
    q4_score = q4 #类别数量平衡
    q5_score = q5 #类别分类难度
    q6_score = np.mean(q6[1]) #任务相关性
    return [q1_score,q2_score,q3_score,q4_score,q5_score,q6_score]

def show(t):
    x = [ ("Q"+str(i+1),score) for i,score in enumerate(t)]
    print(x)
    return x

def calculate(p,t,n):
#     ans = 1
#     ans *= (t[0]/p[0]) * (t[1]/p[1]) * (t[2]/p[2]) * (t[3]/(p[3]*n)) * (t[4]/p[4]) * (t[5]/p[5])
    ans = (t[5]/p[5])*(2*(t[0]/p[0]) + 0.5*(t[1]/p[1]) + 0.5*(t[2]/p[2]) - (t[3]/((p[3]+1)*n)) + (t[4]/p[4]) )/4
#     ans = (t[5]/p[5])*(t[0]/p[0])*((t[1]/p[1])**0.5)*((t[2]/p[2])**0.5)*(1-(t[3]/((p[3]+1)*n)))*(t[4]/p[4])
    return ans