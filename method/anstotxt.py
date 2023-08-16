import os
from utils import save_var,load_var
import numpy as np

tempname = "CUB200-size"

c = [i  for i in os.listdir("/root/autodl-tmp/method/temp/"+tempname) if "augmented" in i]


f = open('/root/autodl-tmp/method/temp/'+tempname+'/slow-result.txt','w')

data_name = "CUB200"
#     q1 = load_var(path,"Q1_"+data_name+".pkl")
q1 = load_var("CUB200-origin","quickQ1_"+data_name+".pkl")
q2 = load_var("CUB200-origin","Q2_"+data_name+".pkl")
q3 = load_var("CUB200-origin","Q3_"+data_name+".pkl")
q4 = load_var("CUB200-origin","Q4_"+data_name+".pkl")
q5 = load_var("CUB200-origin","Q5_"+data_name+".pkl")
q6 = load_var("CUB200-origin","quickQ6_"+data_name+".pkl")
#     q6 = load_var(path,"Q6_"+data_name+".pkl")

q1_score = np.mean([i[1] for i in q1]) #特征多样性
q2_score = q2 #纹理多样性
q3_score = q3 #亮度多样性
q4_score = q4 #类别数量平衡
q5_score = q5 #类别分类难度
q6_score = np.mean(q6[1]) #任务相关性

f.write(str(format(q1_score,'.4f')))
f.write("\t")
f.write(str(format(q2_score,'.4f')))
f.write("\t")
f.write(str(format(q3_score,'.4f')))
f.write("\t")
f.write(str(format(q4_score,'.4f')))
f.write("\t")
f.write(str(format(q5_score,'.4f')))
f.write("\t")
f.write(str(format(q6_score,'.4f')))
f.write("\n")




for cnt in range(0,6):
    data_name = "augmented-"+str(cnt)
    
#     st = load_var(tempname+"/"+path,"strategy"+str(cnt-1)+".pkl")
#     print(st)

    q1 = load_var(tempname+"/quick-augmented-"+str(cnt),"Q1_"+data_name+".pkl")
    q6 = load_var(tempname+"/quick-augmented-"+str(cnt),"Q6_"+data_name+".pkl")

#     q1 = load_var(tempname+"/quick-augmented-"+str(cnt),"quickQ1_"+data_name+".pkl")
#     q6 = load_var(tempname+"/quick-augmented-"+str(cnt),"quickQ6_"+data_name+".pkl")
    q2 = load_var(tempname+"/quick-augmented-"+str(cnt),"Q2_"+data_name+".pkl")
    q3 = load_var(tempname+"/quick-augmented-"+str(cnt),"Q3_"+data_name+".pkl")
    q4 = load_var(tempname+"/quick-augmented-"+str(cnt),"Q4_"+data_name+".pkl")
    q5 = load_var(tempname+"/quick-augmented-"+str(cnt),"Q5_"+data_name+".pkl")

    ans = load_var(tempname+"/quick-augmented-"+str(cnt),"Qans_"+data_name+".pkl")
#     time = load_var(tempname+"/quick-augmented-"+str(cnt),"cost_time_"+data_name+".pkl")
    
    q1_score = np.mean([i[1] for i in q1]) #特征多样性
    q2_score = q2 #纹理多样性
    q3_score = q3 #亮度多样性
    q4_score = q4 #类别数量平衡
    q5_score = q5 #类别分类难度
    q6_score = np.mean(q6[1]) #任务相关性
    ans = ans[-1]
#     f.write(str(st[0][0])+","+str(st[0][1]))
    f.write(str(cnt))
    f.write("\t")
    f.write(str(format(q1_score,'.4f')))
    f.write("\t")
    f.write(str(format(q2_score,'.4f')))
    f.write("\t")
    f.write(str(format(q3_score,'.4f')))
    f.write("\t")
    f.write(str(format(q4_score,'.4f')))
    f.write("\t")
    f.write(str(format(q5_score,'.4f')))
    f.write("\t")
    f.write(str(format(q6_score,'.4f')))
    f.write("\t")
    f.write(str(format(ans,'.4f')))
#     f.write("\t")
#     f.write(str(time[0][0]/60))
    f.write("\n")
f.close()