from utils import save_var
from feature_extraction import feature_extract
from class_balance import class_balance
from diversity import diversity,Q2Q3
from task_dependencies import task_dependancies
from sort_by_quality import qualitysort
import time

# '''
origin_path = "CUB200-origin"
origin_name = "CUB200"
for i in range(0,7):
    T = []
    partT = []
    T.append(time.time())#开始时间
    print("--start--")

    data_path = "/root/autodl-tmp/method/data/data/augmented-"+str(i)
    test_path = "/root/autodl-tmp/method/data/data/cifar10/test/"
    out_file_path = "CUB200/quick-augmented-"+str(i)
    data_name = "augmented-"+str(i)
    batch_size = 32

    print("data_path",data_path)
    print("out_file_path",out_file_path)
    print("data_name",data_name)

#     print("--feature_extract--")
#     partT.append(feature_extract(data_path, test_path, out_file_path, data_name, batch_size))
#     print("--class_balance--")
#     partT.append(class_balance(out_file_path, data_name))
#     print("--diversity--")
#     partT.append(diversity(out_file_path, data_name))
#     print("--task_dependancies--")
#     partT.append(task_dependancies(out_file_path, data_name))
    print("--qualitysort--")
    qualitysort(origin_path,origin_name,out_file_path, data_name)
    print("--finish--")
    
    T.append(time.time())
    costT = T[-1]-T[-2]
    total_time = []
    total_time_minute = []
    total_time.append(costT)
    for t in partT:
        for j in t:
            j[1]/=60
            total_time.append(j)
#     save_var([total_time],out_file_path,"slow_cost_time_"+data_name+".pkl")
    print("总共时间:",total_time[0]/60,"分钟")
    
    
'''

#原始数据集处理
data_path = "/root/autodl-tmp/method/data/data/cifar10/train"
test_path = "/root/autodl-tmp/method/data/data/cifar10/test"
out_file_path = "CIFAR10-pretrain/origin"
data_name = "cifar10"
batch_size = 32

print("data_path",data_path)
print("out_file_path",out_file_path)
print("data_name",data_name)

print("--feature_extract--")
feature_extract(data_path, test_path, out_file_path, data_name, batch_size)
print("--class_balance--")
class_balance(out_file_path, data_name)
print("--diversity--")
diversity(out_file_path, data_name)
print("--task_dependancies--")
task_dependancies(out_file_path, data_name)

# #结束总结输出
# for i in range(0,6):
#     data_path = "/root/method/data/augmented-"+str(i)
#     test_path = "/root/method/data/cifar10/test/"
#     out_file_path = "1/augmented-"+str(i)
#     data_name = "augmented-"+str(i)
#     batch_size = 32
    
#     stategy = load_var("1","strategy"+str(i)+".pkl")
#     ans = load_var(out_file_path,"Qans_"+data_name+".pkl")
    
#     pp = [i for i in ans]
#     pp.append(stategy)
#     p.append(pp)
# with open('/root/method/temp/1/ans_augmented.txt','w') as f:
#     for i in p:
#         for j in i:
#             if type(j) == np.float64:
#                 f.write(str(format(j, '.4f')))
#             else:
#                 f.write(str(j))
#             f.write("\t")
#         f.write("\n")
#     f.close()
'''