
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os,sys
import numpy as np
import torch
import torchvision
import math
import time
import cv2 as cv
from utils import save_var
from torch.utils.data import DataLoader
from finch import FINCH


#手动改的内容**********
# data_path = "/root/method/data/augemented-3/0"
# test_path = "/root/method/data/cifar10/test/"
# out_file_path = "augmented-3-10000-0"
# data_name = "augmented-10000"
# batch_size = 32


def feature_extract(data_path,test_path,out_file_path,data_name,batch_size=32):
    T = []
    T.append(time.time())#开始时间
    highf(data_path,test_path,out_file_path,data_name,batch_size)
    lowf(data_path,test_path,out_file_path,data_name,batch_size)
    T.append(time.time())
    costT = T[-1]-T[-2]
    print("fe_run_time:",costT,"秒",costT/60,"分钟")
    return [["feature_extract",costT]]

def highf(data_path,test_path,out_file_path,data_name,batch_size=32):
    torch.cuda.empty_cache()
    
    print("GPU:",torch.cuda.is_available())
    model = torchvision.models.resnet101(pretrained=True)#初始化模型
#     model = torchvision.models.densenet161(pretrained=True)#初始化模型

    model.eval()#一定要验证

    model.to('cuda')
    print("model:",next(model.parameters()).device)  # 输出：cuda:0

#     #特征向量抽取
    with torch.no_grad():

        #将模型最后一个分类器初始化，不输出类别，输出图像的特征矩阵
        resnet50_feature_extractor = model
        resnet50_feature_extractor.fc = torch.nn.Linear(2048,2048)  #resnet 
#         resnet50_feature_extractor.classifier = torch.nn.Linear(2208,2208)  #densenet 

        # ---以下几行必须要有：---

        torch.nn.init.zeros_(resnet50_feature_extractor.fc.bias)#resnet
        torch.nn.init.zeros_(resnet50_feature_extractor.fc.bias)
#         torch.nn.init.zeros_(resnet50_feature_extractor.classifier.bias)#densenet
#         torch.nn.init.zeros_(resnet50_feature_extractor.classifier.bias)
        for param in resnet50_feature_extractor.parameters():
            param.requires_grad = False
        # ---------------------
        resnet50_feature_extractor = resnet50_feature_extractor.cuda()
        print("resnet50_feature_extractor:",next(resnet50_feature_extractor.parameters()).device)  # 输出：cuda:0

        #数据集读取

        data = torchvision.datasets.ImageFolder(data_path, transform=transforms.Compose([
                        # transforms.Resize(256),
                        # transforms.CenterCrop(224),
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))
        loader = DataLoader(data, batch_size=batch_size)
        #获取共有多少类
        class_num = len(data.classes)
        #存储所有图片的特征向量
        feature_array = [ [] for i in range(class_num)]

        for img,label in tqdm(loader):
            q_feature = resnet50_feature_extractor(img.cuda())
    #             feature_array[label].append(q_feature)
            for n in range(len(q_feature)):
                feature_array[label[n]].append(q_feature[n].cpu())#一定要cpu() 不然文件大小会随着batchsize指数级增长
        #存储特征向量变量
        save_var(feature_array,out_file_path,"high_feature_array_"+data_name+".pkl")

        #测试数据集读取 
        test_data = torchvision.datasets.ImageFolder(test_path, transform=transforms.Compose([
                        # transforms.Resize(256),
                        # transforms.CenterCrop(224),
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))
        test_loader = DataLoader(test_data, batch_size=batch_size)
        #获取共有多少类
        test_class_num = len(test_data.classes)
        #存储所有图片的特征向量
        test_feature_array = [ [] for i in range(test_class_num)]

        for img,label in tqdm(test_loader):
            q_feature = resnet50_feature_extractor(img.cuda())
    #             feature_array[label].append(q_feature)
            for n in range(len(q_feature)):
                test_feature_array[label[n]].append(q_feature[n].cpu())#一定要cpu() 不然文件大小会随着batchsize指数级增长
        #存储特征向量变量
        save_var(test_feature_array,out_file_path,"test_feature_array_"+data_name+".pkl")

    del model,test_feature_array
    torch.cuda.empty_cache()


def lowf(data_path,test_path,out_file_path,data_name,batch_size=32):
#     #低维度特征抽取
    eps = np.spacing(1)
    
#     all_imgs = torchvision.datasets.ImageFolder(data_path,transform=transforms.Compose([transforms.Resize((224,224))]))#cub200
    all_imgs = torchvision.datasets.ImageFolder(data_path)#cifar10
    #亮度
    c_imgs=[[],[],[]]

    for img,label in tqdm(all_imgs):
        c_imgs[0].append(np.array(img)[:,:,0].reshape(-1))
        c_imgs[1].append(np.array(img)[:,:,1].reshape(-1))
        c_imgs[2].append(np.array(img)[:,:,2].reshape(-1))

    for i in tqdm(range(3)):
        c_imgs[i] = np.array(c_imgs[i]).reshape(-1)
    c_imgs = np.array(c_imgs)
    cutlenth = 50
    print("cutlenth:",cutlenth)
    
    pixel = np.zeros([3,256])
    for cnt in tqdm(range(3)):
        if len(c_imgs[cnt])>cutlenth: 
            for i in range(int(len(c_imgs[cnt])/cutlenth)):
                pixel[cnt] += np.bincount(c_imgs[cnt][cutlenth*i:cutlenth*(i+1)],minlength=256)
            pixel[cnt] += np.bincount(c_imgs[cnt][cutlenth*int(len(c_imgs[cnt])/cutlenth):len(c_imgs[cnt])],minlength=256)
        else:
            pixel[cnt] += np.bincount(c_imgs[cnt],minlength=256)

    Tp=[0,0,0]
    for i in range(3):
        pixel[i] = pixel[i] / pixel[i].sum()
        for j in range(256):
            Tp[i] = Tp[i] - pixel[i][j] * np.log2(pixel[i][j] + eps)
    Tp = sum(Tp)/3
    
    all_imgs = torchvision.datasets.ImageFolder(data_path,transform=transforms.Compose([transforms.Resize((4,4))]))
    #亮度
    c_imgs=[[],[],[]]

    for img,label in tqdm(all_imgs):
        c_imgs[0].append(np.array(img)[:,:,0].reshape(-1))
        c_imgs[1].append(np.array(img)[:,:,1].reshape(-1))
        c_imgs[2].append(np.array(img)[:,:,2].reshape(-1))
    for i in range(3):
        c_imgs[i] = np.array(c_imgs[i]).reshape(-1)
    c_imgs = np.array(c_imgs)

    texture = np.zeros([3,256])
    for cnt in range(3):
        if len(c_imgs[cnt])>cutlenth: 
            for i in range(int(len(c_imgs[cnt])/cutlenth)):
                texture[cnt] += np.bincount(c_imgs[cnt][cutlenth*i:cutlenth*(i+1)],minlength=256)
            texture[cnt] += np.bincount(c_imgs[cnt][cutlenth*int(len(c_imgs[cnt])/cutlenth):len(c_imgs[cnt])],minlength=256)
        else:
            texture[cnt] += np.bincount(c_imgs[cnt],minlength=256)

    Tt=[0,0,0]
    for i in range(3):
        texture[i] = texture[i] / texture[i].sum()
        for j in range(256):
            Tt[i] = Tt[i] - texture[i][j] * np.log2(texture[i][j] + eps)
    Tt = sum(Tt)/3

#     #类别数量
    class_num = len(all_imgs)
    class_num = [ 0 for i in range(class_num)]
    for i in all_imgs:
        class_num[i[1]]+=1
    ans = [[texture,Tt],[pixel,Tp],class_num]
#     ans = [[1,1],[1,1],class_num]
    save_var(ans,out_file_path,"low_feature_array_"+data_name+".pkl")

    del all_imgs,ans,c_imgs

    