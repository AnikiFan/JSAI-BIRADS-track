from cv2 import imread
from .utils import *
import torch
import os
import numpy as np
from ultralytics import YOLO
from line_profiler import profile
import pandas as pd
TASK = "classify" # 分类任务
YOLO_PARAMS = {"imgsz":224,"half":True,"int8":False,"device":"cuda:0","verbose":False} # 采用半精度模型，调用GPU推理
class Tester:
    @torch.no_grad()
    @profile
    def __init__(self,model_folder_path,format='engine',half=True):
        # 实例化cla所需模型
        try:
            self.cla_full = YOLO(os.path.join(model_folder_path,'cla','full.'+format),task=TASK)
        except Exception as e:
            format = 'torchscript'
            self.cla_full = YOLO(os.path.join(model_folder_path,'cla','full.'+format),task=TASK)           
        self.cla_full(**YOLO_PARAMS)
        self.cla_0_1 = YOLO(os.path.join(model_folder_path,'cla','full_2_3.'+format),task=TASK)
        self.cla_01_2345 = YOLO(os.path.join(model_folder_path,'cla','full_23_4A4B4C5.'+format),task=TASK)
        self.cla_2_345 = YOLO(os.path.join(model_folder_path,'cla','full_4A_4B4C5.'+format),task=TASK)
        self.cla_3_45 = YOLO(os.path.join(model_folder_path,'cla','full_4B_4C5.'+format),task=TASK)
        self.cla_4_5 = YOLO(os.path.join(model_folder_path,'cla','full_4C_5.'+format),task=TASK)

        self.fea2cla_01_2345 = fea2cla_model.load_model(os.path.join(model_folder_path,'cla','fea2cla_01_2345_best.pkl'))  # 使用了testA和train
        
        # 实例化boudary特征所需模型
        self.boundary_full = YOLO(os.path.join(model_folder_path,'boundary','full.'+format),task=TASK)

        # 实例化calcification特征所需模型
        self.calcification_full = YOLO(os.path.join(model_folder_path,'calcification','full.'+format),task=TASK)

        # 实例化direction特征所需模型
        self.direction_full = YOLO(os.path.join(model_folder_path,'direction','full.'+format),task=TASK)

        # 实例化shape特征所需模型
        self.shape_full = YOLO(os.path.join(model_folder_path,'shape','full.'+format),task=TASK)

    @torch.no_grad()
    @profile
    def fea_predict_probs(self,image):
        # 在cla数据集上用fea模型推理
        boundary_full_result = self.boundary_full(image,**YOLO_PARAMS)[0].probs.data[1]
        direction_full_result = self.direction_full(image,**YOLO_PARAMS)[0].probs.data[1]
        calcification_full_result = self.calcification_full(image,**YOLO_PARAMS)[0].probs.data[1]
        shape_full_result = self.shape_full(image,**YOLO_PARAMS)[0].probs.data[1]
        #! 顺序 boundary, calcification, direction, shape 
        #! 注意：calcification 0 代表，1 代表有，而cla中“微钙化”才是恶化的条件
        # 形状不规则 shape 0 规则，1 不规则
        # 垂直生长 direction 0 水平，1 垂直
        # 边缘不光整 boundary 0 光整，1 不光整
        # 微钙化 calcification 0 有，1 无
        # 但是都当1才会有好的效果
        return torch.tensor([boundary_full_result,calcification_full_result,direction_full_result,shape_full_result],device='cpu')
        

    def get_basic_features(self,image):
        # 构造输入数据
        data = self.fea_predict_probs(image)
        data = pd.DataFrame([data],columns=["boundary_pre_probs","calcification_pre_probs","direction_pre_probs","shape_pre_probs"])
        data["fea_probs_sum"] = data.sum(axis=1)
        threshold = 0.5
        data["boundary_pre"] = data["boundary_pre_probs"] > threshold
        data["calcification_pre"] = data["calcification_pre_probs"] > threshold
        data["direction_pre"] = data["direction_pre_probs"] > threshold
        data["shape_pre"] = data["shape_pre_probs"] > threshold
        return data
    
    @torch.no_grad()
    @profile
    def cla_predict(self,image):
        origin = imread(image) # 读取图像
        origin = self.cla_full.predictor.preprocess([origin]) # 预处理
        P_basic = self.cla_full(origin,**YOLO_PARAMS)[0].probs.data
        '''--------------------------------- fea2cla ---------------------------------'''
        data = self.get_basic_features(origin)
        P_fea2cla_01_2345 = self.fea2cla_01_2345.predict(data,return_prob=True)[0] # ! 注意，只用到了这一个，后面注释了
        # 利用P_fea2cla_01_2345更新P_basic

        P_basic_copy = P_basic.clone()

        if P_fea2cla_01_2345[0] > 0.80:  # 如果特征模型强烈倾向于前两类(0,1)
            # 增强P_basic中前两类的概率
            P_basic_copy[0:2] = P_basic_copy[0:2] * 1.2
            # 降低后四类的概率
            P_basic_copy[2:6] = P_basic_copy[2:6] * 0.8
        elif P_fea2cla_01_2345[1] > 0.80:  # 如果特征模型强烈倾向于后四类(2,3,4,5)
            # 降低前两类的概率
            P_basic_copy[0:2] = P_basic_copy[0:2] * 0.8
            # 增强后四类的概率
            P_basic_copy[2:6] = P_basic_copy[2:6] * 1.2
        # 重新归一化P_basic_updated_2
        # P_basic_updated_2 = F.softmax(P_basic_updated_2, dim=0) # 效果变得很糟糕
        P_basic = P_basic / torch.sum(P_basic)

        
        P_bayes = torch.zeros(6).cuda()
        # ----------------------------- layer 1 区分 01|2345 -----------------------------
        p_01,p_2345 = self.cla_01_2345(origin,**YOLO_PARAMS)[0].probs.data
        P_bayes[0:2] = p_01
        P_bayes[2:6] = p_2345
        if p_01 < p_2345: # 01 < 2345
            # [p01/2, p01/2, p2345, p2345, p2345]
            P_bayes[0:2] = P_bayes[0:2]/2
            # ----------------------------- layer 2 区分 2|345 -----------------------------
            p_2,p_345 = self.cla_2_345(origin,**YOLO_PARAMS)[0].probs.data # [p2,p345]
            # [p01/2, p01/2, p2345*p2, p2345*p345, p2345*p345, p2345*p345]
            P_bayes[2:3] = P_bayes[2:3]*p_2
            P_bayes[3:6] = P_bayes[3:6]*p_345
            if p_2 < p_345: # 2 < 345
                # ----------------------------- layer 3 区分 3|45 -----------------------------
                p_3,p_45 = self.cla_3_45(origin,**YOLO_PARAMS)[0].probs.data # [p3,p45]
                # [p01/2, p01/2, p2345*p2, p2345*p345*p3, p2345*p345*p45, p2345*p345*p45]
                P_bayes[3:4] = P_bayes[3:4]*p_3
                P_bayes[4:6] = P_bayes[4:6]*p_45
                if p_3 < p_45: # 3 < 4,5
                    # ----------------------------- layer 4 区分 4|5 -----------------------------
                    p_4,p_5 = self.cla_4_5(origin,**YOLO_PARAMS)[0].probs.data # [p4,p5]
                    # [p01/2, p01/2, p2345*p2, p2345*p345*p3, p2345*p345*p45*p4, p2345*p345*p45*p5]  final
                    P_bayes[4] = P_bayes[4]*p_4
                    P_bayes[5] = P_bayes[5]*p_5
                else: # 3 > 4,5 | 不需要再区分
                    # [p01/2, p01/2, p2345*p2, p2345*p345*p3, p2345*p345*p45/2, p2345*p345*p45/2] final 
                    P_bayes[4:6] = P_bayes[4:6]/2
            else: # 2 > 3,4,5
                # [p01/2, p01/2, p2345*p2, p2345*p345/3, p2345*p345/3, p2345*p345/3] final
                P_bayes[3:6] = P_bayes[3:6]/3
        else: # 0,1 > 2,3,4,5
            # [p01, p01, p2345/4, p2345/4, p2345/4, p2345/4]
            P_bayes[2:6] = P_bayes[2:6]/4
            # ----------------------------- layer 2 区分 0,1|2 -----------------------------
            p_0,p_1 = self.cla_0_1(origin,**YOLO_PARAMS)[0].probs.data # [p0,p1]
            # [p01*p0, p01*p1, p2345/4, p2345/4, p2345/4, p2345/4] final
            P_bayes[0] = P_bayes[0]*p_0
            P_bayes[1] = P_bayes[1]*p_1
            
        result = self.majority_ensemble([P_basic,P_bayes])
        return result
    
    @torch.no_grad()
    @profile
    def fea_predict(self, image):
        origin = imread(image) # 读取图像
        #x,y,w,h = cv_crop(origin)
        #origin = origin[y:y+h,x:x+w]
        origin = self.cla_full.predictor.preprocess([origin]) # 预处理
        # 使用YOLO模型推理
        # boundary特征
        boundary = self.boundary_full(origin,**YOLO_PARAMS)[0].probs.top1
        # calcification特征
        calcification = self.calcification_full(origin,**YOLO_PARAMS)[0].probs.top1
        # direction特征
        direction = self.direction_full(origin,**YOLO_PARAMS)[0].probs.top1
        # shape特征
        shape = self.shape_full(origin,**YOLO_PARAMS)[0].probs.top1
        return boundary,calcification,direction,shape
    

    @staticmethod
    def average_ensemble(tensors):
        # 以average方式集成
        return argmax(mean(stack(tensors,dim=0),dim=0)).item()

    @staticmethod
    def max_ensemble(tensors):
        # 以max方式集成
        return torch.argmax(torch.max(torch.stack(tensors,dim=0),dim=0)[0]).item()

    @staticmethod
    def majority_ensemble(tensors):
        # 以majority方式集成
        return torch.argmax(torch.sum(torch.stack(tensors,dim=0)>0.5,dim=0),dim=0).item()


