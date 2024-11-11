import pandas as pd
from .segmentation.segmentation_model import FCBFormer
from torchvision import transforms
import torch.nn.functional as F
import cv2
from .utils import removeFrame,make_mask,make_box_map,make_masked
import torch
import numpy as np
import os
from ultralytics import YOLO
from logging import info
from line_profiler import profile
from .cla.fea2cla.model import fea2cla_model
from collections import defaultdict



def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    # elif torch.backends.mps.is_available():
    #     return torch.device('mps')
    else:
        return torch.device('cpu')


class Tester:
    def __init__(self,model_folder_path):
        info("init test")
        self.seg_model = FCBFormer().to(get_device()) # 实例化FCB模型
        self.seg_model.load_state_dict(torch.load(os.path.join(model_folder_path,'segmentation','FCB_checkpoint.pt'),map_location=get_device())) # 加载预训练模型
        # 自定义图像转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((352,352),antialias=True),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        self.seg_model.eval()


        # 实例化cla所需模型
        self.cla_full = YOLO(os.path.join(model_folder_path,'cla','full.pt'))
        # self.cla_full = YOLO(os.path.join(model_folder_path,'cla','full_final.pt'))  #yzl 11月11号tune的模型
        self.cla_0_1 = YOLO(os.path.join(model_folder_path,'cla','full_2_3_t.pt'))
        self.cla_01_2345 = YOLO(os.path.join(model_folder_path,'cla','full_23_4A4B4C5_t.pt'))
        self.cla_2_345 = YOLO(os.path.join(model_folder_path,'cla','full_4A_4B4C5_t.pt'))
        self.cla_3_45 = YOLO(os.path.join(model_folder_path,'cla','full_4B_4C5_t.pt'))
        self.cla_4_5 = YOLO(os.path.join(model_folder_path,'cla','full_4C_5_t.pt'))
        # self.fea2cla_01_2345 = fea2cla_model.load_model(os.path.join(model_folder_path,'cla','fea2cla_01_2345_best.pkl'))  # 使用了testA和train
        self.fea2cla_01_2345 = fea2cla_model.load_model(os.path.join(model_folder_path,'cla','fea2cla_01_2345_train_only.pkl'))  # 只使用了train

        self.fea2cla_012_345 = fea2cla_model.load_model(os.path.join(model_folder_path,'cla','fea2cla_012_345_best.pkl'))
        self.fea2cla_0123_45 = fea2cla_model.load_model(os.path.join(model_folder_path,'cla','fea2cla_0123_45_best.pkl'))
        
        # 实例化boudary特征所需模型
        self.boundary_full = YOLO(os.path.join(model_folder_path,'boundary','full.pt'),task="classify")
        self.boundary_box = YOLO(os.path.join(model_folder_path,'boundary','box.pt'),task="classify")

        # 实例化calcification特征所需模型
        self.calcification_full = YOLO(os.path.join(model_folder_path,'calcification','full.pt'),task="classify")
        self.calcification_box = YOLO(os.path.join(model_folder_path,'calcification','box.pt'),task="classify")
        self.calcification_masked = YOLO(os.path.join(model_folder_path,'calcification','masked.pt'),task="classify")

        # 实例化direction特征所需模型
        self.direction_full = YOLO(os.path.join(model_folder_path,'direction','full.pt'),task="classify")

        # 实例化shape特征所需模型
        self.shape_full = YOLO(os.path.join(model_folder_path,'shape','full.pt'),task="classify")
        self.shape_box = YOLO(os.path.join(model_folder_path,'shape','box.pt'),task="classify")
        self.shape_masked = YOLO(os.path.join(model_folder_path,'shape','masked.pt'),task="classify")
        info("finish init")
        
        for model in [self.cla_full, self.cla_0_1, self.cla_01_2345, self.cla_2_345, self.cla_3_45, self.cla_4_5,
                     self.boundary_full, self.boundary_box,
                     self.calcification_full, self.calcification_box, self.calcification_masked,
                     self.direction_full,
                     self.shape_full, self.shape_box, self.shape_masked]:
            model.to(get_device())

    @torch.no_grad()
    @profile
    def cla_predict(self,image):
        info("cla_predict开始")
        info("cla读取图像")
        origin = cv2.imread(image)
        info("cla_full")
        cla_full = self.cla_full(origin,verbose=False)[0].probs.data
        info("cla_01_2345")
        cla_01_2345__01,cla_01_2345__2345 = self.cla_01_2345(origin,verbose=False)[0].probs.data
        info("cla_0_1")
        cla_0_1__0,cla_0_1__1 = self.cla_0_1(origin,verbose=False)[0].probs.data
        info("cla_2_345")
        cla_2_345__2,cla_2_345__345 = self.cla_2_345(origin,verbose=False)[0].probs.data
        info("cla_3_45")
        cla_3_45__3,cla_3_45__45 = self.cla_3_45(origin,verbose=False)[0].probs.data
        info("cla_4_5")
        cla_4_5__4,cla_4_5__5 = self.cla_4_5(origin,verbose=False)[0].probs.data
        info("bayes")
        prob_result = torch.Tensor([
        cla_01_2345__01 * cla_0_1__0,
        cla_01_2345__01 * cla_0_1__1,
        cla_01_2345__2345 * cla_2_345__2,
        cla_01_2345__2345 * cla_2_345__345 * cla_3_45__3,
        cla_01_2345__2345 * cla_2_345__345 * cla_3_45__45 * cla_4_5__4,
        cla_01_2345__2345 * cla_2_345__345 * cla_3_45__45 * cla_4_5__5,
        ]).to(get_device())
        info("ensemble cla")
        result = Tester.majority_ensemble([cla_full,prob_result])
        return result

    def get_basic_features(self,image):
        # data = pd.DataFrame(columns=["boundary_pre_probs","calcification_pre_probs","direction_pre_probs","shape_pre_probs"])
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
    def cla_predict2(self,image):
        '''--------------------------------- basic yolo ---------------------------------'''
        P_basic = self.cla_full(image,verbose=False)[0].probs.data
        '''--------------------------------- fea2cla ---------------------------------'''
        data = self.get_basic_features(image)
        P_fea2cla_01_2345 = self.fea2cla_01_2345.predict(data,return_prob=True)[0] # ! 注意，只用到了这一个，后面注释了
        # P_fea2cla_012_345 = self.fea2cla_012_345.predict(data,return_prob=True)[0]
        # P_fea2cla_0123_45 = self.fea2cla_0123_45.predict(data,return_prob=True)[0]
        # P_fea2cla_0 = (P_fea2cla_01_2345[0]/2 + P_fea2cla_012_345[0]/3 + P_fea2cla_0123_45[0]/4)/3
        # P_fea2cla_1 = (P_fea2cla_01_2345[0]/2 + P_fea2cla_012_345[0]/3 + P_fea2cla_0123_45[0]/4)/3
        # P_fea2cla_2 = (P_fea2cla_01_2345[1]/4 + P_fea2cla_012_345[0]/3 + P_fea2cla_0123_45[0]/4)/3
        # P_fea2cla_3 = (P_fea2cla_01_2345[1]/4 + P_fea2cla_012_345[1]/3 + P_fea2cla_0123_45[0]/4)/3
        # P_fea2cla_4 = (P_fea2cla_01_2345[1]/4 + P_fea2cla_012_345[1]/3 + P_fea2cla_0123_45[1]/2)/3
        # P_fea2cla_5 = (P_fea2cla_01_2345[1]/4 + P_fea2cla_012_345[1]/3 + P_fea2cla_0123_45[1]/2)/3
        # P_fea2cla = [P_fea2cla_0,P_fea2cla_1,P_fea2cla_2,P_fea2cla_3,P_fea2cla_4,P_fea2cla_5]
        '''--------------------------------- bayes ---------------------------------'''
        P_bayes = torch.zeros(6).to(get_device())
        result_01_2345 = torch.zeros(2).to(get_device())
        result_2_345 = torch.zeros(2).to(get_device())
        result_3_45 = torch.zeros(2).to(get_device())
        result_4_5 = torch.zeros(2).to(get_device())
        
        # ----------------------------- layer 1 区分 01|2345 -----------------------------
        result_01_2345 = self.cla_01_2345(image,verbose=False)[0].probs.data
        P_bayes[0:2] = torch.tensor([result_01_2345[0],result_01_2345[0]]).to(get_device())
        P_bayes[2:6] = torch.tensor([result_01_2345[1],result_01_2345[1],result_01_2345[1],result_01_2345[1]])
        if result_01_2345[0] < result_01_2345[1]: # 01 < 2345
            # [p01/2, p01/2, p2345, p2345, p2345]
            P_bayes[0:2] = P_bayes[0:2]/2
            # ----------------------------- layer 2 区分 2|345 -----------------------------
            result_2_345 = self.cla_2_345(image,verbose=False)[0].probs.data # [p2,p345]
            # [p01/2, p01/2, p2345*p2, p2345*p345, p2345*p345, p2345*p345]
            P_bayes[2:3] = P_bayes[2:3]*result_2_345[0]
            P_bayes[3:6] = P_bayes[3:6]*result_2_345[1]
            if result_2_345[0] < result_2_345[1]: # 2 < 345
                # ----------------------------- layer 3 区分 3|45 -----------------------------
                result_3_45 = self.cla_3_45(image,verbose=False)[0].probs.data # [p3,p45]
                # [p01/2, p01/2, p2345*p2, p2345*p345*p3, p2345*p345*p45, p2345*p345*p45]
                P_bayes[3:4] = P_bayes[3:4]*result_3_45[0]
                P_bayes[4:6] = P_bayes[4:6]*result_3_45[1]
                if result_3_45[0] < result_3_45[1]: # 3 < 4,5
                    # ----------------------------- layer 4 区分 4|5 -----------------------------
                    result_4_5 = self.cla_4_5(image,verbose=False)[0].probs.data # [p4,p5]
                    # [p01/2, p01/2, p2345*p2, p2345*p345*p3, p2345*p345*p45*p4, p2345*p345*p45*p5]  final
                    P_bayes[4:6] = P_bayes[4:6]*result_4_5
                else: # 3 > 4,5 | 不需要再区分
                    # [p01/2, p01/2, p2345*p2, p2345*p345*p3, p2345*p345*p45/2, p2345*p345*p45/2] final 
                    P_bayes[4:6] = P_bayes[4:6]/2
                    pass
            else: # 2 > 3,4,5
                # [p01/2, p01/2, p2345*p2, p2345*p345/3, p2345*p345/3, p2345*p345/3] final
                P_bayes[3:6] = P_bayes[3:6]/3
                pass
        else: # 0,1 > 2,3,4,5
            # [p01, p01, p2345/4, p2345/4, p2345/4, p2345/4]
            P_bayes[2:6] = P_bayes[2:6]/4
            # ----------------------------- layer 2 区分 0,1|2 -----------------------------
            result_0_1 = self.cla_0_1(image,verbose=False)[0].probs.data # [p0,p1]
            # [p01*p0, p01*p1, p2345/4, p2345/4, p2345/4, p2345/4] final
            P_bayes[0] = P_bayes[0]*result_0_1[0]
            P_bayes[1] = P_bayes[1]*result_0_1[1]
            
        assert (torch.sum(P_bayes)-1).abs() < 1e-3 , "bayes概率和不为1"
        '''--------------------------------- ensemble ---------------------------------'''
        # 利用P_fea2cla_01_2345更新P_basic
        P_basic_updated = P_basic.clone()  #

        if P_fea2cla_01_2345[0] > 0.80:  # 如果特征模型强烈倾向于前两类(0,1)
            # 增强P_basic中前两类的概率
            P_basic_updated[0:2] = P_basic[0:2] * 1.2
            # 降低后四类的概率
            P_basic_updated[2:6] = P_basic[2:6] * 0.8
        elif P_fea2cla_01_2345[1] > 0.80:  # 如果特征模型强烈倾向于后四类(2,3,4,5)
            # 降低前两类的概率
            P_basic_updated[0:2] = P_basic[0:2] * 0.8
            # 增强后四类的概率
            P_basic_updated[2:6] = P_basic[2:6] * 1.2
        # 重新归一化P_basic_updated_2
        # P_basic_updated_2 = F.softmax(P_basic_updated_2, dim=0) # 效果变得很糟糕
        P_basic_updated = P_basic_updated / torch.sum(P_basic_updated)

        # ground_truth = self.get_ground_truth(image)
        result = self.majority_ensemble([P_basic_updated,torch.tensor(P_bayes)])
        return result
    
    @staticmethod
    def get_ground_truth(img_path):
        """根据图片路径返回真实标签"""
        img_id = int(os.path.basename(img_path).split('.')[0])
        if img_id <= 126:
            return int(0)
        elif img_id <= 364:
            return int(1)
        elif img_id <= 485:
            return int(2)
        elif img_id <= 566:
            return int(3)
        elif img_id <= 634:
            return int(4)
        else:
            return int(5)
            
    @torch.no_grad()
    @profile
    def fea_predict(self, image):
        info("fea_predict开始")
        info("fea读取图像")
        origin = cv2.imread(image)
        info("removeFrame")
        top,left,h,w = removeFrame(origin.transpose((2,0,1)))
        cropped = origin[top:top+h,left:left+w]
        heatmap = self.seg_model(self.transform(cropped).unsqueeze(0).to(get_device())).sigmoid() # 生成热图
        info("heatmap插值")
        heatmap = F.interpolate(heatmap,size=(h,w),mode='bilinear',align_corners=False).cpu()
        info("mask")
        mask = make_mask(heatmap)
        info("box")
        box = make_box_map(cropped,cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY))
        info("masked")
        masked = make_masked(heatmap,cropped)

        info("boundary_full")
        # boundary_full_result = self.boundary_full(origin,verbose=False)
        boundary_full = self.boundary_full(origin,verbose=False)[0].probs.data
        info("boundary_box")
        boundary_box = self.boundary_box(box,verbose=False)[0].probs.data
        info("ensemble boundary")
        boundary = Tester.max_ensemble([boundary_full,boundary_box])

        info("calcification_full")
        calcification_full = self.calcification_full(origin,verbose=False)[0].probs.data
        info("calcification_box")
        calcification_box = self.calcification_box(box,verbose=False)[0].probs.data
        info("calcification_masked")
        calcification_masked = self.calcification_masked(masked,verbose=False)[0].probs.data
        info("ensemble calcification")
        calcification = Tester.majority_ensemble([calcification_full,calcification_box,calcification_masked])

        info("direction_full")
        direction = self.direction_full(origin,verbose=False)[0].probs.top1

        info("shape_full")
        shape_full = self.shape_full(origin,verbose=False)[0].probs.data
        info("shape_box")
        shape_box = self.shape_box(box,verbose=False)[0].probs.data
        info("shape_masked")
        shape_masked = self.shape_masked(masked,verbose=False)[0].probs.data


        info("ensemble shape")
        shape = Tester.average_ensemble([shape_full,shape_box,shape_masked])
        
        return boundary,calcification,direction,shape
    
    @torch.no_grad()
    @profile
    def fea_predict_probs(self,image):
        boundary_full_result = self.boundary_full(image,verbose=False)
        direction_full_result = self.direction_full(image,verbose=False)
        calcification_full_result = self.calcification_full(image,verbose=False)
        shape_full_result = self.shape_full(image,verbose=False)
        
        #! 顺序 boundary, calcification, direction, shape 
        #! 注意：calcification 0 代表，1 代表有，而cla中“微钙化”才是恶化的条件
        # 形状不规则 shape 0 规则，1 不规则
        # 垂直生长 direction 0 水平，1 垂直
        # 边缘不光整 boundary 0 光整，1 不光整
        # 微钙化 calcification 0 有，1 无
        # 但是都当1才会有好的效果
        fea = [float(boundary_full_result[0].probs.data[1]),float(calcification_full_result[0].probs.data[1]),float(direction_full_result[0].probs.data[1]),float(shape_full_result[0].probs.data[1])]
        # if get_device() != torch.device('cuda'):
        #     fea = [float(i) for i in fea]
        return fea
        
    @staticmethod
    def average_ensemble(tensors):
        return torch.argmax(torch.mean(torch.stack(tensors,dim=0),dim=0)).item()

    @staticmethod
    def max_ensemble(tensors):
        return torch.argmax(torch.max(torch.stack(tensors,dim=0),dim=0)[0]).item()

    @staticmethod
    def majority_ensemble(tensors):
        return torch.argmax(torch.sum(torch.stack(tensors,dim=0)>0.5,dim=0),dim=0).item()


