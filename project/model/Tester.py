from torchvision import transforms
import torch.nn.functional as F
import cv2
from .utils import removeFrame,make_mask,make_box_map,make_masked,Preprocess
import torch
import os
from ultralytics import YOLO
from logging import info
TASK = "classify"
YOLO_PARAMS = {"imgsz":224,"half":True,"int8":False,"device":"cuda:0","verbose":False}
class Tester:
    def __init__(self,model_folder_path,format='pt',half=True):
        info("init test")
        self.seg_model = torch.jit.load(os.path.join(model_folder_path,'segmentation','FCB_half.torchscript')).to("cuda")
        # 自定义图像转换
        self.transform = transforms.Compose([
            transforms.Resize((352,352),antialias=True),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        self.seg_model.eval()
        self.preprocess = Preprocess(half)
        # 实例化cla所需模型
        self.cla_full = YOLO(os.path.join(model_folder_path,'cla','full.'+format),task=TASK)
        self.cla_full(**YOLO_PARAMS)
        self.cla_0_1 = YOLO(os.path.join(model_folder_path,'cla','full_2_3.'+format),task=TASK)
        self.cla_01_2345 = YOLO(os.path.join(model_folder_path,'cla','full_23_4A4B4C5.'+format),task=TASK)
        self.cla_2_345 = YOLO(os.path.join(model_folder_path,'cla','full_4A_4B4C5.'+format),task=TASK)
        self.cla_3_45 = YOLO(os.path.join(model_folder_path,'cla','full_4B_4C5.'+format),task=TASK)
        self.cla_4_5 = YOLO(os.path.join(model_folder_path,'cla','full_4C_5.'+format),task=TASK)
        # 实例化boudary特征所需模型
        self.boundary_full = YOLO(os.path.join(model_folder_path,'boundary','full.'+format),task=TASK)
        self.boundary_box = YOLO(os.path.join(model_folder_path,'boundary','box.'+format),task=TASK)

        # 实例化calcification特征所需模型
        self.calcification_full = YOLO(os.path.join(model_folder_path,'calcification','full.'+format),task=TASK)
        self.calcification_box = YOLO(os.path.join(model_folder_path,'calcification','box.'+format),task=TASK)
        self.calcification_masked = YOLO(os.path.join(model_folder_path,'calcification','masked.'+format),task=TASK)

        # 实例化direction特征所需模型
        self.direction_full = YOLO(os.path.join(model_folder_path,'direction','full.'+format),task=TASK)

        # 实例化shape特征所需模型
        self.shape_full = YOLO(os.path.join(model_folder_path,'shape','full.'+format),task=TASK)
        self.shape_box = YOLO(os.path.join(model_folder_path,'shape','box.'+format),task=TASK)
        self.shape_masked = YOLO(os.path.join(model_folder_path,'shape','masked.'+format),task=TASK)
        info("finish init")


    @torch.no_grad()
    def cla_predict(self,image):
        origin = cv2.imread(image)
        origin = self.cla_full.predictor.preprocess([origin])
        cla_full = self.cla_full(origin,**YOLO_PARAMS)[0].probs.data
        cla_01_2345__01,cla_01_2345__2345 = self.cla_01_2345(origin,**YOLO_PARAMS)[0].probs.data
        cla_0_1__0,cla_0_1__1 = self.cla_0_1(origin,**YOLO_PARAMS)[0].probs.data
        cla_2_345__2,cla_2_345__345 = self.cla_2_345(origin,**YOLO_PARAMS)[0].probs.data
        cla_3_45__3,cla_3_45__45 = self.cla_3_45(origin,**YOLO_PARAMS)[0].probs.data
        cla_4_5__4,cla_4_5__5 = self.cla_4_5(origin,**YOLO_PARAMS)[0].probs.data
        prob_result = torch.Tensor([
        cla_01_2345__01 * cla_0_1__0,
        cla_01_2345__01 * cla_0_1__1,
        cla_01_2345__2345 * cla_2_345__2,
        cla_01_2345__2345 * cla_2_345__345 * cla_3_45__3,
        cla_01_2345__2345 * cla_2_345__345 * cla_3_45__45 * cla_4_5__4,
        cla_01_2345__2345 * cla_2_345__345 * cla_3_45__45 * cla_4_5__5,
        ]).to('cuda')
        result = Tester.majority_ensemble([cla_full,prob_result])
        return result


    @torch.no_grad()
    def fea_predict(self, image):
        origin = torch.tensor(cv2.imread(image),device='cuda').permute(2,0,1)
        top,left,h,w = removeFrame(origin)
        cropped = origin[:,top:top+h,left:left+w]
        heatmap = self.seg_model(self.transform(cropped/255).half().unsqueeze(0)).sigmoid() # 生成热图
        heatmap = F.interpolate(heatmap,size=(h,w),mode='bilinear',align_corners=False)
        mask = make_mask(heatmap)
        box = make_box_map(cropped,mask)
        masked = make_masked(heatmap,cropped)
        origin = self.preprocess(origin)
        box = self.preprocess(box)
        masked = self.preprocess(masked)
        boundary_full = self.boundary_full(origin,**YOLO_PARAMS)[0].probs.data
        boundary_box = self.boundary_box(box,**YOLO_PARAMS)[0].probs.data
        boundary = Tester.max_ensemble([boundary_full,boundary_box])
        calcification_full = self.calcification_full(origin,**YOLO_PARAMS)[0].probs.data
        calcification_box = self.calcification_box(box,**YOLO_PARAMS)[0].probs.data
        calcification_masked = self.calcification_masked(masked,**YOLO_PARAMS)[0].probs.data
        calcification = Tester.majority_ensemble([calcification_full,calcification_box,calcification_masked])
        direction = self.direction_full(origin,**YOLO_PARAMS)[0].probs.top1
        shape_full = self.shape_full(origin,**YOLO_PARAMS)[0].probs.data
        shape_box = self.shape_box(box,**YOLO_PARAMS)[0].probs.data
        shape_masked = self.shape_masked(masked,**YOLO_PARAMS)[0].probs.data
        shape = Tester.average_ensemble([shape_full,shape_box,shape_masked])
        return boundary,calcification,direction,shape
        
    @staticmethod
    def average_ensemble(tensors):
        return torch.argmax(torch.mean(torch.stack(tensors,dim=0),dim=0)).item()

    @staticmethod
    def max_ensemble(tensors):
        return torch.argmax(torch.max(torch.stack(tensors,dim=0),dim=0)[0]).item()

    @staticmethod
    def majority_ensemble(tensors):
        return torch.argmax(torch.sum(torch.stack(tensors,dim=0)>0.5,dim=0),dim=0).item()


