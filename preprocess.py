from ultralytics import YOLO
import os
from glob import glob
import cv2
import numpy as np
if __name__ == "__main__":
    model = YOLO(os.path.join(os.curdir,'project','model','boundary','box.pt'),task='classify')
    rt_model = YOLO(os.path.join(os.curdir,'project',"linux",'model','boundary','box.engine'),task='classify')
    model()
    img = cv2.imread(os.path.join(os.curdir,'testB','cla','1.jpg'))
    transform = model.predictor.preprocess
    processed = transform([img])