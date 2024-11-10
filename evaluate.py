import pandas as pd
import torch
import os
from torcheval.metrics.functional import multiclass_accuracy,multiclass_f1_score,binary_f1_score,multiclass_confusion_matrix
cla_pre = pd.read_csv(os.path.join(os.curdir,'project','cla_pre.csv'))
fea_pre = pd.read_csv(os.path.join(os.curdir,'project','fea_pre.csv'))
cla_gt = pd.read_csv(os.path.join(os.curdir,'cla_gt.csv'))
fea_gt = pd.read_csv(os.path.join(os.curdir,'fea_gt.csv'))
cla_pre = torch.Tensor(cla_pre.label).to(torch.int64)
cla_gt = torch.Tensor(cla_gt.label).to(torch.int64)
cla_acc = multiclass_accuracy(cla_pre,cla_gt,num_classes=6,average='micro').item()
cla_f1 = multiclass_f1_score(cla_pre,cla_gt,num_classes=6,average='macro').item()
cla_overall = cla_acc*0.6+cla_f1*0.4
print("cla_overall",cla_overall)
features = ['boundary','calcification','direction','shape']
acc,f1 = 0,0
for feature in features:
    pre = torch.Tensor(fea_pre[feature]).to(torch.int64)
    gt = torch.Tensor(fea_gt[feature]).to(torch.int64)
    acc += multiclass_accuracy(pre,gt,num_classes=2,average='micro').item()/4
    f1 += binary_f1_score(pre,gt).item()/4
    print(multiclass_confusion_matrix(pre,gt,num_classes=2))
print("feaoverall",acc*0.6+f1*0.4)
print("overall",cla_overall*0.5+(acc*0.6+f1*0.4)*0.5)