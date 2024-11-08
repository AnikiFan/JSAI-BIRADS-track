from glob import glob
import os
from myBenchmark import benchmark
task = ["cla","boundary","calcification","direction","shape"]
data = [
    [],
    ["full","box"],
    ["full","box","masked"],    
    ["full"],
    ["full","box","masked"]
]
def test(task,data):
    model = os.path.join(os.curdir,"project","model",task,data+".pt")
    benchmark(model,data=task+"_"+data+".yaml",imgsz=224,half=False,int8=False,device="cuda:0")
    benchmark(model,data=task+"_"+data+".yaml",imgsz=224,half=True,int8=False,device="cuda:0")

for t,d in zip(task,data):
    for dd in d:
        test(t,dd)