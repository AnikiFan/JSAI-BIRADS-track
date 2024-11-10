from glob import glob
import os
from myBenchmark import benchmark
task = ["cla","boundary","calcification","direction","shape"]
data = [
    ["full","full_2_3","full_23_4A4B4C5","full_4A_4B4C5","full_4B_4C5","full_4C_5"],
    # ["full","box"],
    # ["full","box","masked"],
    # ["full"],
    # ["full","box","masked"]
]
def test(task,data):
    model = os.path.join(os.curdir,"project","model",task,data+".pt")
    benchmark(model,data=task+"_"+data+".yaml",imgsz=224,half=False,int8=False,device="cuda:0")
    benchmark(model,data=task+"_"+data+".yaml",imgsz=224,half=True,int8=False,device="cuda:0")

if __name__ == '__main__':
    for t,d in zip(task,data):
        for dd in d:
            print(t,dd)
            test(t,dd)