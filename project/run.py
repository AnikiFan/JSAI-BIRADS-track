import os
import pandas as pd
from model.Tester import Tester
from glob import glob
import logging
from tqdm import tqdm
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    tqdm.pandas()

    # 定义路径
    data_folder_path = os.path.join(os.pardir,'testB') # testB数据集所在路径
    cla_folder_path = os.path.join(data_folder_path,'cla') # cla数据集所在路径
    fea_folder_path = os.path.join(data_folder_path,'fea') # fea数据集所在路径
    model_folder_path = os.path.join(os.curdir,'model') # model文件夹所在路径

    # 收集待测试图像路径，文件名按字典序排序
    cla_files = glob(os.path.join(cla_folder_path,'*')) # 收集cla数据集文件夹下的所有文件
    cla_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0])) # 按照字典序将文件名排序
    cla_files = pd.DataFrame({"img":cla_files})

    # 收集待测试图像路径，文件名按字典序排序
    fea_files = glob(os.path.join(fea_folder_path,'*')) # 收集fea数据集文件夹下的所有文件
    fea_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0])) # 按照字典序将文件名排序
    fea_files = pd.DataFrame({"img":fea_files})

    # 生成待填写的pre DataFrame
    cla_pre = pd.DataFrame({"id":range(1,len(cla_files)+1)})
    fea_pre = pd.DataFrame({"id":range(1,len(fea_files)+1)})

    # 实例化用于测试的类
    tester = Tester(model_folder_path,format='engine')

    # 对cla数据集进行推理
    cla_pre['label'] = cla_files.img.progress_apply(tester.cla_predict)

    # 对fea数据集进行推理
    fea_pre[['boundary','calcification','direction','shape']] = fea_files.img.progress_apply(tester.fea_predict).to_list()

    # 导出pre csv文件
    cla_pre.to_csv("cla_pre.csv",index=False)
    fea_pre.to_csv("fea_pre.csv",index=False)


