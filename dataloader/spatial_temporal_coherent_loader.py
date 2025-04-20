import os
import torch
import random
import torchvision
import pickle
import cv2
import json
import time
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F

import sys
sys.path.append("/data2/JM/code/STCCL")

class TemporalDataloader(Dataset):
    def __init__(self, mode="train") -> None:
        super(TemporalDataloader).__init__()
        self.size = 224
        self.root = "/data3/JM/data/MEAD"
        with open("/data2/JM/code/STCCL/files/aligned_path36.json", "r") as json_file:
            self.data_file = json.load(json_file)
        if mode == "train":
            self.vid_list = sorted(list(self.data_file.keys()))
            self.vid_list.remove("W021")
        else:
            self.vid_list = ["W021"]
        
        self._img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((self.size, self.size)),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.num_img = self.calculate_length()
        print(f"{mode} 数据集大小为： {self.num_img}")
        
        
    def calculate_length(self):

        len_pair = 0
        for vid in self.vid_list:
            emotion = self.data_file[vid].keys()
            for em in emotion:
                vid_sub = sorted(self.data_file[vid][em])
                for vid_sub_sub in vid_sub:
                    len_pair += len(self.data_file[vid][em][vid_sub_sub][0])
        return len_pair
            
    
    def __len__(self):
        return self.num_img
    
    def __getitem__(self, index, n=2):
        ## 获取文件目录如 “M003”
        vid = random.choice(self.vid_list)
        ## 获取情绪标签 ['angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
        emotion_list = list(self.data_file[vid].keys())
        ## 随机选择一种情绪
        emn = random.choice(emotion_list)
        ## 获取对应vid 和 emn下的pair目录
        vid_sub = list(self.data_file[vid][emn].keys())
        ## 随机获取pair目录如 “001_001”
        sub = random.choice(vid_sub)
        ## 获取该目录下的文件索引
        vid_len = len(self.data_file[vid][emn][sub][0])
        assert len(self.data_file[vid][emn][sub][0]) == len(self.data_file[vid][emn][sub][1])
        ## 随机选取一个文件
        idx = random.choice(range(n, vid_len-n-1))
        ## 索引pair数据对
        source_vid = sub.split("_")[0]
        target_vid = sub.split("_")[1]

        ## 获取连续帧索引
        source_img_idx = self.data_file[vid][emn][sub][0][idx-n : idx+n+1]
        target_img_idx = self.data_file[vid][emn][sub][1][idx-n : idx+n+1]

        source_imgs = None
        target_imgs = None

        for idx_source, idx_target in zip(source_img_idx, target_img_idx):
            source_path = os.path.join(self.root, vid, "align_img", "neutral", source_vid, str(idx_source).zfill(6)+".jpg")
            target_path = os.path.join(self.root, vid, "align_img", emn, target_vid, str(idx_target).zfill(6)+".jpg")
            
            source = cv2.imread(source_path)
            target = cv2.imread(target_path)
            
            source = self._img_transform(source)
            target = self._img_transform(target)
            
            if source_imgs is None:
                source_imgs = source.unsqueeze(0) 
                target_imgs = target.unsqueeze(0)
            else:
                source_imgs = torch.cat((source_imgs, source.unsqueeze(0)), dim=0)
                target_imgs = torch.cat((target_imgs, target.unsqueeze(0)), dim=0)
        return {"source_img": source_imgs,
                "target_img": target_imgs}
