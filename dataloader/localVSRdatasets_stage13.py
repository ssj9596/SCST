import os
import glob
# from turtle import position
import cv2
import pandas as pd
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils import data as data

import torch.nn.functional as F
import torch
import os
import random
import tarfile
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import io
from basicsr.utils.img_util import imfrombytes
from dataloader.realbasicvsr import RealBasicVSR_degradation
# from dataloader.realesrgan import RealESRGAN_VSR_degradation
from basicsr.utils import  img2tensor

class Bicubic_degradation:
    def degrade_process(self, img_gts):
     
        img_gts = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for img in img_gts]
        img_lqs = []
        for k in range(len(img_gts)):
            # has been rgb
            img_gts[k] = img2tensor(img_gts[k], bgr2rgb = False)

            img_lqs.append(F.interpolate(img_gts[k].unsqueeze(0), scale_factor=0.25, mode='bicubic', align_corners=False).squeeze(0))

        return img_gts, img_lqs
    

class LocalVideoDataset(Dataset):
    def __init__(
            self, 
            meta_path, 
            hr_root,
            mode="all",
            image_size=512,
            tokenizer=None,
            interval_list=[1],
            num_frame=5,
            degradation=[RealBasicVSR_degradation()],
            resize_bank=False,
            crop_size=None,
            caption_path=None,
            null_txt_ratio=0.5,
        ):
        super(LocalVideoDataset, self).__init__()
        self.tokenizer = tokenizer
        self.meta_path = meta_path
        self.hr_root = hr_root
        self.mode = mode
        self.interval_list = interval_list
        self.num_frame = num_frame
        self.resize_bank = resize_bank
        self.crop_size = crop_size 
        self.img_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.degradations = degradation
        self.keys = []
        self.frame_dict = {}
        self.captions = {}
        self.null_txt_ratio = null_txt_ratio
        
        if caption_path:
            caption_df = pd.read_csv(caption_path)
            for _, row in caption_df.iterrows():
                clip_name = os.path.basename(row['path'])[:-4]
                self.captions[clip_name] = row['prompt']

        with open(meta_path, 'r') as fin:
            for line in fin:
                folder, frame_num = line.strip().split(' ')
                frame_num = int(frame_num)
                self.frame_dict[folder] = frame_num
                # if self.mode == "xxx":
       
                if self.mode == "all":
                    self.keys.extend([f'{folder}/{i:08d}' for i in range(frame_num)])
                elif self.mode == "skip":
                    position = int(folder[-3:])
                    if position in [1, 4, 9, 10, 11, 12]:
                        continue
                    self.keys.extend([f'{folder}/{i:08d}' for i in range(0, frame_num, num_frame)])
                else:
                    self.keys.extend([f'{folder}/{i:08d}' for i in range(0, frame_num, num_frame)])

    def tokenize_caption(self, caption):
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def _get_image_from_tar(self, tar_path, image_name):
  
        with tarfile.open(tar_path, 'r') as tar:
            member = tar.getmember(image_name)
          
            f = tar.extractfile(member)
            img = imfrombytes(f.read(), float32=False)
           
            return img

    def __getitem__(self, index):
        try:
            example = dict()
            key = self.keys[index]
            clip_name, frame_name = key.split('/')
            interval = random.choice(self.interval_list)
            start_frame_idx = int(frame_name)
            if start_frame_idx > self.frame_dict[clip_name] - self.num_frame * interval:
                start_frame_idx = random.randint(0, self.frame_dict[clip_name] - self.num_frame * interval)
            end_frame_idx = start_frame_idx + self.num_frame * interval

            neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))
            img_gts = []

            input_ids = []
            tar_hr_path = f"{self.hr_root}/{clip_name}.tar"
            for neighbor in neighbor_list:
                img_hr_name = f'{clip_name}/{neighbor:08d}.png'
                img_hr = self._get_image_from_tar(tar_hr_path, img_hr_name)
                img_gts.append(img_hr)

            if self.tokenizer is not None:
                caption = self.captions.get(
                            clip_name, "") if random.random() > self.null_txt_ratio else ""
                input_ids.append(self.tokenize_caption(caption=caption).repeat(self.num_frame, 1))

    
            degradation = random.choice(self.degradations)

                

            img_gts, img_lqs = degradation.degrade_process(img_gts)

            img_gts = [torch.clamp(img_hr * 2.0 - 1.0, min=-1.0, max=1.0) for img_hr in img_gts]

            if self.resize_bank:
                mode = random.choice(['bilinear', 'bicubic'])
                img_lqs = [torch.clamp(F.interpolate(img_lr.unsqueeze(0),
                    scale_factor=4, mode=mode, align_corners=False).squeeze(0), min=0.0, max=1.0) for img_lr in img_lqs]

            
            if self.crop_size:
                # random crop
                h, w = img_gts[0].shape[-2:]
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)
                img_gts = [img[:, top:top + self.crop_size, left:left + self.crop_size] for img in img_gts]
                img_lqs = [img[:, top:top + self.crop_size, left:left + self.crop_size] for img in img_lqs]


            example["pixel_values"] = torch.stack(img_gts)
            example["conditioning_pixel_values"] = torch.stack(img_lqs)

            if self.tokenizer is not None:
                example["input_ids"] = torch.cat(input_ids, dim=0) 

            return example
        except Exception as e:
            print(f"Error in __getitem__ at index {index}: {e}")

            random_index = random.randint(0, len(self.keys) - 1)
            return self.__getitem__(random_index)

    def __len__(self):
        return len(self.keys)