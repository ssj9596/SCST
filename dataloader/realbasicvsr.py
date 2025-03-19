# from MGLD
from multiprocessing import Pool, cpu_count
import random
import shutil
import sys
import os
import tarfile
import torch.nn.functional as F

import torch
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# print(sys.path)
from copy import deepcopy
from pathlib import Path
from cv2 import imwrite
from torch.utils import data as data
from tqdm.auto import tqdm

from basicsr.data.mmcv_transforms import Clip, UnsharpMasking, RescaleToZeroOne
from basicsr.data.mmcv_transforms import RandomBlur, RandomResize, RandomNoise, RandomJPEGCompression, RandomVideoCompression
from basicsr.utils import FileClient, imfrombytes, img2tensor,tensor2img,imwrite



# @DATASET_REGISTRY.register()
class RealBasicVSR_degradation(object):
    def __init__(self, opt_name='/maindata/data/shared/public/aigame/chengxiu/VSR_ALL/SCST/dataloader/params_realbasicvsr.yml'):
        super(RealBasicVSR_degradation, self).__init__()

        with open(opt_name, 'r') as f:
            config = yaml.safe_load(f)
        opt = config['params']

        # the first degradation
        self.random_blur_1 = RandomBlur(
            params=opt['degradation_1']['random_blur']['params'],
            keys=opt['degradation_1']['random_blur']['keys']
        )
        self.random_resize_1 = RandomResize(
            params=opt['degradation_1']['random_resize']['params'],
            keys=opt['degradation_1']['random_resize']['keys']
        )
        self.random_noise_1 = RandomNoise(
            params=opt['degradation_1']['random_noise']['params'],
            keys=opt['degradation_1']['random_noise']['keys']
        )
    
        self.random_jpeg_1 = RandomJPEGCompression(
            params=opt['degradation_1']['random_jpeg']['params'],
            keys=opt['degradation_1']['random_jpeg']['keys']
        )
        self.random_mpeg_1 = RandomVideoCompression(
            params=opt['degradation_1']['random_mpeg']['params'],
            keys=opt['degradation_1']['random_mpeg']['keys']
        )

        # the second degradation
        self.random_blur_2 = RandomBlur(
            params=opt['degradation_2']['random_blur']['params'],
            keys=opt['degradation_2']['random_blur']['keys']
        )
        self.random_resize_2 = RandomResize(
            params=opt['degradation_2']['random_resize']['params'],
            keys=opt['degradation_2']['random_resize']['keys']
        )
        self.random_noise_2 = RandomNoise(
            params=opt['degradation_2']['random_noise']['params'],
            keys=opt['degradation_2']['random_noise']['keys']
        )
        self.random_jpeg_2 = RandomJPEGCompression(
            params=opt['degradation_2']['random_jpeg']['params'],
            keys=opt['degradation_2']['random_jpeg']['keys']
        )
        self.random_mpeg_2 = RandomVideoCompression(
            params=opt['degradation_2']['random_mpeg']['params'],
            keys=opt['degradation_2']['random_mpeg']['keys']
        )

        # final
        self.resize_final = RandomResize(
            params=opt['degradation_2']['resize_final']['params'],
            keys=opt['degradation_2']['resize_final']['keys']
        )
        self.blur_final = RandomBlur(
            params=opt['degradation_2']['blur_final']['params'],
            keys=opt['degradation_2']['blur_final']['keys']
        )

        # transforms
        self.usm = UnsharpMasking(
            kernel_size=opt['transforms']['usm']['kernel_size'],
            sigma=opt['transforms']['usm']['sigma'],
            weight=opt['transforms']['usm']['weight'],
            threshold=opt['transforms']['usm']['threshold'],
            keys=opt['transforms']['usm']['keys']
        )
        self.clip = Clip(keys=opt['transforms']['clip']['keys'])
        self.rescale = RescaleToZeroOne(keys=opt['transforms']['rescale']['keys'])

    @torch.no_grad()
    def degrade_process(self, img_gts, resize_bak=False):
        # img_gts: [h,w,c] * t   [0,255]

        # no augment， should augment before
        img_lqs = deepcopy(img_gts) # t, h, w，c
        out_dict = {'lqs': img_lqs, 'gts': img_gts}

        # first
        out_dict = self.usm.transform(out_dict)

        ## the first degradation
        out_dict = self.random_blur_1(out_dict)
        out_dict = self.random_resize_1(out_dict)
        out_dict = self.random_noise_1(out_dict)
        # bgr->rgb  float32->int8
        out_dict = self.random_jpeg_1(out_dict)
        # int8->float32
        out_dict = self.random_mpeg_1(out_dict)

        ## the second degradation
        out_dict = self.random_blur_2(out_dict)
        out_dict = self.random_resize_2(out_dict)
        out_dict = self.random_noise_2(out_dict)
        out_dict = self.random_jpeg_2(out_dict)
        out_dict = self.random_mpeg_2(out_dict)

        ## final resize
        out_dict = self.resize_final(out_dict)
        out_dict = self.blur_final(out_dict)

        # post process
        out_dict = self.clip(out_dict)

        # rescale  [0,1]
        out_dict = self.rescale.transform(out_dict)

        # # list-to-list
        # hwc chw
        for k in out_dict.keys():
            out_dict[k] = img2tensor(out_dict[k])

        img_gts = out_dict['gts']
        img_lqs = out_dict['lqs']
        
        return img_gts, img_lqs
    
  