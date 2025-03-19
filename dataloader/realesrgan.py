import io
import os
import av
import numpy as np
import cv2
import glob
import math
import yaml
import random
from collections import OrderedDict
import torch
import torch.nn.functional as F

from basicsr.data.transforms import augment
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import DiffJPEG, USMSharp, img2tensor, tensor2img
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize, rgb_to_grayscale)




def apply_random_video_compression(imgs):
    """This is the function to apply random video compression on images.
        input [b,c,h,w] rgb 0,1 tensor
        need [h,w,c] * b rgb 0,255 uint8 

        output [h,w,c] * b rgb 0,1 float32
    """
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    imgs = (imgs * 255).astype(np.uint8) 


    def pad_to_divisible_by_2(img):
        h, w = img.shape[:2]
        new_h = h if h % 2 == 0 else h + 1
        new_w = w if w % 2 == 0 else w + 1
        padded_img = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)
        padded_img[:h, :w, :] = img
        return padded_img, h, w

    padded_imgs = []
    original_sizes = []
    for img in imgs:
        padded_img, orig_h, orig_w = pad_to_divisible_by_2(img)
        padded_imgs.append(padded_img)
        original_sizes.append((orig_h, orig_w))

    codec_type = ['libx264', 'h264', 'mpeg4']
    codec_prob = [0.3333, 0.3333, 0.3334]
    bitrate_choice = [1e4, 1e5]
    codec = random.choices(codec_type, codec_prob)[0]
    bitrate = bitrate_choice
    bitrate = np.random.randint(bitrate[0], bitrate[1] + 1)
    buf = io.BytesIO()
    with av.open(buf, 'w', 'mp4') as container:
        stream = container.add_stream(codec, rate=1)
        stream.height = padded_imgs[0].shape[0]
        stream.width = padded_imgs[0].shape[1]
        stream.pix_fmt = 'yuv420p'
        stream.bit_rate = bitrate
        for img in padded_imgs:
            img = img.astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            frame.pict_type = 'NONE'
            for packet in stream.encode(frame):
                container.mux(packet)
        # Flush stream
        for packet in stream.encode():
            container.mux(packet)

    outputs = []
    with av.open(buf, 'r', 'mp4') as container:
        if container.streams.video:
            for frame in container.decode(**{'video': 0}):
                outputs.append(frame.to_rgb().to_ndarray().astype(
                    np.float32) / 255.0)


    cropped_outputs = []
    for output, (orig_h, orig_w) in zip(outputs, original_sizes):
        cropped_outputs.append(output[:orig_h, :orig_w, :])

    return cropped_outputs



def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def opt_parse(opt_path):
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)  # ignore_security_alert_wait_for_fix RCE

    return opt

class RealESRGAN_VSR_degradation(object):
    def __init__(self, opt_name='xxx/params_realesrgan.yml', device='cpu'):
        opt_path = opt_name
        self.opt = opt_parse(opt_path)
        self.device = device #torch.device('cpu')
        optk = self.opt['kernel_info']       
        # blur settings for the first degradation
        self.blur_kernel_size = optk['blur_kernel_size']
        self.kernel_list = optk['kernel_list']
        self.kernel_prob = optk['kernel_prob']
        self.blur_sigma = optk['blur_sigma']
        self.betag_range = optk['betag_range']
        self.betap_range = optk['betap_range']
        self.sinc_prob = optk['sinc_prob']

        # blur settings for the second degradation
        self.blur_kernel_size2 = optk['blur_kernel_size2']
        self.kernel_list2 = optk['kernel_list2']
        self.kernel_prob2 = optk['kernel_prob2']
        self.blur_sigma2 = optk['blur_sigma2']
        self.betag_range2 = optk['betag_range2']
        self.betap_range2 = optk['betap_range2']
        self.sinc_prob2 = optk['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = optk['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        self.usm_shaper = USMSharp().to(self.device)
    
    def color_jitter_pt(self, img, brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img



    def random_kernels(self):
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range,
                    self.betap_range,
                    noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2) 

        return kernel, kernel2, sinc_kernel

    @torch.no_grad()
    def degrade_process(self, img_gts, resize_bak=False):

        #  [0,1] bgr to rgb,
        img_gts = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for img in img_gts]

        # hwc->chw 
        img_gts_combine = torch.stack([torch.from_numpy(img) for img in img_gts]).permute(0, 3, 1, 2).float()  # [t, h, w, c] -> [t, c, w, h]

        kernel1, kernel2, sinc_kernel = self.random_kernels()
        img_gts_combine, kernel1, kernel2, sinc_kernel = img_gts_combine.to(self.device), kernel1.to(self.device), kernel2.to(self.device), sinc_kernel.to(self.device)
        #img_gts = self.usm_shaper(img_gts) # shaper gt
        ori_h, ori_w = img_gts_combine.size()[2:4]

        #scale_final = random.randint(4, 16)
        scale_final = 4

        # ----------------------- The first degradation process ----------------------- #
        # blur
        # print(img_gts_combine.shape)
        out = filter2D(img_gts_combine, kernel1)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # noise
        gray_noise_prob = self.opt['gray_noise_prob']
        if np.random.uniform() < self.opt['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.opt['second_blur_prob']:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / scale_final * scale), int(ori_w / scale_final * scale)), mode=mode)
        # noise
        gray_noise_prob = self.opt['gray_noise_prob2']
        if np.random.uniform() < self.opt['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=mode)
            out = filter2D(out, sinc_kernel)

        if np.random.uniform() < self.opt['gray_prob']:
            out = rgb_to_grayscale(out, num_output_channels=1)

        if np.random.uniform() < self.opt['color_jitter_prob']:
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            out = self.color_jitter_pt(out, brightness, contrast, saturation, hue)

        # clamp and round
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        # print(out.shape)

        # apply video compression
        # img_lqs = apply_random_video_compression(out)
        img_lqs = []
        # # hwc chw
        for k in range(len(img_gts)):
            # has been rgb
            img_gts[k] = img2tensor(img_gts[k], bgr2rgb = False)
            img_lqs.append(out[k])

        return img_gts, img_lqs


