#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import argparse
from collections import OrderedDict
import copy
import gc
import logging
import math
import os
import random
from pathlib import Path
import subprocess
import sys
import time
import accelerate
from einops import rearrange
import numpy as np
from regex import P
from sympy import use
import torch
import torch.nn.functional as F
import torch.nn as nn

import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
#from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms, utils
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
# from basicsr.archs.basicvsr_arch import BasicVSR

from dataloader.localVSRdatasets_stage2 import LocalVideoDataset
from dataloader.realbasicvsr import RealBasicVSR_degradation
from dataloader.realesrgan import RealESRGAN_VSR_degradation
from torch.utils.data import ConcatDataset


from models.controlnet.controlnet import ControlNetModel,InfoNCE_loss
from omegaconf import OmegaConf

# from taming.modules.losses.lpips import LPIPS

from safetensors.torch import load_file

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")
logger = get_logger(__name__)




def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate2",
        type=float,
        default=-100.0,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )

    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        required=True,
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
    "--not_use_controlnet",
    action="store_true",
    help="no add."
    )
    
    parser.add_argument('--num_frame', type=int, default=5)

    parser.add_argument('--trainable_modules', type=str, default="")
    parser.add_argument('--degradation', type=str, default='basicvsr')
    parser.add_argument('--controlnet_unet', action="store_true",)
    parser.add_argument('--only_unet', action="store_true",)
    parser.add_argument('--resume_path', type=str, default=None)

    parser.add_argument('--train_high_quality', action="store_true")

    parser.add_argument('--mix_train', action="store_true")
    parser.add_argument('--mix_setting', type=int, default=None)
    parser.add_argument('--contrastive_loss', type=str, default=None)
    parser.add_argument('--use_label', action="store_true")

    parser.add_argument('--fix_high', action="store_true")


    parser.add_argument('--from_start', action="store_true")

    parser.add_argument('--use_temporal', action="store_true")
    parser.add_argument('--temporal_config', type=str, default=None)
    parser.add_argument('--cl_mode', type=int, default=3,)
    parser.add_argument('--cl_weight', type=float, default=1.0)


    parser.add_argument('--only_controlnet', action="store_true")
    parser.add_argument('--restart_controlnet', action="store_true")
    
    parser.add_argument('--use_temporal_block',action="store_true",)

    parser.add_argument('--crop_size',type=int,default=512)
    parser.add_argument('--unet_config_path',type=str,default=None,)
    parser.add_argument('--no_controlnet', action="store_true",)

    parser.add_argument('--high_ratio', type=float, default=0.3)

    parser.add_argument('--contrastive_loss_only_lq', action="store_true",)
    parser.add_argument('--use_caption', action="store_true")
    parser.add_argument('--resnet_time_scale_shift',required=True,choices=["default","scale_shift"])

    parser.add_argument('--use_degradation_estimate', action="store_true")
 
    parser.add_argument('--normal_controlcond', action="store_true",)
    parser.add_argument('--return_controlcond', action="store_true",)

    parser.add_argument('--controlnet_use_projection_block', action="store_true",)
    parser.add_argument('--projection_block_one_layer', action="store_true")

    parser.add_argument('--moco', action="store_true",)
    parser.add_argument('--momentum', type=float, default = 0.999)
    parser.add_argument('--neg_feature_size', type=int, default = 1024)
    parser.add_argument('--temperature', type=float, default = 0.07,)
    parser.add_argument('--simsiam', action="store_true")
    parser.add_argument('--use_projection_controlnet', action="store_true")

    parser.add_argument('--linear_constant_ratio', action="store_true")
    parser.add_argument('--max_high_ratio',  type=float, default=0.9)
    parser.add_argument('--min_high_ratio',  type=float, default=0.1)
    parser.add_argument('--high_ratio_steps',  type=int, default=22500)

    # ssl_setting2, when input is hq, unet part will be train
    parser.add_argument('--ssl_setting1',action="store_true")
    parser.add_argument('--ssl_setting2',action="store_true")
    parser.add_argument('--use_diffloss',action="store_true")
    
    parser.add_argument('--overwrite',action="store_true")


    if input_args is not None:
        args = parser.parse_args(input_args)
        print(input_args)
    else:
        args = parser.parse_args()

    return args



def get_contrastive_loss(loss_type,feat1,feat2,cl_mode,cl_weight,neg_feat=None,temperature=0.07):
    if loss_type == "infonce":
        contrastive_loss = InfoNCE_loss(feat1, feat2, neg_feat=neg_feat,mode=cl_mode,temperature=temperature) * cl_weight

    else:
        raise ValueError("contrastive_loss must be clip or MSE")  
    return contrastive_loss


def log_grad_norms(model, top_k=5, make_print=False):
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append((name, grad_norm))
    
    grad_norms.sort(key=lambda x: x[1], reverse=True)
    avg_norm = sum(norm for _, norm in grad_norms) / len(grad_norms)

    if make_print:
        print(f"Top {top_k} largest gradient norms:")
        for name, norm in grad_norms[:top_k]:
            print(f"{name}: {norm:.4f}")
        
        
        print(f"Average gradient norm: {avg_norm:.4f}")
        print("Sum of all gradients: ", sum(norm for _, norm in grad_norms))
    return avg_norm

def calculate_high_ratio(current_step, args):
    max_steps, max_value, min_value = args.high_ratio_steps, args.max_high_ratio, args.min_high_ratio
    if current_step >= max_steps:
        return min_value
    else:
        return max_value - (max_value - min_value) * (current_step / max_steps)
    
def save_moco_checkpoint(step, path, moco_controlnet,):
    ckpt_path = os.path.join(path,"moco_ckpt.pth")
    state = {
        'model_state': {k: v.cpu() for k, v in moco_controlnet.state_dict().items()},
    }
    torch.save(state, ckpt_path)
    print(f"Checkpoint saved at {path}")

def load_moco_checkpoint(path, moco_controlnet, device,dtype):
    ckpt_path = os.path.join(path,"moco_ckpt.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)
    moco_controlnet.load_state_dict(checkpoint['model_state'])
    print("load moco_controlnet success")



def random_sample_features(mid_block_res_sample_hq, mid_block_res_sample_lq, num_samples):

    b, c, h, w = mid_block_res_sample_hq.shape
    random_indices = torch.randint(0, h * w, (num_samples,))
    neg_feature_hq = mid_block_res_sample_hq.view(b, c, -1)[:, :, random_indices]
    neg_feature_lq = mid_block_res_sample_lq.view(b, c, -1)[:, :, random_indices]

    neg_feature_hq = neg_feature_hq.permute(0, 2, 1).contiguous().view(-1, c) 
    neg_feature_lq = neg_feature_lq.permute(0, 2, 1).contiguous().view(-1, c) 

    return neg_feature_hq, neg_feature_lq



def get_neg_feature_and_update_queue(moco_controlnet, noisy_latents, video_length, 
                                     timesteps, encoder_hidden_states, controlnet_cond_hq, 
                                     class_labels_hq, controlnet_cond_lq, class_labels_lq,neg_queue, neg_feature_size):
    with torch.no_grad():
        # hq
        _, _, _, mid_block_res_sample_hq = moco_controlnet(
            rearrange(noisy_latents, "b c f h w -> (b f) c h w", f=video_length),
            timesteps.repeat_interleave(video_length, dim=0),
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond_hq,
            return_dict=False,
            class_labels=class_labels_hq,
        )
        # lq
        _, _, _, mid_block_res_sample_lq = moco_controlnet(
            rearrange(noisy_latents, "b c f h w -> (b f) c h w", f=video_length),
            timesteps.repeat_interleave(video_length, dim=0),
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond_lq,
            return_dict=False,
            class_labels=class_labels_lq,
        )
    # patch sampling
    neg_feature_hq,neg_feature_lq = random_sample_features(mid_block_res_sample_hq[1].detach(),
                                                           mid_block_res_sample_lq[1].detach(),
                                                           num_samples=1)

    neg_feature_hq = F.normalize(neg_feature_hq, dim=1)
    neg_feature_lq = F.normalize(neg_feature_lq, dim=1)

    neg_queue = torch.cat((neg_feature_hq,neg_feature_lq, neg_queue), dim=0)
    # update neg_queue
    if neg_queue.size(0) > neg_feature_size:
        neg_queue = neg_queue[:neg_feature_size]

    return neg_queue

def get_another_feature(moco_controlnet, noisy_latents, video_length, 
                        timesteps, encoder_hidden_states, controlnet_cond, 
                        class_labels,):
    with torch.no_grad():
        _, _, _, mid_block_res_sample = moco_controlnet(
            rearrange(noisy_latents, "b c f h w -> (b f) c h w", f=video_length),
            timesteps.repeat_interleave(video_length, dim=0),
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            return_dict=False,
            class_labels=class_labels,
        )
    
    return mid_block_res_sample



def main(args):

    
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir = args.logging_dir,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    from models.vsr.unet_3d import UNet3DConditionModel

    unet_additional_kwargs = OmegaConf.load(args.unet_config_path).unet_additional_kwargs
    unet_3d = UNet3DConditionModel.from_pretrained_2d(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        unet_additional_kwargs=unet_additional_kwargs
    )
    unet_2d = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    controlnet = ControlNetModel.from_unet(unet_2d, num_class_embeds=2,
                                           resnet_time_scale_shift=args.resnet_time_scale_shift,
                                           use_projection_block=args.controlnet_use_projection_block,
                                           only_mid_projection=True)

    
    del unet_2d
    gc.collect()

    if args.resume_path:
        # unet
        load_path = os.path.join(args.resume_path, "unet_3d", "unet_3d.pth")
        print(f"Loading model from {load_path} to device {accelerator.device}")
        state_dict = torch.load(load_path,map_location=accelerator.device)
        m, u = unet_3d.load_state_dict(state_dict, strict=False)
        logger.info(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        del state_dict
        gc.collect()
        
        # controlnet
        load_path = os.path.join(args.resume_path, "controlnet", "diffusion_pytorch_model.safetensors")
        state_dict = load_file(load_path)
        
        if args.controlnet_use_projection_block:
            m, u = controlnet.load_state_dict(state_dict, strict=False)
            assert len(m)>0 
            assert len(u)==0
        elif args.use_projection_controlnet:
            m, u = controlnet.load_state_dict(state_dict, strict=False)
            assert len(m)==0
            assert len(u)>0
        else:
            controlnet.load_state_dict(state_dict)
        del state_dict
        gc.collect()



    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            i = len(weights) - 1
   
            for i, model in enumerate(models):
             
                sub_dir = "unet_3d" if isinstance(model, UNet3DConditionModel) else "controlnet"
                if isinstance(model, UNet3DConditionModel): # only save motion_model
                    os.makedirs(os.path.join(output_dir, sub_dir),exist_ok=True)
                    save_path = os.path.join(output_dir, sub_dir, "unet_3d.pth")
                    state_dict = model.state_dict()
                    torch.save(state_dict, save_path)
                elif isinstance(model, ControlNetModel): #controlnet
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            # assert len(models) == 2
            print(f"load model --> {accelerator.device}")
            for i in range(len(models)):
  
                model = models.pop()
       
                if isinstance(model, ControlNetModel):
                    load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet") # , low_cpu_mem_usage=False, ignore_mismatched_sizes=True
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                    gc.collect()
                elif isinstance(model, UNet3DConditionModel):
                    load_path = os.path.join(input_dir, "unet_3d", "unet_3d.pth")
                    state_dict = torch.load(load_path,map_location=accelerator.device)
                    # model.load_state_dict(state_dict)
                    m, u = model.load_state_dict(state_dict, strict=True)
                    # logger.debug(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
                    print("Resume from checkpoint unet3D")
                    print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
                    del state_dict
                    gc.collect()


                
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    

    vae.requires_grad_(False)
    unet_3d.requires_grad_(False)
    text_encoder.requires_grad_(False)


    if args.moco:
        assert args.mix_train == True
        print("copy orginal controlnet and not train it!!")
        moco_controlnet = copy.deepcopy(controlnet)
        moco_controlnet.requires_grad_(False)

    controlnet.train()
    unet_3d.train()


    if args.only_unet:
        controlnet.requires_grad_(False)
    elif args.controlnet_unet:
        pass
            
    # unet param
    for name, param in unet_3d.named_parameters():
        if "controlnet_guide_attention" in name:
            param.requires_grad = True
             
        if "motion_modules" in args.trainable_modules:
            if "motion_modules" in name:
                param.requires_grad = True

    for name, param in unet_3d.named_parameters():
        if param.requires_grad or name in unet_3d._buffers:
            if accelerator.is_main_process:
                param_path = os.path.join(args.output_dir,"training_param.txt")
                with open(param_path, "a") as f:
                        f.write(name + "\n")

    for name, param in controlnet.named_parameters():
        if param.requires_grad or name in controlnet._buffers:
            if accelerator.is_main_process:
                param_path = os.path.join(args.output_dir,"controlnet_training_param.txt")
                with open(param_path, "a") as f:
                        f.write(name + "\n")
    for name, param in moco_controlnet.named_parameters():
        if param.requires_grad or name in controlnet._buffers:
            if accelerator.is_main_process:
                param_path = os.path.join(args.output_dir,"moco_controlnet_training_param.txt")
                with open(param_path, "a") as f:
                        f.write(name + "\n")


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet_3d.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet_3d.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )
    if accelerator.unwrap_model(unet_3d).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(unet_3d).dtype}. {low_precision_error_string}"
        )
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    optimizer_class = torch.optim.AdamW

    params_to_optimize = []

    if args.learning_rate2 == -100.0:
        args.learning_rate2 = args.learning_rate


    if args.only_unet:
        params_to_optimize.append({'params': unet_3d.parameters(), 'lr': args.learning_rate2})
        

    elif args.controlnet_unet:
        params_to_optimize.append({'params': controlnet.parameters(), 'lr': args.learning_rate})
        params_to_optimize.append({'params': unet_3d.parameters(), 'lr': args.learning_rate2})



    optimizer = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    hq_optimizer = None
    if args.ssl_setting2:
        params_to_optimize_hq = [{'params': unet_3d.parameters(), 'lr': args.learning_rate2}]

        hq_optimizer = optimizer_class(
        params_to_optimize_hq,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
        



    degradation = [RealBasicVSR_degradation()] if args.degradation == 'basicvsr' else [RealESRGAN_VSR_degradation()]
    if args.degradation == 'random':
        degradation =  [RealBasicVSR_degradation(),RealESRGAN_VSR_degradation()]
    # meta_path1 = "xxx/youhq_meta_info_train_512.txt"
    # hr_root1 = "xxx/Train_sub_512"
    # caption_path1 = "xxx/youhq_caption.csv"
    # train_dataset1 = LocalVideoDataset(meta_path = meta_path1,
    #                                 hr_root = hr_root1,
    #                                 mode='skip',
    #                                 resize_bank=True,
    #                                 num_frame=args.num_frame,
    #                                 tokenizer = tokenizer,
    #                                 crop_size = args.crop_size,
    #                                 caption_path=caption_path1 if args.use_caption else None,
    #                                 degradation=degradation)
    meta_path2 = "datasets_example/reds_meta_info_example.txt"
    hr_root2 = "datasets_example/REDS"
    # meta_path2 = "REDS/reds_meta_info_train_512.txt"
    # hr_root2 = "REDS/train_sharp_sub_512"
    caption_path2 = None
    train_dataset2 = LocalVideoDataset(meta_path = meta_path2,
                                    hr_root = hr_root2,
                                    mode='not_all',
                                    resize_bank=True,
                                    num_frame=args.num_frame,
                                    tokenizer = tokenizer,
                                    crop_size = args.crop_size,
                                    caption_path=caption_path2 if args.use_caption else None,
                                    degradation=degradation)

    # train_dataset  = ConcatDataset([train_dataset1, train_dataset2])
    train_dataset = train_dataset2

    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=args.dataloader_num_workers,
        batch_size=args.train_batch_size,
        #prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
        pin_memory=True,
        shuffle=True
    )



    overrode_max_train_steps = False
    train_dataloader_length = len(train_dataloader)

    num_update_steps_per_epoch = math.ceil(train_dataloader_length / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if hq_optimizer is not None:

        hq_lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=hq_optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )
            
        unet_3d, controlnet, optimizer, lr_scheduler, hq_optimizer, hq_lr_scheduler,train_dataloader = accelerator.prepare(
            unet_3d, controlnet, optimizer, lr_scheduler, hq_optimizer, hq_lr_scheduler,train_dataloader
            )
            

    else:
        unet_3d, controlnet, optimizer, lr_scheduler,train_dataloader = accelerator.prepare(
            unet_3d, controlnet, optimizer, lr_scheduler,train_dataloader
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.moco:
        moco_controlnet.to(accelerator.device, dtype=weight_dtype)
  



    # We need to recalculate our total training steps as the size of the training dataloader may have changed.

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("trainable_modules")
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")

    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  num_frame = {args.num_frame}")
    logger.info(f"  trainable_modules = {args.trainable_modules}")
    logger.info(f"  degradation = {degradation}")
    logger.info(f"  Train only_unet = {args.only_unet}")
    logger.info(f"  Train controlnet_unet = {args.controlnet_unet}")
    logger.info(f"  contrastive_loss = {args.contrastive_loss}")
    logger.info(f"  use_label = {args.use_label}")


    global_step = 0
    first_epoch = 0
    neg_queue = None
    neg_feature_size = args.neg_feature_size


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        path = os.path.basename(args.resume_from_checkpoint)
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
    

            if args.moco:
                load_moco_checkpoint(args.resume_from_checkpoint,moco_controlnet,
                                                accelerator.device,weight_dtype)
        
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            
        print("sleep 10s")
        time.sleep(10)
        gc.collect()
        torch.cuda.empty_cache()
        
    else:
        initial_global_step = 0
        
    accelerator.wait_for_everyone()



    if args.contrastive_loss == "infonce" and neg_queue is None:
        # init
        neg_queue = torch.randn(args.neg_feature_size,  
                                moco_controlnet.config.block_out_channels[-1],device=accelerator.device)
        neg_queue = F.normalize(neg_queue, dim=1)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )


    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(controlnet), accelerator.accumulate(unet_3d):

                pixel_values, conditioning_pixel_values, input_ids = batch["pixel_values"], batch["conditioning_pixel_values"],batch["input_ids"]
                bsz_3d = pixel_values.shape[0]
                pixel_values = pixel_values.to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                conditioning_pixel_values = conditioning_pixel_values.to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                video_length = pixel_values.shape[1]


                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                conditioning_pixel_values = rearrange(conditioning_pixel_values, "b f c h w -> (b f) c h w")
                input_ids = rearrange(input_ids, "b f seq -> (b f) seq")
 
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)

                noise = torch.randn_like(latents)

                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz_3d,), device=latents.device)

                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(input_ids.to(accelerator.device))[0]
    
                conditioning_pixel_values = conditioning_pixel_values * 2 - 1

                hq_label = torch.ones((pixel_values.shape[0], 1, pixel_values.shape[2], pixel_values.shape[3]), device=accelerator.device, dtype=weight_dtype)
                pixel_values = torch.cat((pixel_values, hq_label), dim=1)
                
                lq_label = torch.zeros((conditioning_pixel_values.shape[0], 1, conditioning_pixel_values.shape[2], conditioning_pixel_values.shape[3]), device=accelerator.device, dtype=weight_dtype)
                conditioning_pixel_values = torch.cat((conditioning_pixel_values, lq_label), dim=1)


                class_labels_hq = torch.ones(bsz_3d*video_length, device=accelerator.device, dtype=weight_dtype).long()
                class_labels_lq = torch.zeros(bsz_3d*video_length, device=accelerator.device, dtype=weight_dtype).long()
          
                contrastive_loss = None
                flag = None

                if args.mix_train:
                 
                    assert args.train_high_quality
                    high_ratio = args.high_ratio if not args.linear_constant_ratio else calculate_high_ratio(global_step,args)
            
                    if random.random() < high_ratio:
                        flag = "hq"
                        if args.ssl_setting1:          
                            _, control_temb, down_block_res_samples, mid_block_res_sample = controlnet(
                                rearrange(noisy_latents, "b c f h w -> (b f) c h w", f=video_length),
                                timesteps.repeat_interleave(video_length, dim=0),
                                encoder_hidden_states=encoder_hidden_states,
                                controlnet_cond=pixel_values,
                                return_dict=False,
                                class_labels=class_labels_hq,
                            )
                        elif args.ssl_setting2:
                            with torch.no_grad():
                                _,control_temb, down_block_res_samples, mid_block_res_sample = moco_controlnet(
                                    rearrange(noisy_latents, "b c f h w -> (b f) c h w", f=video_length),
                                    timesteps.repeat_interleave(video_length, dim=0),
                                    encoder_hidden_states=encoder_hidden_states,
                                    controlnet_cond=pixel_values,
                                    return_dict=False,  
                                    class_labels=class_labels_hq,
                                )
                   
                                control_temb = control_temb.detach()
                                down_block_res_samples[0] = [sample.detach() for sample in down_block_res_samples[0]]
                                mid_block_res_sample[0] = mid_block_res_sample[0].detach()

                        else:
                            raise ValueError("must use one ssl setting")
                            

                    else:
                        flag="lq"
                        _,control_temb, down_block_res_samples, mid_block_res_sample = controlnet(
                            rearrange(noisy_latents, "b c f h w -> (b f) c h w", f=video_length),
                            timesteps.repeat_interleave(video_length, dim=0),
                            encoder_hidden_states=encoder_hidden_states,
                            controlnet_cond=conditioning_pixel_values,
                            return_dict=False,
                            class_labels=class_labels_lq,
                        )

                        another_feature = get_another_feature(moco_controlnet, noisy_latents, video_length, 
                                            timesteps, encoder_hidden_states, 
                                            controlnet_cond=pixel_values, 
                                            class_labels=class_labels_hq)
    



                if args.mix_train and args.contrastive_loss:
                    
                    if flag=="lq":
                        if args.contrastive_loss == "infonce":
                            feat1 = mid_block_res_sample[1]
                            feat2 = another_feature[1].detach()
                        else:
                            feat1 = mid_block_res_sample[0]
                            feat2 = another_feature[0].detach()
                        feat1 = rearrange(feat1,"b c h w -> (b h w) c")
                        feat2 = rearrange(feat2,"b c h w -> (b h w) c")
                        contrastive_loss = get_contrastive_loss(loss_type=args.contrastive_loss,
                                                                    feat1=feat1, feat2=feat2,
                                                                    cl_mode=args.cl_mode,
                                                                    cl_weight=args.cl_weight,
                                                                    neg_feat=neg_queue,
                                                                    temperature=args.temperature)
                        
                    else:
                        contrastive_loss = None
             
                    if args.contrastive_loss == "infonce":
                        neg_queue = get_neg_feature_and_update_queue(
                            moco_controlnet, noisy_latents, video_length, 
                            timesteps, encoder_hidden_states, 
                            controlnet_cond_hq=pixel_values, 
                            class_labels_hq=class_labels_hq, 
                            controlnet_cond_lq=conditioning_pixel_values, 
                            class_labels_lq=class_labels_lq,
                            neg_queue=neg_queue, neg_feature_size=neg_feature_size
                        )
                    
            
               
                down_block_res_samples = [rearrange(sample, "(b f) c h w -> b c f h w", f=video_length) for sample in down_block_res_samples[0]]
                mid_block_res_sample = rearrange(mid_block_res_sample[0], "(b f) c h w -> b c f h w", f=video_length)
  
                # start_time = time.time()
                model_pred = unet_3d(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    class_labels=None,
                    control_temb=control_temb,
                ).sample

               
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean") 

                if contrastive_loss is not None:
        
                    loss = contrastive_loss + mse_loss
                else:
                    loss = mse_loss
                


   
                   
                accelerator.backward(loss)

                try:
                    avg_norm_controlnet = log_grad_norms(controlnet, make_print=False)
                    avg_norm_unet = log_grad_norms(unet_3d, make_print=False)
                except:
                    avg_norm_controlnet = None
                    avg_norm_unet = None

                
                

                if args.ssl_setting2 and flag == "hq":
                    if accelerator.sync_gradients:
                        params_to_clip = list(unet_3d.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    hq_optimizer.step()
                    hq_lr_scheduler.step()
                    hq_optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                else:
           
                    if accelerator.sync_gradients:
                        params_to_clip = list(controlnet.parameters()) + list(unet_3d.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

       
                if args.moco:
                    for param_q, param_k in zip(controlnet.parameters(), moco_controlnet.parameters()):
                        param_q_data_ = param_q.data.to(param_k.dtype)
                        param_k.data = args.momentum * param_k.data + (1 - args.momentum) * param_q_data_



            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:




                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")


                        if args.overwrite:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            elder_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[0]
                            os.rename(os.path.join(args.output_dir, elder_checkpoint), save_path)

                        accelerator.save_state(save_path)
                        if args.moco:
                            save_moco_checkpoint(global_step,save_path,moco_controlnet=moco_controlnet)

                        logger.info(f"Saved state to {save_path}")
  

            logs = {"full_loss": loss.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0],
                    "mse_loss" : mse_loss.detach().item(),
                    "high_ratio": high_ratio if args.mix_train and args.train_high_quality else None,
                    "avg_norm_controlnet": avg_norm_controlnet,
                    "avg_norm_unet":avg_norm_unet,
                    
                    }
            if contrastive_loss is not None:
                logs["contrastive_loss"] = contrastive_loss.detach().item()

            if flag is not None:
                logs["flag"] = flag


            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

    

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
