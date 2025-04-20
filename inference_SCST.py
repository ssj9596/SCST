import os
import sys
import tarfile
import cv2
import glob
import argparse
from einops import rearrange
from omegaconf import OmegaConf
# import open_clip
import numpy as np
from PIL import Image
import safetensors.torch

import torch
from torchvision import transforms
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import PNDMScheduler, LCMScheduler, UniPCMultistepScheduler,DDIMScheduler, DPMSolverMultistepScheduler#, StableDiffusionControlNetPipeline
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from basicsr.archs.basicvsr_arch import BasicVSR

from pipelines.pipeline_SCST import StableDiffusionControlNetPipeline

from myutils.wavelet_color_fix import wavelet_color_fix
import imageio



from tqdm.auto import tqdm
#from annotator.retinaface import RetinaFaceDetection

sys.path.append('SCST')

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


        


def load_vsr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    print("args.added_noise_level")
    print(args.added_noise_level)

    # Load scheduler, tokenizer and models.

    # if args.ddim:
    #     scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    scheduler = UniPCMultistepScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.pretrained_model_path}/feature_extractor")


    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")

    infer_config = OmegaConf.load(args.unet_config_path)
    unet_additional_kwargs = infer_config.unet_additional_kwargs
    
    from models.vsr.unet_3d import UNet3DConditionModel
    unet_3d = UNet3DConditionModel.from_pretrained_2d(
        args.pretrained_model_path,
        subfolder="unet",
        unet_additional_kwargs=unet_additional_kwargs,
    )

    params = [p.numel() if "motion" in n else 0 for n, p in unet_3d.named_parameters()]
    print(f"Loaded {sum(params) / 1e6}M-parameter motion module")
    

    load_path = os.path.join(args.ckpt_model_path)
    print(f"Loading model from {load_path} to device {accelerator.device}")

    state_dict = torch.load(load_path,map_location=accelerator.device)
    m,n = unet_3d.load_state_dict(state_dict,strict=False)
    print(f"Load Stage in EVAL Stage ## miss:{m}, extra:{n}")
    del state_dict


    from models.controlnet.controlnet import ControlNetModel
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path)
    print(f"ControlNet parameters: {sum(p.numel() for p in controlnet.parameters()) / 1e6}M")

    # if args.mococtrl:
    #     ckpt_path = os.path.join(args.ckpt_model_path,"moco_ckpt.pth")
    #     checkpoint = torch.load(ckpt_path, map_location=accelerator.device)
    #     controlnet.load_state_dict(checkpoint['model_state'],strict=False)


    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet_3d.requires_grad_(False)
    controlnet.requires_grad_(False)
    

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet_3d.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet_3d.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet_3d=unet_3d, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )
    validation_pipeline.enable_vae_tiling()
    validation_pipeline._init_tiled_vae(encoder_tile_size=args.encoder_tiled_size, decoder_tile_size=args.decoder_tiled_size)


    return validation_pipeline

def main(args, enable_xformers_memory_efficient_attention=True):
 
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)


    if accelerator.is_main_process:
        accelerator.init_trackers("VSR")

    pipeline = load_vsr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)

 


    resize_preproc = transforms.Compose([
        transforms.Resize(args.process_size, interpolation=transforms.InterpolationMode.BILINEAR),
    ])


    def load_imgs(lq_path):
        lq_images = sorted(glob.glob(f'{lq_path}/*.*'))
        return lq_images

                
    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        if os.path.isdir(args.video_path):
            images = load_imgs(args.video_path)

        else:
            images = [args.video_path]
        input_filenames = [os.path.splitext(os.path.basename(img))[0] for img in images]

        num_frame = args.num_frame

        train_images = []
        input_images = []
        all_prompts = []
        negative_prompts = []

        img_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])

        

        for lq_image_name in images[:]:
            validation_image = Image.open(lq_image_name).convert("RGB")
            validation_prompt = args.prompt
            negative_prompt = args.negative_prompt

            ori_width, ori_height = validation_image.size
    
            rscale = args.upscale
            validation_image = validation_image.resize((validation_image.size[0] * rscale, validation_image.size[1] * rscale))


            if min(validation_image.size) < args.process_size:
                validation_image = resize_preproc(validation_image)
            validation_image = validation_image.resize((validation_image.size[0] // 8 * 8, validation_image.size[1] // 8 * 8))
            train_image = img_preproc(validation_image)


            # [-1,1]
            train_image = torch.clamp(train_image * 2.0 - 1.0, min=-1.0, max=1.0)
            train_images.append(train_image)
            input_images.append(validation_image)
            all_prompts.append(validation_prompt)
            negative_prompts.append(negative_prompt)


        images_input = torch.stack(train_images)
        prompts = all_prompts

        images_output,latents = pipeline(
            args, prompts, images_input, num_inference_steps=args.num_inference_steps, generator=generator,
            guidance_scale=args.guidance_scale, negative_prompt=negative_prompts, conditioning_scale=args.conditioning_scale,
            num_frame=num_frame,upscale=args.upscale, overlap_frame = args.overlap_frame
        
        )
        # choose save latent or ori_img
        # ori_img_dir = os.path.join(args.output_dir, 'ori_img')
        # latent_dir = os.path.join(args.output_dir, 'latent')
        # color_fix_dir = os.path.join(args.output_dir, 'color_fix')
        color_fix_dir = args.output_dir
        video_path = os.path.join(args.output_dir, 'output_video.mp4')
        frame_width, frame_height = ori_width * rscale, ori_height * rscale

        # os.makedirs(ori_img_dir, exist_ok=True)
        # os.makedirs(latent_dir, exist_ok=True)
        os.makedirs(color_fix_dir, exist_ok=True)
        image_cv_list = []


        for i in tqdm(range(len(images_output))):
            image = images_output[i]
            image_name = input_filenames[i]
            # ori_img = image.resize((frame_width, frame_height))
            # ori_img.save(f'{ori_img_dir}/{image_name}.png')
            # latent_path = f'{latent_dir}/{image_name}.pt'
            # torch.save(latents[i].detach().cpu(), latent_path)
            image = wavelet_color_fix(image, input_images[i])
            image = image.resize((frame_width, frame_height))
            image.save(f'{color_fix_dir}/{image_name}.png')
            # import pdb
            # pdb.set_trace()
            image_array = np.array(image)
            image_cv_list.append(image_array)

        if args.save_video:
            imageio.mimwrite(video_path, image_cv_list, fps=args.frame_rate,  quality=8, output_params=["-loglevel", "error"],codec='libx264') 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="xxx/stable-diffusion-2-1-base", help="path of base SD model")
    parser.add_argument("--ckpt_model_path", type=str,)
    parser.add_argument("--controlnet_path", type=str,)
    parser.add_argument("--num_frame", type=int, default=8, help="num frame of video")
    parser.add_argument("--overlap_frame", type=int, default=0, help="num frame of video")
    parser.add_argument("--unet_config_path", type=str, default="", help="path of unet config")

    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--frame_rate", type=int, default=12)

    # parser.add_argument("--mococtrl",  action="store_true")
    parser.add_argument("--prompt", type=str, default="", help="prompt for image generation")
    parser.add_argument("--added_prompt", type=str, default="", help="additional prompt")
    parser.add_argument("--negative_prompt", type=str, default="", help="negative prompt")

    parser.add_argument("--video_path", type=str,  help="test image path or folder")
    parser.add_argument("--output_dir", type=str, default="", help="output folder")
    parser.add_argument("--mixed_precision", type=str, default="fp16", help="mixed precision mode")
    parser.add_argument("--guidance_scale", type=float, default=-1.0, help="classifier-free guidance scale")
    parser.add_argument("--conditioning_scale", type=float, default=1.0, help="conditioning scale for controlnet")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="denoising steps")
    parser.add_argument("--process_size", type=int, default=768, help="minimal input size for processing") # 512?
    parser.add_argument("--decoder_tiled_size", type=int, default=224, help="decoder tile size for saving GPU memory") # for 24G
    parser.add_argument("--encoder_tiled_size", type=int, default=1024, help="encoder tile size for saving GPU memory") # for 24G
    parser.add_argument("--latent_tiled_size", type=int, default=256, help="unet latent tile size for saving GPU memory") # for 24G
    parser.add_argument("--latent_tiled_overlap", type=int, default=16, help="unet lantent overlap size for saving GPU memory") # for 24G
    parser.add_argument("--upscale", type=int, default=4, help="upsampling scale")
    parser.add_argument("--init_latent_with_noise", action="store_true", help="initial latent with pure noise or not")
    parser.add_argument("--added_noise_level", type=int, default=900, help="additional noise level")
    parser.add_argument("--init_noise_level", type=int, default=999, help="init noise level")
    parser.add_argument("--offset_noise_scale", type=float, default=0.0, help="offset noise scale, not used")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    args = parser.parse_args()
    main(args)
