
import argparse
import os
import subprocess
import pandas as pd
import time


if __name__ == "__main__":
    pretrained_model_path = "checkpoints/stable-diffusion-2-1-base"
    controlnet_path = "checkpoints/controlnet"
    init_noise_level = 999
    # added_noise_level ⬇ -> PSNR ⬆ 
    # choose from [999, 949, 899, 849, 799, 749, 699, 649, 599, 549, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50]
    added_noise_level = 350
    num_frame = 8
    overlap_frame = 2

    video_path = "inputs/videolq_046"
    prompt = "A dog is sitting on a bed in a room."

    config_path = "models/configs/stcm.yaml"
    ckpt_path = "checkpoints/stcm_unet.pth"

    output_dir = "outputs/stcm"

    
    if os.path.isdir(video_path):
        output_dir = os.path.join(output_dir,os.path.basename(video_path))

    command = [
            "python", "inference_SCST.py",
            "--ckpt_model_path", ckpt_path,
            "--decoder_tiled_size", "224",
            "--encoder_tiled_size", "2048",
            "--num_inference_steps", "20",
            "--latent_tiled_size", "96",
            "--video_path", video_path,
            "--added_noise_level", str(added_noise_level),  # 使用动态的 added_noise_level
            "--init_noise_level", str(init_noise_level),
            "--output_dir", output_dir,
            "--num_inference_steps", "20",
            "--upscale", "4",
            "--process_size", "768",
            "--overlap_frame", str(overlap_frame),
            "--unet_config_path", config_path,
            "--seed", "42",
            "--num_frame", str(num_frame),
            "--prompt", prompt,
            "--negative_prompt", "blurry, dotted, noise, raster lines, unclear, lowres, over-smoothed",
            "--guidance_scale", "5.0",
            "--frame_rate", "12",
            "--pretrained_model_path", pretrained_model_path,
            "--controlnet_path", controlnet_path,
            "--save_video",
        ]
    subprocess.run(command, check=True)
