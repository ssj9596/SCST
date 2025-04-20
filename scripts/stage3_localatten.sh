PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch \
	--num_processes=2 \
	train_stage13.py \
 	--pretrained_model_name_or_path="checkpoints/stable-diffusion-2-1-base" \
	--output_dir="stage3_localatten" \
	--resolution=512 \
	--learning_rate=2e-5 \
	--gradient_accumulation_steps=2 \
	--num_frame=8 \
	--train_batch_size=4 \
	--num_train_epochs=1000 \
	--tracker_project_name="vsr" \
	--checkpointing_steps=5000 \
	--mixed_precision="fp16" \
	--dataloader_num_workers=2 \
	--trainable_modules="motion_modules" \
	--only_unet \
	--resnet_time_scale_shift="scale_shift" \
	--use_caption \
	--crop_size=512 \
	--seed=42 \
	--max_train_steps=100000 \
	--unet_config_path="models/configs/localatten.yaml" \
	--logging_dir="stage3_localatten/logs" \
	--resume_path="stage2/checkpoint-55000" \
	--use_projection_controlnet \
	--enable_xformers_memory_efficient_attention \


