PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch \
	--num_processes=2 \
	train_stage13.py \
 	--pretrained_model_name_or_path="checkpoints/stable-diffusion-2-1-base" \
	--output_dir="stage1" \
	--resolution=512 \
	--learning_rate=5e-5 \
	--gradient_accumulation_steps=2 \
	--num_frame=8 \
	--train_batch_size=4 \
	--num_train_epochs=1000 \
	--tracker_project_name="vsr" \
	--checkpointing_steps=5000 \
	--mixed_precision="fp16" \
	--dataloader_num_workers=4 \
	--train_high_quality \
	--controlnet_unet \
	--mix_train \
	--min_high_ratio=0.3 \
	--max_high_ratio=1 \
	--high_ratio_steps=15000 \
	--linear_constant_ratio \
	--resnet_time_scale_shift="scale_shift" \
	--pixelwise_attention_temb \
	--use_caption \
	--crop_size=512 \
	--seed=42 \
	--max_train_steps=100000 \
	--unet_config_path="models/configs/stage1.yaml" \
	--logging_dir="stage1/logs" \
	--enable_xformers_memory_efficient_attention \
	# --resume_from_checkpoint stage1/checkpoint-5000



