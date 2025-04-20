PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch \
	--num_processes=2 \
	train_stage2.py \
 	--pretrained_model_name_or_path="checkpoints/stable-diffusion-2-1-base" \
	--output_dir="stage2" \
	--resolution=512 \
	--learning_rate=2e-5 \
	--gradient_accumulation_steps=2 \
	--num_frame=8 \
	--train_batch_size=3 \
	--num_train_epochs=1000 \
	--tracker_project_name="vsr" \
	--checkpointing_steps=5000 \
	--mixed_precision="fp16" \
	--dataloader_num_workers=3 \
	--train_high_quality \
	--controlnet_unet \
	--mix_train \
	--high_ratio=0.5 \
	--contrastive_loss="infonce" \
	--neg_feature_size=1024 \
	--moco \
	--ssl_setting2 \
	--cl_weight=5e-4 \
	--momentum=0.999 \
	--temperature=0.2 \
	--resnet_time_scale_shift="scale_shift" \
	--use_caption \
	--crop_size=512 \
	--seed=42 \
	--max_train_steps=100000 \
	--unet_config_path="models/configs/stage1.yaml" \
	--logging_dir="stage2/logs" \
	--resume_path="stage1/checkpoint-xxx" \
	--controlnet_use_projection_block \
	--enable_xformers_memory_efficient_attention \



# --resume_path="stage1/checkpoint-30000" \