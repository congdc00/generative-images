# generative-images

# Requirement

python 3.10

## Setup environtment

`pip install -r requirements.txt`
`pip3 install torch torchvision torchaudio`
`pip install git+https://github.com/huggingface/diffusers.git`
`pip install git+https://github.com/openai/CLIP.git`
`accelerate config`

## RUN

`CUDA_VISIBLE_DEVICES=1 accelerate launch train_dreambooth.py   --pretrained_model_name_or_path=$MODEL_NAME --instance_data_dir=$INSTANCE_DIR   --output_dir=$OUTPUT_DIR   --instance_prompt="a photo of CONGDC bedroom, warm light" --resolution=1280   --train_batch_size=1   --gradient_accumulation_steps=1   --learning_rate=5e-6   --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=4000 --use_8bit_adam --gradient_checkpointing --enable_xformers_memory_efficient_attention --set_grads_to_none --num_validation_images=100 --validation_steps=100 --validation_prompt="a photo of CONGDC bedroom, warm light"`
