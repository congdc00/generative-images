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

`export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="path/to/dataset"
export OUTPUT_DIR="path/to/checkpoint"`

### Dreambooth

`CUDA_VISIBLE_DEVICES=1 accelerate launch train_dreambooth.py   --pretrained_model_name_or_path=$MODEL_NAME --instance_data_dir=$INSTANCE_DIR   --output_dir=$OUTPUT_DIR   --instance_prompt="a photo of CONGDC bedroom, warm light" --resolution=1280   --train_batch_size=1   --gradient_accumulation_steps=1   --learning_rate=5e-6   --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=4000 --use_8bit_adam --gradient_checkpointing --enable_xformers_memory_efficient_attention --set_grads_to_none --num_validation_images=100 --validation_steps=100 --validation_prompt="a photo of CONGDC bedroom, warm light"`

### Stable-Diffusion

`python train.py --resume_ckpt "runwayml/stable-diffusion-v1-5" --gpuid 0 --max_epochs 300 --data_root "input/train" --lr_scheduler constant --project_name classic_house --batch_size 1 --sample_steps 200 --lr 3e-6 --resolution 1280 --save_every_n_epochs 100 --wandb --grad_accu 100`

### Loras

`accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \  
  --pretrained_model_name_or_path=$MODEL_NAME \  
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=1280 \
  --center_crop \
  --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="a cozy house" \
  --seed=1337`
