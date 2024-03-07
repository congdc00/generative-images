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

'cài diffuser bản mới nhất"
`pip install git+https://github.com/huggingface/diffusers``

`export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="path/to/dataset"
export OUTPUT_DIR="path/to/checkpoint"`

### Dreambooth

`CUDA_VISIBLE_DEVICES=1 accelerate launch train_dreambooth.py   --pretrained_model_name_or_path=$MODEL_NAME --instance_data_dir=$INSTANCE_DIR   --output_dir=$OUTPUT_DIR   --instance_prompt="a photo of CONGDC bedroom, warm light" --resolution=1280   --train_batch_size=1   --gradient_accumulation_steps=1   --learning_rate=5e-6   --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=4000 --use_8bit_adam --gradient_checkpointing --enable_xformers_memory_efficient_attention --set_grads_to_none --num_validation_images=100 --validation_steps=100 --validation_prompt="a photo of CONGDC bedroom, warm light"`

### Stable-Diffusion

`python train.py --resume_ckpt "runwayml/stable-diffusion-v1-5" --gpuid 0 --max_epochs 300 --data_root "input/train" --lr_scheduler constant --project_name classic_house --batch_size 1 --sample_steps 200 --lr 3e-6 --resolution 1280 --save_every_n_epochs 100 --wandb --grad_accu 100`

### Loras

`accelerate launch train_dreambooth_lora_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
 --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
 --mixed_precision="fp16" \
 --instance_prompt="a cozy house, in house" \
 --resolution=1024 \
 --train_batch_size=1\
 --gradient_accumulation_steps=8 \
 --learning_rate=1e-4 \
 --report_to="wandb" \
 --lr_scheduler="constant" \
 --lr_warmup_steps=0 \
 --max_train_steps=2000 \
 --validation_prompt="a cozy house, in house, the image depicts a spacious and well-lit living room with a fireplace, the room features a large tv mounted on the wall, and a comfortable couch is placed in the background, there are several chairs scattered throughout the room, and a dining table is located near the center,in addition to the furniture, the living room is adorned with various decorative elements, such as a potted plant, a vase, and a bowl, there are also multiple books placed around the room, adding to the cozy atmosphere, the room is further enhanced by the presence of a fireplace, which creates a warm and inviting ambiance" \
 --validation_epochs=50 \
 --seed="0"
`
