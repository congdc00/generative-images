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

`CUDA_VISIBLE_DEVICES=1 accelerate launch train_dreambooth.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --instance_data_dir="data/cozy-house_all/" --output_dir="checkpoints_dreambooth/" --instance_prompt="a cozy house" --resolution=512 --train_batch_size=12 --gradient_accumulation_steps=4 --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=500 --use_8bit_adam --gradient_checkpointing --num_validation_images=20 --validation_steps=100 --validation_prompt="the image features a bedroom with a large window, providing a stunning view of the city at night, the bed is positioned in front of the window, and there are several lit candles scattered around the room, creating a cozy and intimate atmosphere, the room also contains a fireplace, adding warmth and ambiance to the space, the combination of the lit candles, the city view, and the fireplace creates a serene and inviting environment" --report_to "wandb"`

### Stable-Diffusion

`python train.py --resume_ckpt "runwayml/stable-diffusion-v1-5" --gpuid 0 --max_epochs 300 --data_root "input/train" --lr_scheduler constant --project_name classic_house --batch_size 1 --sample_steps 200 --lr 3e-6 --resolution 1280 --save_every_n_epochs 100 --wandb --grad_accu 100`

### Loras

` accelerate launch train_loras.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" --dataset_name="data/cozy-house_all" --dataloader_num_workers=2 --resolution=1024 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps=500 --learning_rate=1e-04 --max_grad_norm=1 --lr_scheduler="cosine" --lr_warmup_steps=0 --output_dir="checkpoints_lora_sdxl" --report_to=wandb --checkpointing_steps=250 --validation_prompt="The image depicts a cozy and warm interior of a room, possibly a library or a personal study. The room is adorned with wooden shelves filled with books, a large window that offers a view of a snowy landscape outside, and a comfortable seating area with cushions and a table. There's also a fireplace with a roaring fire, and various decorative items like candles, a fish tank, and a neon sign that reads 'Jolly Cafe'. The ambiance is enhanced by the soft glow of lights and the presence of a few pets, including a cat and a dog, adding to the homely feel"`
