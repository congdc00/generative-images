from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch

unet = UNet2DConditionModel.from_pretrained("data/output/dreambooth_sd15_v2/unet")

# if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
text_encoder = CLIPTextModel.from_pretrained("data/output/dreambooth_sd15_v2/text_encoder")

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", unet=unet, text_encoder=text_encoder,
    ).to("cuda")

image = pipeline("a cozy house", 
                 num_inference_steps=50, 
                 guidance_scale=4,
                 height=512, width=768).images[0]
image.save("test_dreambooth.png")