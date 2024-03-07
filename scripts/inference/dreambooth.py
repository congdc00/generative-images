from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch

unet = UNet2DConditionModel.from_pretrained("data/output/dreambooth_sd15_v1/unet")

# if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
text_encoder = CLIPTextModel.from_pretrained("data/output/dreambooth_sd15_v1/text_encoder")

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", unet=unet, text_encoder=text_encoder,
    ).to("cuda")

image = pipeline("a cozy house, in house, the image depicts a spacious and well-lit living room with a fireplace, the room features a large tv mounted on the wall, and a comfortable couch is placed in the background, there are several chairs scattered throughout the room, and a dining table is located near the center,in addition to the furniture, the living room is adorned with various decorative elements, such as a potted plant, a vase, and a bowl, there are also multiple books placed around the room, adding to the cozy atmosphere, the room is further enhanced by the presence of a fireplace, which creates a warm and inviting ambiance", 
                 num_inference_steps=50, 
                 guidance_scale=7.5,
                 height=768, width=1280).images[0]
image.save("test_dreambooth.png")