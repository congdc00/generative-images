import requests
from PIL import Image
import os
import shutil
from glob import glob
import csv
import tqdm

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import transformers
data_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_quant_type="nf4",
)

model_id = "llava-hf/llava-1.5-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        quantization_config=data_config,
    )
transformers.logging.set_verbosity_error()

SRC_DATA = "data/input/images/processed/house/best_quality"
DST_DATA = "data/input/images/test/"
def predict_label(image):
    prompt = "USER: <image>\nDescribe this image as prompt.\nASSISTANT:"
    processor = AutoProcessor.from_pretrained(model_id)
    
    inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)

    output = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    result = processor.decode(output[0][2:], skip_special_tokens=True)
    result = result.split("ASSISTANT: ")[1]
    return result

if __name__ == "__main__":

    list_folder = glob(f"{SRC_DATA}/*")
    metadata = []
    for folder in list_folder:
        list_img_path = glob(f"{folder}/*")
        print(folder)
        for src_img_path in list_img_path:
            image = Image.open(src_img_path)
            import time
            start_time = time.time()
            
            result = predict_label(image)
            result = result.lower()
            result = result.replace("\n", "")
            result = result.replace(".", ",")
            result = result[:-1]
            
            
            name_image = os.path.basename(src_img_path).split(".")[0]
            with open(f"{DST_DATA}/{name_image}.txt", 'w', newline='') as file:
                file.write(result)