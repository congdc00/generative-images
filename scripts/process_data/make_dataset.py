from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="/path/to/folder")
dataset["train"][0]