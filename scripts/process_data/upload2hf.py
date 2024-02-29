from datasets import load_dataset

# example 1: local folder
dataset = load_dataset("cozy_house", data_dir="data/input/images/raw_images")

# # example 2: local files (supported formats are tar, gzip, zip, xz, rar, zstd)
# dataset = load_dataset("./", data_files="path_to_zip_file")

# # example 3: remote files (supported formats are tar, gzip, zip, xz, rar, zstd)
# dataset = load_dataset(
#     "imagefolder",
#     data_files="https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip",
# )

# # example 4: providing several splits
# dataset = load_dataset(
#     "imagefolder", data_files={"train": ["path/to/file1", "path/to/file2"], "test": ["path/to/file3", "path/to/file4"]}
# )

dataset.push_to_hub("congdc/IMG-GEN", private=True)