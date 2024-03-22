import os
import json
import requests
import threading
import shutil
import dateutil.parser
from queue import Queue
from tqdm import tqdm

total_files = 0
downloaded_files = 0
download_lock = threading.Lock()

def download_image(folder, url, filename, extension):
    global downloaded_files
    full_path = f"{folder}/{filename}.{extension}"
    if os.path.exists(full_path):
        return False
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(full_path, "wb") as f:
                f.write(response.content)
            with download_lock:
                downloaded_files += 1
            return True
        else:
            return False
    except Exception as e:
        return False
        
def write_metadata(folder, filename, tags):
    with open(f"{folder}/{filename}.txt", "w") as f:
        f.write(tags)
    
def generate_tags(data):
    def process_tags(tag_str):
        processed_tags = []
        for tag_name in tag_str.split(" "):
            if len(tag_name) > 3:
                tag_name = tag_name.replace("_", " ")
            processed_tags.append(tag_name)
        return ", ".join(processed_tags)

    created_at = data.get("media_asset", {}).get("created_at", "")
    try:
        parsed_date = dateutil.parser.isoparse(created_at)
        year = parsed_date.year
        if 2005 <= year <= 2010:
            year_tag = "oldest"
        elif 2011 <= year <= 2014:
            year_tag = "early"
        elif 2015 <= year <= 2018:
            year_tag = "mid"
        elif 2019 <= year <= 2021:
            year_tag = "late"
        elif 2022 <= year <= 2023:
            year_tag = "newest"
        else:
            year_tag = "unknown"
    except (ValueError, AttributeError):
        print("Invalid or missing created_at date.")
        year_tag = "unknown"

    rating = data.get("rating")
    score = data.get("score")

    tags_general = process_tags(data.get("tag_string_general", ""))
    tags_character = process_tags(data.get("tag_string_character", ""))
    tags_copyright = process_tags(data.get("tag_string_copyright", ""))
    tags_artist =  process_tags(data.get("tag_string_artist", ""))
    tags_meta =  process_tags(data.get("tag_string_meta", ""))
    
    quality_tag = ""
    if score > 150:
        quality_tag = "masterpiece, "
    elif 100 <= score <= 150:
        quality_tag = "best quality"
    elif 75 <= score < 100:
        quality_tag = "high quality"
    elif 25 <= score < 75:
        quality_tag = "medium quality"
    elif 0 <= score < 25 :
        quality_tag = "normal quality"
    elif -5 <= score < 0:
        quality_tag = "low quality"
    elif score < -5:
        quality_tag = "worst quality"

    if rating in "q":
        nsfw_tags = "rating: questionable, nsfw"
    elif rating in "e":
        nsfw_tags = "rating: explicit, nsfw"
    elif rating in "s":
        nsfw_tags = "rating: sensitive"
    else:
        nsfw_tags = "rating: general"

    tags_general_list = tags_general.split(', ')
    special_tags = [
        "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls", "multiple girls",
        "1boy", "2boys", "3boys", "4boys", "5boys", "6+boys", "multiple boys", "male focus"
    ]

    found_special_tags = [tag for tag in tags_general_list if tag in special_tags]

    for tag in found_special_tags:
        tags_general_list.remove(tag)

    first_general_tag = ', '.join(found_special_tags)
    rest_general_tags = ', '.join(tags_general_list)

    tags_separator = "|||"
    
    pre_separator_tags = []
    post_separator_tags = []

    if first_general_tag:
        pre_separator_tags.append(first_general_tag)
    if tags_character:
        pre_separator_tags.append(tags_character)
    if tags_copyright:
        pre_separator_tags.append(tags_copyright)
    if tags_artist:
        pre_separator_tags.append(tags_artist)

    if nsfw_tags:
        post_separator_tags.append(nsfw_tags)
    if rest_general_tags:
        post_separator_tags.append(rest_general_tags)
    if year_tag:
        post_separator_tags.append(year_tag)
    if tags_meta:
        post_separator_tags.append(tags_meta)
    if quality_tag:
        post_separator_tags.append(quality_tag)

    pre_separator_str = ', '.join(pre_separator_tags)
    post_separator_str = ', '.join(post_separator_tags)

    caption = f"{pre_separator_str}, {tags_separator} {post_separator_str}"
    
    print(caption)
    print()
    return caption

def process_file(json_folder, json_file):
    with open(f"{json_folder}/{json_file}", "r") as f:
        data = json.load(f)

    extension = data.get("file_ext")
    rating_map = {'g': 'general', 's': 'sensitive', 'q': 'questionable', 'e': 'explicit'}
    rating = rating_map.get(data.get("rating"), "")
    file_url = data.get("file_url")

    if extension not in ["png", "jpg", "jpeg", "webp", "bmp"]:
        return

    tags = generate_tags(data)
    
    if download_image(rating, file_url, json_file.split(".")[0], extension):
        processed_folder = f"{json_folder}_processed"
        os.makedirs(processed_folder, exist_ok=True)
        shutil.move(f"{json_folder}/{json_file}", f"{processed_folder}/{json_file}")
        
    write_metadata(rating, json_file.split(".")[0], tags)

def worker(queue, json_folder, pbar):
    while True:
        json_file = queue.get()
        if json_file is None:
            break  # Exit signal
        process_file(json_folder, json_file)
        queue.task_done()
        pbar.update(1)  # Update progress bar

def main(json_folder):
    global total_files
    total_files = len(os.listdir(json_folder))

    rating_folders = ['general', 'sensitive', 'questionable', 'explicit']

    for folder in rating_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)

    queue = Queue()
    threads = []
    num_worker_threads = 4

    with tqdm(total=total_files, desc="Processing Files") as pbar:
        for _ in range(num_worker_threads):
            t = threading.Thread(target=worker, args=(queue, json_folder, pbar))
            t.start()
            threads.append(t)
        for json_file in os.listdir(json_folder):
            queue.put(json_file)
        queue.join()
        for _ in range(num_worker_threads):
            queue.put(None)
        for t in threads:
            t.join()

    print("Scraping complete.")

if __name__ == "__main__":
    project_path = "data/premodel/animagine-xl-3.1"
    os.makedirs(project_path, exist_ok=True)
    os.chdir(project_path)
    main("/home/congdc/hdd/tool/generative-images/data/premodel/animagine-xl-3.1/logs")