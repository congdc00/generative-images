'''
Format lại dữ liệu có cấu trúc dạng 

- name_yotube/
    - <id_thumb>.png
    - <id_thumb>.txt


'''

import os

import json

folder_path = "data/model-house"  # Thay đổi đường dẫn nếu cần

jsonl_file_path = "data/model-house/metadata.jsonl"
list_folder = os.listdir(folder_path)

with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:

    for folder_chanel in list_folder:
        if folder_chanel.endswith(".zip"):
            continue
        for filename in os.listdir(f"{folder_path}/{folder_chanel}"):
            if filename.endswith(".txt"):
                file_path = os.path.join(f"{folder_path}/{folder_chanel}", filename)
                
                with open(file_path, 'r', encoding='utf-8') as txtfile:
                    content_a = txtfile.read().strip()
                    
                    # Tạo đối tượng JSON
                    name = f"{folder_chanel}/" + filename.replace(".txt", ".jpg")
                    metadata_entry = {"file_name": name, "text": content_a}
                    
                    # Ghi thông tin vào file JSONL
                    jsonl_file.write(json.dumps(metadata_entry, ensure_ascii=False) + '\n')

print(f"File JSONL đã được tạo tại: {jsonl_file_path}")
