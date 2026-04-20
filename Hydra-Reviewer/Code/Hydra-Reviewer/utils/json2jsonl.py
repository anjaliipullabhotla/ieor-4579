import json

file_path = r''
save_path = file_path + 'l'
# 读取 JSON 文件
with open(file_path, 'r', encoding='utf-8') as json_file:
    datas = json.load(json_file)

# 转换为 JSONL 并写入新文件
with open(save_path, 'w',
          encoding='utf-8') as jsonl_file:
    for data in datas:
        json_string = json.dumps(data, ensure_ascii=False)
        jsonl_file.write(json_string + '\n')
