import json

jsonl_path = r''
json_path = jsonl_path[:-1]
with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file:
    lines = jsonl_file.readlines()

json_array = []

for line in lines:
    json_data = json.loads(line)
    json_array.append(json_data)

with open(json_path, 'w', encoding='utf-8') as json_file:
    json.dump(json_array, json_file, indent=4, ensure_ascii=False)
