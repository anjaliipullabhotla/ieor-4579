import os
import json

import pandas as pd

jsonl_path = r''
# 移除文件扩展名
base_path, _ = os.path.splitext(jsonl_path)
xlsx_path = base_path + '.xlsx'

with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file:
    lines = jsonl_file.readlines()

datas = []

for line in lines:
    json_data = json.loads(line)
    datas.append(json_data)

df = pd.DataFrame(datas)
df.to_excel(xlsx_path, index=False)

print("Excel 表格已生成")
