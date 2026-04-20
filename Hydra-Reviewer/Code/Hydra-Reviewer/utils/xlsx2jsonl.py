import pandas as pd
import json
import os

# Excel文件路径
xlsx_path = r''

# 移除文件扩展名，准备JSON Lines文件路径
base_path, _ = os.path.splitext(xlsx_path)
jsonl_path = base_path + '.jsonl'
# jsonl_path = r'C:\Users\keke\Desktop\数据集处理\variant2\ablation_400_variant2_with_flag.jsonl'

# 使用pandas读取Excel文件
df = pd.read_excel(xlsx_path)

# 打开JSON Lines文件准备写入
with open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
    for index, row in df.iterrows():
        # 将每行数据转换为字典
        json_data = row.to_dict()
        # 将字典转换为JSON格式的字符串
        json_str = json.dumps(json_data, ensure_ascii=False)
        # 写入文件，并添加换行符以符合JSON Lines格式
        jsonl_file.write(json_str + '\n')

print("JSON Lines 文件已生成")
