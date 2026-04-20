import json


def read_jsonl_file(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 解析每一行的JSON数据并添加到列表中
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
    return data


def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 直接从文件中加载JSON数据
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except json.JSONDecodeError:
        print(f"JSON解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None


def save_jsonl_file(file_path, save_data):
    with open(file_path, 'a', encoding='utf-8') as jsonl_file:
        json.dump(save_data, jsonl_file, ensure_ascii=False)
        jsonl_file.write('\n')  # 写入换行符表示一个 JSON 对象的结束
    # print("save success")
