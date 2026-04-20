# 函数：根据名称查找节点并返回代码片段
# 解析代码并查找函数定义
def find_code_by_name(node, name):
    # 查找函数、方法或类定义
    if node.type in ['function_definition', 'class_definition']:
        identifiers = [child for child in node.children]
        for identifier in identifiers:
            if identifier.text.decode('utf8') == name:
                return node.text.decode('utf8')

    # 递归查找所有子节点
    for child in node.children:
        result = find_code_by_name(child, name)
        if result:
            return result

    return None


