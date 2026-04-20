import os
import sys


def dynamic_import(lang):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    module_name = f"search.{lang}"
    module = __import__(module_name, fromlist=['find_code_by_name'])
    return module
