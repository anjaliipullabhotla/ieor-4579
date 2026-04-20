import os
import json
import sys

from openai import OpenAI
from tree_sitter import Language, Parser


def dynamic_import(lang):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    module_name = f"search.{lang}"
    module = __import__(module_name, fromlist=['find_code_by_name'])
    return module


def extract_pure_identifier_name(full_name):
    last_part = full_name.split('(')[0]
    name = last_part.split('.')[-1]
    return name


def get_additional_information(tmp_data):
    patch = tmp_data['patch']
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    OPENAI_API_BASE = os.environ["OPENAI_API_BASE"]
    GPT_MODEL = os.environ["OPENAI_GPT_MODEL"]
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "return_name",
                "description": '''Return the name of a function, method, class.''',
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of a function, method, class.",
                        },
                    },
                    "required": ["name"],
                },
            }
        },
    ]

    user_prompt = ("Your input is a patch, which consists of all the diffs for a single file within a single code "
                   "change. Assume you are conducting a code review of this patch. Please list the names of the most "
                   "specific and independent functions, methods or classes that you have questions about or do not "
                   "understand, stripping all class, module, or namespace prefixes, ranked in order of your lack of "
                   "understanding—starting with the one you understand the least. For example, instead of outputting "
                   "'utils.calculateSum' or 'MathOperations.add', simply output 'calculateSum' or 'add'. Avoid "
                   "including names of any higher-level or encompassing methods or functions unless they themselves "
                   "are unclear. If you have no questions about the functions, methods, or classes in this patch, "
                   "then output nothing.")

    input_patch = "Patch:\n" + patch
    completion = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "user", "content": user_prompt},
            {"role": "user", "content": input_patch},
        ],
        tools=tools,
    )
    tool_calls_info = []
    identifier_name_list = []
    if completion.choices[0].message.tool_calls is not None:
        tool_calls_info = completion.choices[0].message.tool_calls
    print("----------------------------------------------------------")
    for tool_call in tool_calls_info:
        identifier_name = json.loads(tool_call.function.arguments)['name']
        identifier_name = extract_pure_identifier_name(identifier_name)
        identifier_name_list.append(identifier_name)
        print(identifier_name)
    print("----------------------------------------------------------")
    source_code = tmp_data['current_file']

    lang = tmp_data['lang']
    if lang == 'objective-c':
        tmp_lang = 'objc'
    elif lang == 'csharp':
        tmp_lang = 'c_sharp'
    else:
        tmp_lang = lang

    current_directory = os.path.dirname(__file__)

    build_directory = os.path.join(current_directory, '..', 'build', 'my-languages.so')

    LANGUAGE = Language(build_directory, tmp_lang)

    parser = Parser()
    parser.set_language(LANGUAGE)

    tree = parser.parse(
        bytes(source_code, "utf8")
    )
    additional_informational = ""

    lang_module = dynamic_import(lang)
    find_code_by_name = lang_module.find_code_by_name

    for identifier_name in identifier_name_list:
        code_snippet = find_code_by_name(tree.root_node, identifier_name)
        if code_snippet is not None:
            if additional_informational == "":
                additional_informational += "{additional information}:"
            additional_informational += "\n" + code_snippet + "\n"

    return "{Patch}:\n" + patch + "\n\n" + additional_informational
