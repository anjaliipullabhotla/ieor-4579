import os

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence

from langgraph.graph import END, MessageGraph

from Agent import config


def build_model(tmp_prompt, tmp_model):
    tmp_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system", tmp_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    model = tmp_prompt_template | tmp_model
    return model


llm = config.get_llm()

comprehensive_prompt = ("As a code reviewer, imagine you are conducting a code review, and your team leader requests "
                        "you to provide a review comment on a patch from different dimensions. The patch is a "
                        "collection of diffs for a file within a code change. Please generate a review comment based "
                        "on the patch from different dimensions. Below are the names and definitions of all the "
                        "dimensions: \n")

code_semantic_correctness = ("Code Semantic Correctness: You should evaluate whether there are logical errors in the "
                             "code change or if the functionality has not been implemented correctly. If you believe "
                             "there are logical errors or the functionality has not been correctly implemented in the "
                             "code change, output comments and specific suggestions.\n")

code_syntax_correctness = ("Code Syntax Correctness: You should evaluate whether there are any obvious, potential or "
                           "minor code syntax errors in this code change. If you think there is a code syntax error, "
                           "output comments and specific suggestions.\n")

security_compliance = ("Security Compliance: You should evaluate whether this code change introduces system security "
                       "issues such as memory leaks, data overflows. If you think there is a security issue, "
                       "output comments and specific suggestions.\n")

programming_handling_conventions = ("Programming Handling Conventions: You should evaluate whether the code change "
                                    "complies with the handling conventions, that is, whether the coding practices "
                                    "conform to best practices. If you think the code change does not comply with the "
                                    "programming handling conventions, output comments and specific suggestions.\n")

identifier_naming_style = ("Identifier Naming Style: You should evaluate whether identifiers such as parameters, "
                           "variables, functions, classes, and so on in the changed code have consistent and good "
                           "naming styles, such as consistent naming forms, naming methods, case, and so on. If you "
                           "think the identifier naming style of the changed code is inconsistent or not good enough, "
                           "output comments and specific suggestions.\n")

code_formatting_style = ("Code Formatting Style: You should evaluate whether the modified code maintains good "
                         "formatting style. This evaluation should include, but is not limited to, indentation, "
                         "line wrapping, limiting the number of characters per line, and removing unnecessary "
                         "parentheses. If you find any issues with the format style, or other related problems in the "
                         "modified code, output comments and specific suggestions.\n")

comment_style = ("Comment Style: You should evaluate whether the format, application scenarios and content of comments "
                 "in the changed code follow a good and consistent style, including whether comment indentation is "
                 "consistent, aligned, correctly placed next to the relevant code block, and whether the form of "
                 "comments on entities such as classes or functions is appropriate. If you think the comment style of "
                 "the changed code is not good enough, output comments and specific suggestions.\n")

identifier_naming_readability = ("Identifier Naming Readability: You should evaluate whether the naming of variables, "
                                 "functions, classes, and other identifiers in the changed code is clear and "
                                 "understandable. If you believe the naming of the identifiers is not clear enough or "
                                 "lacks readability, output comments and specific suggestions.\n")

code_logic_readability = ("Code Logic Readability: You should evaluate whether the modified code can be simplified or "
                          "its logic readability enhanced. If you believe that the code can be simplified or its "
                          "logic readability can be enhanced, output comments and specific suggestions.\n")

comment_quality = ("Comment Quality: You should evaluate whether the changed code needs comments added for code "
                   "segments or parameter names, whether there are inconsistencies between the comments and the "
                   "corresponding code, or whether the comments are too simple and need to be supplemented, deleted, "
                   "or modified to enhance readability. If you believe that the changed code has issues with the "
                   "quality of its comments that require adding, supplementing, deleting, or modifying comments, "
                   "output comments and specific suggestions.\n")

redundancy = ("Redundancy: You should evaluate whether this code change includes any unnecessary repetitive, "
              "superfluous or mergeable codes or symbols, or unused code or superfluous code logic. If you believe "
              "the modified code has redundancies, output comments and specific suggestions.\n")

compatibility = ("Compatibility: You should evaluate whether this code change remains compatible with the existing "
                 "system, components, or API versions, and whether it introduces conflicts with old code or "
                 "dependency libraries. If you believe there are compatibility issues with this code change, "
                 "output comments and specific suggestions.\n")

name_and_logic_consistency = ("Name And Logic Consistency: You should evaluate whether the logic of the code segments "
                              "corresponding to functions, classes, or files in the modified code is consistent with "
                              "the naming of the code components, or whether there are common parts within the code "
                              "components that can be moved to a general component. Additionally, indicate the "
                              "appropriate adjustments when the naming and logic are inconsistent, such as relocating "
                              "a code segment to a more suitable function, class, or file. If you believe there are "
                              "naming and logic inconsistency issues with this code change, output comments and "
                              "specific suggestions.\n")

runtime_observability = ("Runtime Observability: You should evaluate whether the code change requires adding or "
                         "modifying mechanisms such as logging or assertions to help developers better trace and "
                         "understand the system's behavior during execution. If you believe it is necessary to add or "
                         "modify existing mechanisms such as logging or assertions, output comments and specific "
                         "suggestions.\n")

fault_tolerance = ("Fault Tolerance: You should evaluate whether the modified code might have issues with fault "
                   "tolerance, such as necessary parameters not being checked for null values, abnormal page data, "
                   "and whether it can handle various types of input. If you believe that the fault tolerance of this "
                   "code change is not adequate, output comments and specific suggestions.\n")

code_testing = ("Code Testing: You should evaluate whether the changed code needs to have tests added or whether the "
                "tests need to be adjusted. If you think you need to add tests for this code change or if the tests "
                "need to be adjusted, output comments and specific suggestions.\n")

extensibility = ("Extensibility: You should evaluate whether the modified code has good extensibility to accommodate "
                 "potential future changes in functionality or environment. If you believe that the extensibility of "
                 "the modified code is not adequate, output comments and specific suggestions.\n")

performance = ("Performance: You should evaluate whether there are potential points for performance optimization in "
               "the modified code that could enhance its operational efficiency, such as memory usage, "
               "algorithm efficiency, response time, and runtime efficiency. If you believe that there are potential "
               "points for performance optimization, output comments and specific suggestions.\n")

dimension_list = [code_semantic_correctness, code_syntax_correctness, security_compliance,
                  programming_handling_conventions, identifier_naming_style, code_formatting_style, comment_style,
                  identifier_naming_readability, code_logic_readability, comment_quality, redundancy, compatibility,
                  name_and_logic_consistency, runtime_observability, fault_tolerance, code_testing, extensibility,
                  performance]

for dimension in dimension_list:
    comprehensive_prompt += dimension

comprehensive_prompt += ("You are required to review the code changes from different dimensions. Note that it is not "
                         "necessary to provide comments for every dimension—only offer suggestions in the dimensions "
                         "where you identify potential issues in the code changes. Your output should be a list of "
                         "suggestions and refrain from adding meaningless words.")

output_example = '''
Correct Output Example:
1.suggestion1
2.suggestion2
3.suggestion3
4.suggestion4
5.suggestion5
......
'''

comprehensive_prompt += output_example
# print(comprehensive_prompt)

comprehensive_gpt = build_model(comprehensive_prompt, llm)


def comprehensive_gpt_node(state: Sequence[BaseMessage]):
    return comprehensive_gpt.invoke({"messages": state})


def run_graph(tmp_input):
    builder = MessageGraph()
    builder.add_node("comprehensive_gpt", comprehensive_gpt_node)

    builder.set_entry_point("comprehensive_gpt")
    builder.add_edge("comprehensive_gpt", END)
    graph = builder.compile()

    messages = graph.invoke(tmp_input)
    comment = messages[-1].content
    return comment
