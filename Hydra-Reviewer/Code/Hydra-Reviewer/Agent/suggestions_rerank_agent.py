from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence
from langgraph.graph import END, MessageGraph
from Agent import config

llm = config.get_llm()


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


rerank_prompt = ("Your input is a list of suggestions for a patch. The patch is a collection of diffs for a file "
                 "within a code change. You need to sort these suggestions according to their importance, "
                 "with the priorities as follows:\n1.Fault Tolerance Suggestions (highest priority, for example, "
                 "recommending element checks)\n2.Correctness Of Code Semantic Suggestions (for example, recommending "
                 "changes to the code semantic)\n3.Compatibility Suggestions (for example, recommending adjustments "
                 "to ensure compatibility across different systems, platforms, or environments)\n4.Code Performance "
                 "Suggestions (such as proposing methods to improve efficiency)\n5.Security Compliance Suggestions ("
                 "for example, recommending fixes for potential resource release issues or other security "
                 "vulnerabilities)\n6.Code Comment Quality Suggestions (for example, recommending adding or removing "
                 "comments, or improving comment clarity, completeness, or relevance)\n7.Runtime Observability "
                 "Suggestions (for example, recommending adding or modifying logging or assertions)\n8.Identifier "
                 "Naming Style Suggestions (for example, recommending changes to naming conventions to conform to "
                 "naming style guidelines)\n9.Code Formatting Style Suggestions (for example, recommending changes to "
                 "improve code indentation, spacing, and overall formatting consistency)\n10.Other Suggestions ("
                 "lowest priority)\nPlease reorder the suggestions based on the above priorities and renumber them, "
                 "ensuring that the most critical suggestions are listed first. Do not change the content of the "
                 "suggestions. Your output should be a list of suggestions, and refrain from adding any words.")

rerank_example = '''
Input example：*a list of suggestion*
Correct Output Example: (Assume that the suggestions are sorted in order of priority.)
1.suggestion1
2.suggestion2
3.suggestion3
4.suggestion4
5.suggestion5
'''
rerank_prompt = rerank_prompt + rerank_example
rerank = build_model(rerank_prompt, llm)


def rerank_node(state: Sequence[BaseMessage]):
    return rerank.invoke({"messages": state})


def run_graph(tmp_input):
    builder = MessageGraph()
    builder.add_node("rerank", rerank_node)

    builder.set_entry_point("rerank")
    builder.add_edge("rerank", END)
    graph = builder.compile()

    messages = graph.invoke(tmp_input)
    comment = messages[-1].content
    return comment
