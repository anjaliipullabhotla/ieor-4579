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

end2end_prompt = ("As a code reviewer, imagine you are conducting a code review, and your team leader requests you to "
                  "provide a review comment on a patch. The patch is a collection of diffs for a file within a code "
                  "change. Please generate a review comment based on the patch. Your output should be a list of "
                  "suggestions and refrain from adding meaningless words.")
output_example = '''
Correct Output Example:
1.suggestion1
2.suggestion2
3.suggestion3
4.suggestion4
5.suggestion5
...
'''
end2end_prompt += output_example
end2end_gpt = build_model(end2end_prompt, llm)


def end2end_gpt_node(state: Sequence[BaseMessage]):
    return end2end_gpt.invoke({"messages": state})


def run_graph(tmp_input):
    builder = MessageGraph()
    builder.add_node("end2end_gpt", end2end_gpt_node)

    builder.set_entry_point("end2end_gpt")
    builder.add_edge("end2end_gpt", END)
    graph = builder.compile()

    messages = graph.invoke(tmp_input)
    comment = messages[-1].content
    return comment
