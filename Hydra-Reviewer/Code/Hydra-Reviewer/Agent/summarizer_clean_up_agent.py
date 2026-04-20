from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence
from langgraph.graph import END, MessageGraph
from Agent import config, prompt_template

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


def using_function_calling(function_calling_flag):
    global clean_up
    clean_up_prompt = prompt_template.get_clean_up_prompt(function_calling_flag)
    clean_up = build_model(clean_up_prompt, llm)


clean_up = None


def clean_up_node(state: Sequence[BaseMessage]):
    return clean_up.invoke({"messages": state})


def run_graph(tmp_input):
    builder = MessageGraph()
    builder.add_node("clean_up", clean_up_node)

    builder.set_entry_point("clean_up")
    builder.add_edge("clean_up", END)
    graph = builder.compile()

    messages = graph.invoke(tmp_input)
    comment = messages[-1].content
    return comment
