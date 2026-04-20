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
    global summarize
    summarize_prompt = prompt_template.get_summarize_prompt(function_calling_flag)
    summarize = build_model(summarize_prompt, llm)


summarize = None


def summary_node(state: Sequence[BaseMessage]):
    return summarize.invoke({"messages": state})


def run_graph(tmp_input):
    builder = MessageGraph()
    builder.add_node("summarize", summary_node)

    builder.set_entry_point("summarize")
    builder.add_edge("summarize", END)
    graph = builder.compile()

    messages = graph.invoke(tmp_input)
    comment = messages[-1].content
    return comment
