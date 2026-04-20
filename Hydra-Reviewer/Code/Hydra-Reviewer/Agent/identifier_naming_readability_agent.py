from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Sequence
from langgraph.graph import END, MessageGraph
from Agent import prompt_template, config

generate_llm = config.get_generate_llm()
llm = config.get_llm()
reflection_times = config.get_reflection_times()


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
    global generate
    input_introduction = prompt_template.get_input_introduction(function_calling_flag)
    function_calling_prompt = prompt_template.get_function_calling_prompt(function_calling_flag)
    generate_prompt = get_generate_prompt(input_introduction, function_calling_prompt)
    generate = build_model(generate_prompt, generate_llm)


def get_generate_prompt(input_introduction, function_calling_prompt):
    return input_introduction + (f"As a code reviewer focused on the identifier naming readability, you are required "
                                 f"to meticulously review each line of the diff. After understanding the diff, "
                                 f"evaluate whether the naming of variables, functions, classes, and other "
                                 f"identifiers in the changed code is clear and understandable. If you believe "
                                 f"the naming of the identifiers is not clear enough or lacks readability, "
                                 f"output comments and specific suggestions. You must carefully check each line, "
                                 f"with your attention focused solely on the identifier naming readability. "
                                 f"{function_calling_prompt}Only output comments related to identifier naming "
                                 f"readability. If the user provides critique, regenerate the comments based on the "
                                 f"patch and the critique, and then only output the new comments.")


generate = None

requirements = ("As a code reviewer focused on the identifier naming readability, you are required to meticulously "
                "review each line of the diff. After understanding the diff, evaluate whether the naming of "
                "variables, functions, classes, and other identifiers in the changed code is clear and and "
                "understandable. If you believe the naming of the identifiers is not clear enough or lacks "
                "readability, output comments and specific suggestions.")

reflect_prompt = prompt_template.get_reflect(requirements)
reflect = build_model(reflect_prompt, llm)

clean_up_prompt = prompt_template.get_clean_up(requirements)
clean_up = build_model(clean_up_prompt, llm)


def generation_node(state: Sequence[BaseMessage]):
    return generate.invoke({"messages": state})


def reflection_node(messages: Sequence[BaseMessage]) -> HumanMessage:
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    translated = [messages[0]] + [
        cls_map[msg.type](content=msg.content) for msg in messages[1:]
    ]
    translated = [translated[-1]]
    res = reflect.invoke({"messages": translated})
    return HumanMessage(content=res.content)


def clean_up_node(messages: Sequence[BaseMessage]) -> HumanMessage:
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    translated = [messages[0]] + [
        cls_map[msg.type](content=msg.content) for msg in messages[1:]
    ]
    translated = [translated[-2]]
    res = clean_up.invoke({"messages": translated})
    return HumanMessage(content=res.content)


def should_continue(state: List[BaseMessage]):
    reflection = state[-1].content
    if len(state) > reflection_times or "<Good>" in reflection:
        return "clean_up"
    return "generate"


def run_graph(patch):
    builder = MessageGraph()
    builder.add_node("generate", generation_node)
    builder.add_node("reflect", reflection_node)
    builder.add_node("clean_up", clean_up_node)

    builder.set_entry_point("generate")
    builder.add_edge("generate", "reflect")
    builder.add_conditional_edges("reflect", should_continue)
    builder.add_edge("clean_up", END)
    graph = builder.compile()

    messages = graph.invoke(patch)
    comment = messages[-1].content
    return comment
