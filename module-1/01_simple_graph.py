import random
from typing import Literal

from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class State(TypedDict):
    graph_state: str


def node_1(state):
    print("--- Node 1 ---")
    return {"graph_state": state["graph_state"] + " I am"}


def node_2(state):
    print("--- Node 2 ---")
    return {"graph_state": state["graph_state"] + " happy!"}


def node_3(state):
    print("--- Node 3 ---")
    return {"graph_state": state["graph_state"] + " content!"}


def decide_mood(state) -> Literal["second", "third"]:

    user_input = state["graph_state"]

    if random.random() < 0.5:

        return "second"

    return "third"


builder = StateGraph(State)
builder.add_node("first", node_1)
builder.add_node("second", node_2)
builder.add_node("third", node_3)

builder.add_edge(START, "first")
builder.add_conditional_edges("first", decide_mood)
builder.add_edge("second", END)
builder.add_edge("third", END)

graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path="simple_graph.png")

response = graph.invoke({"graph_state": "Hi, this is Vishal. "})

print(response)
