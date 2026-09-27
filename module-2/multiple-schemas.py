from dotenv import load_dotenv
from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class OverallState(TypedDict):
    foo: int


class PrivateState(TypedDict):
    baz: int


def node_1(state: OverallState) -> PrivateState:
    print(" --- node 1 --- ")
    return {"baz": state["foo"] + 1}


def node_2(state: PrivateState) -> OverallState:
    print(" --- node 2 --- ")
    return {"foo": state["baz"] + 1}


# build the graph
builder = StateGraph(OverallState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)

# logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

# compile the graph
graph = builder.compile()

print("Output of the graph for foo and baz :", graph.invoke({"foo": 3}))

# multiple schemas


class OverallState(TypedDict):
    question: str
    answer: str
    notes: str


def thinking_island(state: OverallState):
    return {"answer": "bye", "notes": " .. his name is Vishal"}


def answer_node(state: OverallState):
    return {"answer": "Bye Vishal"}


graph = StateGraph(OverallState)

graph.add_node("node_1", thinking_island)
graph.add_node("node_2", answer_node)

graph.add_edge(START, "node_1")
graph.add_edge("node_1", "node_2")
graph.add_edge("node_2", END)

graph = graph.compile()

print("Overall state of the graph is the output : ", graph.invoke({"question": "hi"}))


class InputState(TypedDict):
    question: str


class OutputState(TypedDict):
    answer: str


class OverallState(TypedDict):
    question: str
    answer: str
    notes: str


def thinking_node(state: InputState):
    return {"answer": "bye", "notes": "... his name is Vishal"}


def answer_node(state: OverallState) -> OutputState:
    return {"answer": "bye Vishal."}


graph = StateGraph(
    state_schema=OverallState, input_schema=InputState, output_schema=OutputState
)

graph.add_node("thinking_node", thinking_node)
graph.add_node("answer_node", answer_node)

graph.add_edge(START, "thinking_node")
graph.add_edge("thinking_node", "answer_node")
graph.add_edge("answer_node", END)

graph = graph.compile()

print("Only the output is printed : ", graph.invoke({"question": "hi"}))
