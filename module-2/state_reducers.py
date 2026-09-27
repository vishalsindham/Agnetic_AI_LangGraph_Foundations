from operator import add
from typing import Annotated

from langchain_core.messages import (AIMessage, AnyMessage, HumanMessage,
                                     RemoveMessage)
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    foo: int


def node_1(state):
    print(" --- node 1 --- ")
    return {"foo": state["foo"] + 1}


# builder = StateGraph(State)

# builder.add_node("node_1", node_1)

# builder.add_edge(START, "node_1")
# builder.add_edge("node_1", END)

# graph = builder.compile()

# overwrite_response = graph.invoke({"foo": 15})
# print(overwrite_response)

# The above case was when there is a single node which will update or nodes are in sequence.
# Let's obeserve a case when there are parallel nodes updating to the same key/channel
# where reducer is not defined to handle the updates.


def node_2(state):
    print("node 2")
    return {"foo": state["foo"] + 1}


def node_3(state):
    print("node 3")
    return {"foo": state["foo"] + 1}


# build the graph
builder = StateGraph(State)

builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)
builder.add_edge("node_1", "node_3")

graph = builder.compile()

# the invoke should throw a error, as it doesn't know to handle updates from parallel nodes.
# try:
#     response = graph.invoke({"foo": 4})
#     print(response)
# except InvalidUpdateError as e:
#     print(f"InvalidUpdateError occured : {e}")


# Building a graph which uses reducer to upate the graph state


class State(TypedDict):
    foo: Annotated[list[int], add]


def node_1(state):
    print(" -- Node 1 -- ")
    return {"foo": [state["foo"][-1] + 1]}


# builder = StateGraph(State)

# builder.add_node("node_1", node_1)

# builder.add_edge(START, "node_1")
# builder.add_edge("node_1", END)

# graph = builder.compile()

# response = graph.invoke({"foo": [6]})
# print(response)


def node_2(state):
    print(" -- Node 2 --")
    return {"foo": [state["foo"][-1] + 1]}


def node_3(state):
    print(" -- Node 3 -- ")
    return {"foo": [state["foo"][-1] + 1]}


builder = StateGraph(State)

builder.add_node("one", node_1)
builder.add_node("two", node_2)
builder.add_node("three", node_3)

builder.add_edge(START, "one")
builder.add_edge("one", "two")
builder.add_edge("one", "three")
builder.add_edge("two", END)
builder.add_edge("three", END)

graph = builder.compile()

# response = graph.invoke({"foo": [4]})
# print(response)

# As we used a built in function as a reducer it has limitiation, if we need a particulat functionality
# we have to write custom function which can be used as reducer

# Here if we pass None as a argument it will throw error as it doesn't know how to hanlde it.
# try:
#     graph.invoke({"foo": None})
# except TypeError as e:
#     print(f"TypeError occured : {e}")

# to address this case let's write a custom reducer


def reduce_list(left: list | None, right: list | None) -> list:
    """Safely combine two lists, handling cases where either or both inputs might be None.

    Args:
        left (list | None) : The first list to combine, or None.
        right (list | None) : The second list to combine, or None.

    Returns:
        list: A new list containing all elements from both input lists.
                If an input is None, it's treadted as an empty list.
    """

    if not left:
        left = []
    if not right:
        right = []
    return left + right


class DefaultState(TypedDict):
    foo: Annotated[list[int], add]


class CustomReducerState(TypedDict):
    foo: Annotated[list[int], reduce_list]


def node_1(state):
    print(" -- Node 1 -- ")
    return {"foo": [2]}


# build
builder = StateGraph(DefaultState)

builder.add_node("node_1", node_1)

builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

graph = builder.compile()
# try:
#     print(graph.invoke({"foo": None}))
# except TypeError as e:
#     print(f"TypeError occured : {e}")

# build the graph with custom reducer
builder = StateGraph(CustomReducerState)

builder.add_node("node_1", node_1)

builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

graph = builder.compile()

# try:
#     print(graph.invoke({"foo" : None}))
# except TypeError as e:
#     print(f"TypeError occured : {e}")


# A custom TypeDict that includes a list of messages with add_messages reducer
class CustomMsgsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    added_key_1: str
    added_key_2: str


# Use MessagesState and extend the class
class ExtendedMsgsState(MessagesState):
    added_key_1: str
    added_key_2: str


initial_msgs = [AIMessage("hello", id="1"), HumanMessage("hello", id="2")]


new_msg = AIMessage("This is a new conversation.", id="3")

# print(f"New list of msgs :  {add_messages(initial_msgs, new_msg)}")

# rewrite the existing message

updated_msg = HumanMessage("Is this a new conversation", id="2")
# print(
#     f" The message with the matching id is updated not appended to the list { add_messages(initial_msgs, updated_msg)}"
# )

delete_msgs = [RemoveMessage(id=msg.id) for msg in initial_msgs[1:]]

print(f" After deleting the messages : {add_messages(initial_msgs, delete_msgs)}")
