import random
from dataclasses import dataclass
from typing import Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError, field_validator
from typing_extensions import TypedDict


# A schema defined using TypedDict class
class TypedDictClass(TypedDict):
    foo: str
    bar: str


class TypedDictState(TypedDict):
    name: str
    mood: Literal["content", "curious"]


@dataclass
class DataclassState:
    name: str
    mood: Literal["relaxed mode", "lazy mood"]


class PydanticState(BaseModel):
    name: str
    mood: str

    @field_validator("mood")
    @classmethod
    def validate_mood(cls, value):
        # validate mood value
        if value not in ["curious", "relaxed"]:
            raise ValueError(
                "Each mood must be either 'curious', 'here in the present mood'"
            )
        return value


try:
    state = PydanticState(name="Vishal", mood="curious")
except ValidationError as e:
    print("Validation Error: ", e)

# We can define state schema using dataclass or typeddict, it is validated by type checkers only
# run time the values are not validated. Using Pydantic to define the schema we can validate the values


# # Access the state values using key value format when the state is defined using TypedDict
# def node_1(state):
#     print(" --- Node 1 ---")
#     return {"name": state["name"] + " is in a "}

# Access the state value using dot operator when the state is defined using dataclass decorator


def node_1(state):
    print(" -- Node 1 --")
    return {"name": state.name + " is in a "}


def node_2(state):
    print(" --- Node 2 ---")
    return {"mood": "relaxed"}


def node_3(state):
    print(" --- Node 3 ---")
    return {"mood": "curious"}


def get_mood(state) -> Literal["node_2", "node_3"]:

    if random.random() < 0.5:

        return "node_2"

    return "node_3"


# Build the graph

# builder = StateGraph(TypedDictState)
# builder = StateGraph(DataclassState)
builder = StateGraph(PydanticState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# execution flow of the graph
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", get_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# compile the graph
graph = builder.compile()

# response = graph.invoke({"name": "Vishal"})
response = graph.invoke(PydanticState(name="Vishal", mood="curious"))

print(response)
