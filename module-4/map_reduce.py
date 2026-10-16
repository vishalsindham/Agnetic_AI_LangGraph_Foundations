import operator
import warnings
from typing import Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel
from typing_extensions import TypedDict
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")

topics_prompt = """Generate a list of sub-topics that are all related to this overall subject : {subject}"""
joke_prompt = """Generate a joke about topic : {topic}"""
best_joke_prompt = """Below are a bunch of jokes about {subject}. Select the best one! Return the ID of the best one, starting 0 as the ID for the first joke. Joke : \n\n {jokes}"""

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.7)


class Topics(BaseModel):
    topics: list[str]


class BestJoke(BaseModel):
    id: int


class OverallState(TypedDict):
    subject: str
    topics: list
    jokes: Annotated[list, operator.add]
    best_selected_joke: str


def generate_topics(state: OverallState):
    prompt = topics_prompt.format(subject=state["subject"])
    response = model.with_structured_output(Topics).invoke(prompt)
    return {"topics": response.topics}


def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"topic": topic}) for topic in state["topics"]]


class JokeState(TypedDict):
    topic: str


class Joke(BaseModel):
    joke: str


def generate_joke(state: JokeState):
    prompt = joke_prompt.format(topic=state["topic"])
    response = model.with_structured_output(Joke).invoke(prompt)
    return {"jokes": [response.joke]}


def best_joke(state: OverallState):
    jokes = "\n\n".join(state["jokes"])
    prompt = best_joke_prompt.format(subject=state["subject"], jokes=jokes)
    response = model.with_structured_output(BestJoke).invoke(prompt)
    return {"best_selected_joke": state["jokes"][response.id]}


builder = StateGraph(OverallState)

builder.add_node("generate_topics", generate_topics)
builder.add_node("generate_joke", generate_joke)
builder.add_node("best_joke", best_joke)

builder.add_edge(START, "generate_topics")
builder.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
builder.add_edge("generate_joke", "best_joke")
builder.add_edge("best_joke", END)

workflow = builder.compile()

response = workflow.invoke({"subject": "Animals"})

print(f" Subject : {response['subject']}", end ="\n\n")
print(f"Topics of the subject : ", end ="\n\n")
for topic in response['topics']:
    print(topic)

print("", end ="\n\n")
print(f"Jokes :", end ="\n\n")
for joke in response['jokes']:
    print(joke)
print("", end ="\n\n")

print(f"Best Joke : {response['best_selected_joke']}")