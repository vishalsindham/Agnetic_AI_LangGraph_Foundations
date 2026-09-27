import sqlite3

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from typing_extensions import Literal

load_dotenv()

db_path = "state_db/example.db"
connection = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(connection)

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")


class State(MessagesState):
    summary: str


def call_model(state: State):

    summary = state.get("summary", "")

    if summary:

        system_msg = SystemMessage(
            content=f"Summary of the earlier conversation : {summary}"
        )

        msgs = [system_msg] + state["messages"]
    else:
        msgs = state["messages"]

    response = model.invoke(msgs)
    return {"messages": response}


def summarize_conversation(state: State):

    summary = state.get("summary", "")

    if summary:

        summary_msg = (
            f"Here is the summary of the conversation to date : {summary} \n\n"
            "Extend the summary by taking into account the new messages above"
        )
    else:
        summary_msg = "Create a summary of the conversation above :"

    msgs = state["messages"] + [HumanMessage(content=summary_msg)]
    response = model.invoke(msgs)

    delete_msgs = [RemoveMessage(id=msg.id) for msg in state["messages"][:-2]]
    return {"summary": response.content[0]["text"], "messages": delete_msgs}


def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Return the next node top execute."""

    messages = state["messages"]

    if len(messages) > 4:
        return "summarize_conversation"
    return END


builder = StateGraph(State)

builder.add_node(summarize_conversation)
builder.add_node("chat", call_model)

builder.add_edge(START, "chat")
builder.add_conditional_edges("chat", should_continue)
builder.add_edge("summarize_conversation", END)

graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

input_msg = HumanMessage(content="Hi, Vishal here.")
response = graph.invoke({"messages": [input_msg]}, config)

input_msg = HumanMessage(content="What's my name..?")
response = graph.invoke({"messages": [input_msg]}, config)

input_msg = HumanMessage(content="Langchain and it's saw")
response = graph.invoke({"messages": [input_msg]}, config)


for msg in response["messages"]:
    msg.pretty_print()


graph_state = graph.get_state(config)
print(graph_state)
