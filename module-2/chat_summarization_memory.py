from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from typing_extensions import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


class State(MessagesState):
    summary: str


def call_model(state: State):

    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:

        system_msg = f"Summary of the conversation up to know : {summary}"

        msgs = [SystemMessage(content=system_msg)] + state["messages"]
    else:
        msgs = state["messages"]
    response = model.invoke(msgs)
    return {"messages": response}


def summarize_conversation(state: State):

    summary = state.get("summary", "")

    if summary:

        summary_msg = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above."
        )
    else:
        summary_msg = "create a summary of the conversation above."

    msgs = state["messages"] + [HumanMessage(summary_msg)]
    response = model.invoke(msgs)

    delete_msgs = [RemoveMessage(id=msg.id) for msg in state["messages"][-2:]]

    return {"summary": response.content[0]["text"], "messages": delete_msgs}


def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Return the next node to execute"""

    msgs = state["messages"]

    if len(msgs) > 4:
        return "summarize_conversation"

    return END


builder = StateGraph(State)

builder.add_node(summarize_conversation)
builder.add_node("chat", call_model)

builder.add_edge(START, "chat")
builder.add_conditional_edges("chat", should_continue)
builder.add_edge("summarize_conversation", END)

memory = MemorySaver()

workflow = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

input_msg = HumanMessage(content="Hello, Vishal here..")
response = workflow.invoke({"messages": [input_msg]}, config)

for msg in response["messages"][-1:]:
    print(msg.content[-1]["text"])

input_msg = HumanMessage(
    content="Share my name in different  English alphabets yet the pronouncation should be same"
)
response = workflow.invoke({"messages": [input_msg]}, config)

for msg in response["messages"][-1:]:
    print(msg.content[-1]["text"])

input_msg = HumanMessage(
    content="Make a list as per the number of letters in each word."
)
response = workflow.invoke({"messages": [input_msg]}, config)

for msg in response["messages"][-1:]:
    print(msg.content[-1]["text"])

print(workflow.get_state(config).values.get("summary", ""))


for msg in response["messages"]:
    msg.pretty_print()
