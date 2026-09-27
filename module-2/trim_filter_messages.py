from dotenv import load_dotenv
from langchain_core.messages import (AIMessage, HumanMessage, RemoveMessage,
                                     trim_messages)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()

msgs = [AIMessage(content="Hello", name="model")]
msgs.append(HumanMessage("Share about sea oaters.", name="Vishal"))

# for msg in msgs:
#     msg.pretty_print()

# We can pass them to a chat model.
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
# response = llm.invoke(msgs)
# print(response)


# We can execute the chat model in a simple graph
def chat_model_node(state: MessagesState):
    return {"messages": llm.invoke(state["messages"])}


# build graph
graph = StateGraph(MessagesState)

graph.add_node("chat", chat_model_node)

graph.add_edge(START, "chat")
graph.add_edge("chat", END)

graph = graph.compile()

# response = graph.invoke({"messages": msgs})
# for msg in response["messages"]:
#     msg.pretty_print()

# filter the messages which are sent to the llm


def filter_messages(state: MessagesState):
    delete_messages = [RemoveMessage(id=msg.id) for msg in state["messages"][:-2]]
    return {"messages": delete_messages}


def chat_model(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(MessagesState)

builder.add_node("filter", filter_messages)
builder.add_node("chat", chat_model)

builder.add_edge(START, "filter")
builder.add_edge("filter", "chat")
builder.add_edge("chat", END)

graph = builder.compile()

msgs = [AIMessage("hello", name="Model", id="1")]
msgs.append(HumanMessage("gello", name="Vishal", id="2"))
msgs.append(AIMessage("So, you were researching about something", id="3", name="Model"))
msgs.append(
    HumanMessage("Share about something anything in 50 words.", name="Visahl", id="4")
)

response = graph.invoke({"messages": msgs})

# for msg in response["messages"]:
#     msg.pretty_print()


# instead of filtering messages other approach is to pass only the messages to the model


def chat_model_node(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"][-1:])]}


builder = StateGraph(MessagesState)

builder.add_node("chat", chat_model_node)

builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile()

msgs.append(response["messages"][-1])
msgs.append(
    HumanMessage(
        f"Tell me more about the first that comes on your neural network without filtering ",
        name="Vishal",
    )
)

# for msg in msgs:
#     msg.pretty_print()

response = graph.invoke({"messages": msgs})
# for msg in response['messages']:
#     msg.pretty_print()


def trim_chat_model(state: MessagesState):
    msgs = trim_messages(
        state["messages"],
        max_tokens=100,
        strategy="last",
        token_counter=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite"),
        allow_partial=True,
    )
    return {"messages": [llm.invoke(msgs)]}


builder = StateGraph(MessagesState)

builder.add_node("trim_chat_model", trim_chat_model)

builder.add_edge(START, "trim_chat_model")
builder.add_edge("trim_chat_model", END)

graph = builder.compile()

msgs.append(response["messages"][-1])
msgs.append(
    HumanMessage(f"Share the first thing that comes on generation.", name="Vishal")
)

trim_response = graph.invoke({"messages": msgs})
# print(trim_response)
for msg in trim_response["messages"]:
    msg.pretty_print()
