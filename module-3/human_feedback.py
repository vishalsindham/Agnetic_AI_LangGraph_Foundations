import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph_sdk import get_client

load_dotenv()


def multiply(a: int, b: int) -> int:
    """Multiplies the arguments a and b and returns the output"""

    return a * b


def add(a: int, b: int) -> int:
    """Adds the inputs a and b and returns the output."""

    return a + b


def divide(a: int, b: int) -> float:
    """Divides a and b and returns the output"""

    return a / b


tools = [multiply, add, divide]
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# builder = StateGraph(MessagesState)

# builder.add_node("assistant", assistant)
# builder.add_node("tools", ToolNode(tools))

# builder.add_edge(START, "assistant")
# builder.add_conditional_edges("assistant", tools_condition)
# builder.add_edge("tools", "assistant")

# memory = MemorySaver()

# graph = builder.compile(interrupt_before=["assistant"], checkpointer=memory)

# Let's execute the graph

# initial_input = {"messages": "Multiply 4 and 5"}

# thread = {"configurable": {"thread_id": "1"}}

# for event in graph.stream(initial_input, thread, stream_mode="values"):
#     event["messages"][-1].pretty_print()

# state = graph.get_state(thread)
# print(state)

# print(f"\n\n\nHere we will access the state and update it")
# graph.update_state(thread, {"messages": [HumanMessage(content="Multiply 7 and 9.?")]})

# updated_state = graph.get_state(thread).values

# for msg in updated_state["messages"]:
#     msg.pretty_print()

# print(
#     f"\n\n\nFrom here we will invoke the graph to proceed using the updated state by passing None as input"
# )

# for event in graph.stream(None, thread, stream_mode="values"):
#     event["messages"][-1].pretty_print()

# print(
#     f"\n\n\n Now we are back at the assistant node, we will again invoke the graph using None to proceed."
# )

# for event in graph.stream(None, thread, stream_mode="values"):
#     event["messages"][-1].pretty_print()

# URL = "http://127.0.0.1:2024"

# client = get_client(url=URL)

# print(
#     f"\n\n Now we will work with LangGraph API. Feature of LangGraph is that we can add interrupts by passing it as part of invocation"
# )


# async def async_invocation():
#     initial_input = {"messages": "Multiply 4 and 3"}
#     thread = await client.threads.create()
#     async for chunk in client.runs.stream(
#         thread["thread_id"],
#         "agent",
#         input=initial_input,
#         stream_mode="values",
#         interrupt_before=["assistant"],
#     ):
#         print(f"Receiving new event of type: {chunk.event}...")
#         msgs = chunk.data.get("messages", [])
#         if msgs:
#             print(msgs[-1])
#         print("-" * 40)

#     print(f"\n\n\n Get the current state of the graph.")
#     current_state = await client.threads.get_state(thread['thread_id'])
#     print(current_state['values']['messages'])

#     print(f"\n\n We can look at the last message of the state")
#     last_msg = current_state['values']['messages'][-1]
#     print(last_msg)

#     print(f"\n\n\n We can update the state")
#     last_msg['content'] = "Call the available tools that are accessible to you on the inputs 4 and 2"
#     print(last_msg)

#     print(f"\n\n\n To over write the message we pass the updated message content with the id of the message.")
#     await client.threads.update_state(thread['thread_id'], {"messages" : last_msg})
#     # current_state = await client.threads.get_state(thread['thread_id'])
#     print(current_state['values']['messages'])

#     print(f"\n\n\n Now we resume by passing None.")

#     async for chunk in client.runs.stream(
#         thread['thread_id'],
#         assistant_id="agent",
#         input=None,
#         stream_mode="values",
#         interrupt_before=['assistant']):
#         print(f"Receiving new event of type: {chunk.event}...")
#         msgs = chunk.data.get('messages', [])
#         if msgs:
#             print(msgs[-1])
#         print("-"*40)

#     print(f"\n\n\n Again invoke the graph with None as the execution is paused at assistant node.")

#     async for chunk in client.runs.stream(
#         thread['thread_id'],
#         assistant_id="agent",
#         stream_mode="values",
#         interrupt_before=["assistant"]):
#         print(f"Receiving new event of type: {chunk.event}...")
#         msgs = chunk.data.get('messages', [])
#         if msgs:
#             print(msgs[-1])
#         print(f"\n\n\n ")

# if __name__ == "__main__":
#     asyncio.run(async_invocation())


## Awaiting user input and adding it as part of the node.


def human_feedback(state: MessagesState):
    pass


builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant)
builder.add_node("human_feedback", human_feedback)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "human_feedback")
builder.add_conditional_edges("human_feedback", assistant)
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "human_feedback")

memory = MemorySaver()

graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)


initial_input = {"messages": "Multiply 2 and 5"}

thread = {"configurable": {"thread_id": "3"}}

for event in graph.stream(initial_input, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

user_input = input("What's the update..")

graph.update_state(thread, {"messages": user_input}, as_node="human_feedback")
# graph.update_state(thread, {"messages" : user_input})

for event in graph.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

for event in graph.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()
