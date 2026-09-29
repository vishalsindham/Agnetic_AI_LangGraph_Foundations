import asyncio

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph_sdk import get_client

load_dotenv()


# The docstring is required for the functions which are passed as tools.
def multiply(a: int, b: int) -> int:
    """
    Function to mutiply the inputs
    """
    return a * b


def add(a: int, b: int) -> int:
    """
    Function to perform addition on the inputs
    """
    return a + b


def divide(a: int, b: int) -> float:
    """
    Function to divide the inputs.
    """
    return a / b


tools = [add, multiply, divide]
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)

builder.add_edge("tools", "assistant")

memory = MemorySaver()

workflow = builder.compile(interrupt_before=["tools"], checkpointer=memory)

initial_msg = {"messages": HumanMessage(content="Multiply 2 and 4")}

thread = {"configurable": {"thread_id": "1"}}

# for event in workflow.stream(initial_msg, thread, stream_mode="values"):
#     event["messages"][-1].pretty_print()


# # The graph is interrupted at the tools node and waiting for the instructions from the user.
# # We can check the next node to be executed.
# state = workflow.get_state(thread)
# print(f" Next node to executed : {state.next}")

# To continue the graph from the last state checkpoint we invoke the graph with None
# langgraph will reemit the values, it will execute the nodes which are not executed till now.

# for event in workflow.stream(None, thread, stream_mode="values"):
#     event['messages'][-1].pretty_print()

# Now let's bring this together with approval step

# thread = {"configurable": {"thread_id": "2"}}

# for event in workflow.stream(initial_msg, thread, stream_mode="values"):
#     event["messages"][-1].pretty_print()

# user_approval = input("Do you want to call the tool..? (yes/no) :")

# if user_approval.lower() == "yes":

#     for event in workflow.stream(None, thread, stream_mode="values"):
#         event["messages"][-1].pretty_print()
# else:
#     print("Operation cancelled by user.")


# breakpoints with LangGraph API
# We can add interrupts to API streaming graph by passing the interrupt when streaming.


async def streamingAPI(initial_msg, client, thread):

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id="agent",
        input=initial_msg,
        stream_mode="values",
        interrupt_before=["tools"],
    ):
        print(f"Receving new event of type : {chunk.event}...")
        messages = chunk.data.get("messages", [])
        if messages:
            print(messages[-1])
        print("-" * 40)


async def main(msg):
    client = get_client(url="http://127.0.0.1:2024")
    thread = await client.threads.create()
    await streamingAPI(msg, client, thread)
    print(
        "Interrupt is evoked and here and it will be continued from the same state as we pass None and stream the graph",
        "\n" * 30,
    )
    await streamingAPI(None, client, thread)


if __name__ == "__main__":
    # The execution of the graph will be interrupted when the tools node is to be executed
    # and it will wait for user input.
    msg = {"messages": HumanMessage(content="Multiply 2 and 3")}
    asyncio.run(main(msg))

    # Invoke the graph again with None it will execute from where it was interrupted
