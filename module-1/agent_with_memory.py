from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """

    return a * b


def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


tools = [add, multiply, divide]
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of given inputs."
)


def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

# Create a png image of the graph
# react_graph.get_graph().draw_mermaid_png(output_file_path="agent_with_memory.png")

# print("Invocation with out memory.")
# print("-" * 10, "First Invocation", "-" * 10)
# msgs = [HumanMessage(content="Add 3 and 4.")]
# msgs = react_graph.invoke({"messages": msgs})
# for msg in msgs["messages"]:
#     msg.pretty_print()

# print("-" * 10, "Second Invocation", "-" * 10)

# msgs = [HumanMessage(content="Mutliply that by 2.")]
# msgs = react_graph.invoke({"messages": msgs})
# for msg in msgs["messages"]:
#     msg.pretty_print()


print("Invocation with memory")
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)

# Specify a thread

config = {"configurable": {"thread_id": "1"}}

print("_" * 10, "First invocation", "_" * 10)

msgs = HumanMessage(content="Add 5 and 4")

msgs = react_graph_memory.invoke({"messages": msgs}, config)
for msg in msgs["messages"]:
    msg.pretty_print()
print("_" * 10, "Second invocation", "_" * 10)

msgs = [HumanMessage(content="Multiply that by 2.")]
msgs = react_graph_memory.invoke({"messages": msgs}, config)
for msg in msgs["messages"]:
    msg.pretty_print()
