from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
llm_with_tools = llm.bind_tools([multiply])


# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tools_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tools_llm")
builder.add_conditional_edges("tools_llm", tools_condition)

builder.add_edge("tools", END)

# Graph
graph = builder.compile()

# Image(graph.get_graph().draw_mermaid_png(output_file_path="router.png"))


msgs = [HumanMessage(content="Hello, What is 5 multiplied by 6.?")]
msgs = graph.invoke({"messages": msgs})

for msg in msgs["messages"]:
    msg.pretty_print()
