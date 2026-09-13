from dotenv import load_dotenv
from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState
from typing import Annotated
from langgraph.graph.message import add_messages
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

load_dotenv()


# msgs = [AIMessage(content="Hey there, how can I help you today.", name="Model")]
# msgs.append(HumanMessage(content=f"Listen to what I say.", name="Vishal"))
# msgs.append(AIMessage(content=f"I am all ears, yours.", name="Model"))
# msgs.append(HumanMessage(content=f"I want to learn about the sea turtles.", name="Vishal"))

# for msg in msgs:
#     msg.pretty_print()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
# response = llm.invoke(msgs)
# print("The object type of response",type(response))

# print(response.content[0]['text'])

# print("There is a metadata in the response", response.response_metadata)


# We can pass a function schema to the LLM which the LLM use when required.

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm_with_tools = llm.bind_tools(multiply)

# # When we invoke the llm with the contex where tool call is needed the llm will not make a tool call.

# invoked_response =  llm_with_tools.invoke([HumanMessage(content=f"Are you there", name="Vishal")])

# print(invoked_response)


# # Here we are asking a query which can be calculated using the tool we passed it.
# tool_call =  llm_with_tools.invoke([HumanMessage(content=f"What is 5 multiplied by 5", name="Vishal")])

# print(tool_call.tool_calls)

# class MessageState(TypedDict):
#     messages: list[AnyMessage]


# class MessagesState(MessagesState):
#     # Add any kets needed beyond messages which is pre-built
#     pass

# # Initial state

# initial_messages = [AIMessage(content="Hello, How can I assist you.?", name="Model"),
#                     HumanMessage(content="Hello, Bye", name="Vishal")
# ]

# # New message to add
# new_message = AIMessage(content="Bye Bye")

# print("Returned list of messages ", add_messages(initial_messages, new_message))


# Node
def tool_calling_llm(state: MessagesState):
    return {"messages" : [llm_with_tools.invoke(state['messages'])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tools_llm", tool_calling_llm)
builder.add_edge(START, "tools_llm")
builder.add_edge("tools_llm", END)

# Compile the graph
graph = builder.compile()

# View
# graph.get_graph().draw_mermaid_png(output_file_path="chain.png")

msgs = graph.invoke({"messages" : HumanMessage(content="Hello")})

for msg in msgs['messages']:
    msg.pretty_print()


msgs = graph.invoke({"messages" : HumanMessage(content="Multiply 6 and 3")})

for msg in msgs['messages']:
    msg.pretty_print()