import asyncio
import uuid

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

load_dotenv()
in_memory_store = InMemoryStore()

# Namespace for the memory to save
# user_id = "1"
# namespace_for_memory = (user_id, "memories")

# Save a memory to namespace as key and value
# key = str(uuid.uuid4())

# The value needs to be a dictionary
# value = {"food_preference": "I like khichdi"}

# Save the memory
# in_memory_store.put(namespace_for_memory, key, value)

# We use search to retrieve objects from the store by namespace
# memories = in_memory_store.search(namespace_for_memory)

# This is a Item instance
# print(memories[0].dict())

# for memory in memories:
#     print(f"{memory.key} : {memory.value}")

# Get the namespace by memory and key
# print("Getting the namespace by memory and key")
# memory = in_memory_store.get(namespace_for_memory, key)
# print(memory)
# print(memory.dict())

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.4)

# Chatbot instructions
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user.
If you have memory for this user, use it personalize  your responses.
Here is the memory (it may be empty) : {memory}"""

# Create new memory from the chat history and any existing memory
CREATE_MEMORY_INSTRUCTION = """You are collecting information about the user to personalize your responses.

CURRENT USER INFORMATION:
{memory}

INSTRUCTIONS:
1. Review the chat history below carefully
2. Identify new information about the user, such as:
    - Personal details (name, location)
    - Preferences (likes, dislikes)
    - Interests and hobbies
    - Past experiences
    - Goals and future plans
3. Merge any new information with existing memory
4. Format the memory as a clear, bulleted list
5. If new information conflicts with the existing memory, keep the most recent version 

Remember: Only include the factual information directly stated by the user. Do not make assumptions or inferences.

Based on the chat history below, please update the user information:"""


def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):

    user_id = config["configurable"]["user_id"]

    namespace = ("memory", "user_id")
    key = "user_memory"
    existing_memory = store.get(namespace, key)

    if existing_memory:
        existing_memory_content = existing_memory.value.get("memory")
    else:
        existing_memory_content = "No existing memory found."

    sys_msg = MODEL_SYSTEM_MESSAGE.format(memory=existing_memory_content)
    response = model.invoke([SystemMessage(content=sys_msg)] + state["messages"])

    return {"messages": response.content[0]["text"]}


def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):

    user_id = config["configurable"]["user_id"]

    namespace = ("memory", user_id)
    key = "user_memory"
    existing_memory = store.get(namespace, key)

    if existing_memory:
        existing_memory_content = existing_memory.value.get("memory")
    else:
        existing_memory_content = "No existing memory found"

    sys_msg = CREATE_MEMORY_INSTRUCTION.format(memory=existing_memory_content)
    updated_memory = model.invoke([SystemMessage(content=sys_msg)] + state["messages"])

    key = "user_memory"

    store.put(namespace, key, {"memory": updated_memory.content[0]["text"]})


builder = StateGraph(MessagesState)

builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)

builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)

across_thread_memory = InMemoryStore()

within_thread_memory = MemorySaver()

graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)

graph.get_graph(xray=1).draw_mermaid_png(output_file_path="memory_store.png")


async def core_func(usr_input: str, config: TypedDict):

    input_msgs = HumanMessage(content=usr_input)

    async for chunk in graph.astream(
        {"messages": input_msgs}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()


async def async_func(config: TypedDict):

    while True:
        usr_input = input("Enter the your message (quit | bye | q to close the chat) :")
        if usr_input == "quit" or usr_input == "bye" or usr_input == "q":
            print(f"\n\n Fetching all the messages saved in the thread : ")
            config = {"configurable": {"thread_id": "1"}}
            current_state = graph.get_state(config).values
            if len(current_state.get("messages", [])) >= 1:
                for msg in current_state["messages"]:
                    msg.pretty_print()
            else:
                print("There was no interaction.")
            break
        elif usr_input == "":
            continue
        else:
            await core_func(usr_input, config)
    print(f"\n\n Fetching the across thread memory information :")
    user_id = "1"
    namespace = ("memory", user_id)
    existing_memory = across_thread_memory.get(namespace, "user_memory")
    print(existing_memory.dict())


if __name__ == "__main__":
    config_1 = {"configurable": {"thread_id": "1", "user_id": "1"}}
    asyncio.run(async_func(config_1))
    print(
        f"\n\nInvoking the model again with different thread_id with same user id to access across thread memory\n\n"
    )
    config_2 = {"configurable": {"thread_id": "2", "user_id": "1"}}
    asyncio.run(async_func(config_2))
