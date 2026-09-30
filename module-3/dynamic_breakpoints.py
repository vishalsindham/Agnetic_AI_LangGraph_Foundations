import asyncio

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph_sdk import get_client
from typing_extensions import TypedDict


class State(TypedDict):
    input: str


def step_1(state: State) -> State:
    print(" --- Step-1 --- ")
    return state


def step_2(state: State) -> State:

    if len(state["input"]) > 5:
        raise NodeInterrupt(f"The length is greater than 5 : {state['input']}")

    print(" --- Step 2 ---")
    return state


def step_3(state: State) -> State:
    print(" --- Step 3 --- ")
    return state


builder = StateGraph(State)

builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

memory = MemorySaver()

graph = builder.compile(checkpointer=memory)

initial_input = {"input": "hello there..!"}
thread_config = {"configurable": {"thread_id": "1"}}

for event in graph.stream(initial_input, thread_config, stream_mode="values"):
    print(event)

# check the state of the graph
state = graph.get_state(thread_config)
print(f"\n  State :  {state} , \n \n State.tasks :  {state.tasks}")

# Invoke with None to continue from where it was interrupted
for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)

state = graph.get_state(thread_config)
print(state.next)

# Update the state
graph.update_state(thread_config, {"input": "hi"})

for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)


# interrupts through LangGraph API
async def main():
    url = "http://127.0.0.1:2024"
    client = get_client(url=url)

    assistants = await client.assistants.search()

    thread = await client.threads.create()
    input_dict = {"input": "hello world..!"}

    async for chunk in client.runs.stream(
        thread["thread_id"],
        input=input_dict,
        assistant_id="dynamic_breakpoints",
        stream_mode="values",
    ):
        print(f"Receiving new event of type : {chunk.event}...")
        print(chunk.data)
        print("\n\n")

        current_state = await client.threads.get_state(thread["thread_id"])

        current_state["next"]

        await client.threads.update_state(thread["thread_id"], {"input": "hi!"})

        async for chunk in client.runs.stream(
            thread["thread_id"],
            assistant_id="dynamic_breakpoints",
            input=None,
            stream_mode="values",
        ):
            print(f"Receiving new event of type: {chunk.event}")
            print(chunk.data)
            print("\n\n")

        current_state = await client.threads.get_state(thread["thread_id"])
        current_state


if __name__ == "__main__":
    asyncio.run(main())
