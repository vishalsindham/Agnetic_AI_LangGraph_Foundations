import asyncio

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph_sdk import get_client

load_dotenv()


# def multiply(a: int, b: int) -> int:
#     """Multiply the inputs and return the result"""

#     return a * b


# def add(a: int, b: int) -> int:
#     """Add the inputs and return the output"""

#     return a + b


# def divide(a: int, b: int) -> int:
#     """Divide the inputs and return the output"""

#     return a / b


# tools = [add, divide, multiply]
# llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
# llm_with_tools = llm.bind_tools(tools)


# sys_msg = SystemMessage(
#     content="You are a helpful assistant tasked with performing arithmetic operations on a set of inputs."
# )


# def assistant(state: MessagesState):
#     return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# builder = StateGraph(MessagesState)

# builder.add_node("assistant", assistant)
# builder.add_node("tools", ToolNode(tools))

# builder.add_edge(START, "assistant")
# builder.add_conditional_edges("assistant", tools_condition)
# builder.add_edge("tools", "assistant")

# memory = MemorySaver()
# graph = builder.compile(checkpointer=memory)


# initial_input = {"messages" : HumanMessage(content="Multiply 4 and 3")}
# thread = {"configurable" : {"thread_id" : "1"}}

# streamed in values mode
# for event in graph.stream(initial_input, thread, stream_mode="values"):
# event['messages'][-1].pretty_print()
# pass

# Execution of the graph is complete let's check the history of the state after execution

# print(f" Gettting the  current state of the graph :  {graph.get_state(thread)}")

# To get the history of the graph use the method get_state_history

# all_state = [ state for state in graph.get_state_history(thread)]
# print(f" \n\n\n Length of the history of states which is number of checkpoints : {len(all_state)} Each checkpoint is a StatesnapShot", end="\n\n\n")

# for state in all_state:
#     print(state, end="\n\n\n")

# Let's get the step where the human input is received.

# to_reply = all_state[-2]
# print(f" The step at which the human input is received {to_reply}, \n Here the values of the step {to_reply.values} \n This is the next node of the step {to_reply.next} \n \n")

# print(f"We can check the config which has checkpoint id and  thread_id : {to_reply.config}")

# print("We will stream the state, which will reply not re-execute.")

# for event in graph.stream(None, to_reply.config, stream_mode="values"):
#     event['messages'][-1].pretty_print()

# print(event['messages'][-1].content[0]['text'])

# to_fork = all_state[-2]
# print(f"Messages of the forked state :  {to_fork.values['messages']}, Configuration of the forked state : {to_fork.config}")

# # Overwrite the message and the new checkpoint with updated state will be added to thread.
# fork_config = graph.update_state(to_fork.config, {"messages" : [HumanMessage("Multiply 8 and 7", id=to_fork.values['messages'][0].id)]})

# print(f"Forked state checkpoint {fork_config}")

# all_states = [state for state in graph.get_state_history(thread)]
# print(all_states[0].values["messages"])

# print(f"StateSnapShot of the current state : {graph.get_state(thread)}")

# # Let's stream with None input as the thread is updated with a new StateSnapShot of the updated messages
# for event in graph.stream(None, fork_config, stream_mode="values"):
#     event['messages'][-1].pretty_print()

# print(f" Current state of the forked state : {graph.get_state(thread)}")

print(f"From here we understand how to time travel in LangGraph API")

client = get_client(url="http://127.0.0.1:2024")

initial_input = {"messages": [HumanMessage(content="Multiply 7 and 4")]}


async def main():

    thread = await client.threads.create()
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id="agent",
        input=initial_input,
        stream_mode="updates",
    ):
        if chunk.data:
            assistant_node = chunk.data.get("assistant", {}).get("messages", [])
            tools_node = chunk.data.get("tools", {}).get("messages", [])
            if assistant_node:
                print(f'{"-"*30} Assistant Node {"-"*30}')
                print(assistant_node[-1])
            elif tools_node:
                print(f'{"-"*30} Tools Node {"-"*30}')
                print(tools_node[-1])
        # pass

    print("Replying from a checkpoint")
    states = await client.threads.get_history(thread["thread_id"])
    # to_reply = states[-2]
    # print(f"The Step at which human input is receivied {to_reply}")

    # print(f"Let's stream in values to get all values at each checkpoint : \n")

    # async for chunk in client.runs.stream(
    #     thread["thread_id"],
    #     assistant_id="agent",
    #     input=None,
    #     stream_mode="values",
    #     checkpoint_id=to_reply["checkpoint_id"],
    # ):
    #     print(f"Receiving new event of type: {chunk.event}")
    #     print(chunk.data)
    #     print("\n\n")

    # print("Streaming in updates mode", end="\n\n\n")
    # async for chunk in client.runs.stream(
    #     thread["thread_id"],
    #     assistant_id="agent",
    #     input=None,
    #     stream_mode="updates",
    #     checkpoint_id=to_reply["checkpoint_id"],
    # ):
    #     if chunk.data:
    #         assistant_node = chunk.data.get("assistant", {}).get("messages", [])
    #         tools_node = chunk.data.get("tools", {}).get("messages", [])
    #         if assistant_node:
    #             print(f'{"-"*30} Assistant Node {"-"*30}' )
    #             print(assistant_node[-1])
    #         elif tools_node:
    #             print(f'{"-"*30} Tools Node {"-"*30}' )
    #             print(tools_node[-1])

    to_fork = states[-2]
    print(
        "To create a fork took a state from where the human input is received : ",
        to_fork["values"],
        end="\n" * 2,
    )
    print(
        f"Messages ID which is passed to messages as it will update the state a create a new checkpoint : {to_fork['values']['messages'][0]['id']}",
        end="\n" * 2,
    )
    print(
        f"Checking the new node which it will go to : {to_fork['next']} ", end="\n" * 2
    )
    print(
        f"Checking the checkpoint id of the StateSnapShot : {to_fork['checkpoint_id']} ",
        end="\n" * 2,
    )

    print(
        f"This is the checkpoint ID of the StateSnapShot of the message which will be updated : {to_fork['checkpoint_id']}",
        end="\n" * 2,
    )
    forked_input = {
        "messages": HumanMessage(
            content="Multiply 4 and 6", id=to_fork["values"]["messages"][0]["id"]
        )
    }

    forked_config = await client.threads.update_state(
        thread["thread_id"], forked_input, checkpoint_id=to_fork["checkpoint_id"]
    )
    print(
        f"This is the new checkpoint ID of the StateSnapShot of the message which we updated : {forked_config['checkpoint_id']}",
        end="\n" * 2,
    )

    states = await client.threads.get_history(thread["thread_id"])
    print(f"Here is the updated message in the states : {states[0]}", end="\n" * 2)

    # Let's stream the graph with new checkpoint id.
    print(
        f"Let's stream the graph with checkpoint ID, graph will re-execute the nodes as the checkpoint is not executed before."
    )

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id="agent",
        input=None,
        stream_mode="updates",
        checkpoint_id=forked_config["checkpoint_id"],
    ):
        if chunk.data:
            assistant_node = chunk.data.get("assistant", {}).get("messages", [])
            tools_node = chunk.data.get("tools", {}).get("messages", [])
            if assistant_node:
                print(f"{'-' * 40} Assistant Node {'-' * 40}")
                print(assistant_node[-1])
            elif tools_node:
                print(f"{'-' * 40} Tools Node {'-' * 40}")
                print(tools_node[-1])


if __name__ == "__main__":
    asyncio.run(main())
