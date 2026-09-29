import asyncio
from typing import List, Literal

from dotenv import load_dotenv
from langchain_core.messages import (HumanMessage, RemoveMessage,
                                     SystemMessage, convert_to_messages)
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph_sdk import get_client

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")


# State
class State(MessagesState):
    summary: str


def call_model(state: State, config: RunnableConfig):

    summary = state.get("summary", "")

    if summary:

        system_msg = (
            f"Here is the summary of the conversation prior to this message : {summary}"
        )

        msgs = [SystemMessage(system_msg)] + state["messages"]
    else:
        msgs = state["messages"]

    response = model.invoke(msgs, config)
    return {"messages": response}


def summarize_conversation(state: State):

    summary = state.get("summary", "")

    if summary:

        summary_msg = (
            f"Here is the summary of the conversation up to date : {summary} \n\n"
            "Update the summary of the conversation taking into account the new messages above."
        )
    else:
        summary_msg = "Create a summary of the conversation for the above messages:"

    msgs = state["messages"] + [HumanMessage(content=summary_msg)]
    response = model.invoke(msgs)

    delete_msgs = [RemoveMessage(id=msg.id) for msg in state["messages"][:-2]]

    return {"messages": delete_msgs, "summary": response.content[0]["text"]}


def should_continue(state: State) -> Literal["summarize", END]:

    msgs = state["messages"]

    if len(msgs) > 4:
        return "summarize"

    return END


builder = StateGraph(State)

builder.add_node("summarize", summarize_conversation)
builder.add_node("conversation", call_model)

builder.add_edge(START, "conversation")
builder.add_edge("summarize", END)
builder.add_conditional_edges("conversation", should_continue)

memory = MemorySaver()

workflow = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

# Stream mode is updates

# for chunk in workflow.stream({"messages" : [HumanMessage(content="Hello, Vishal here.")]}, config=config, stream_mode="updates"):
#     # print(chunk)
#     print(chunk['conversation']['messages'].content[0]['text'])


# stream mode is values

# config = {"configurable" : {"thread_id" : "2"}}

# input_msg = HumanMessage("Hello, Can we eat thoughts and get influenced by them.")

# for event in workflow.stream({"messages" : input_msg}, config=config, stream_mode="values"):
#     # print(event, "\n"*4)
#     for msg in event['messages']:
#         msg.pretty_print()
#     print("-"*30)


# streaming tokens as they get generated

# async def astreammode():
#     config = {"configurable" : {"thread_id" : "3"}}
#     input_msg = HumanMessage(content="In eating others or some thoughts what if we listen and don't think about them to sort if to keep or discard them")

#     async for event in workflow.astream_events({"messages" : [input_msg]}, config, version="v2"):
#         # print(event['metadata'])
#         if event['event'] == "on_chat_model_stream" and event['data']['chunk'].content and event['metadata']['langgraph_node'] == "conversation":
#             print(event['data']['chunk'].content[0]['text'], end="", flush=True)


# Streaming from Langgraph API
async def streamAPI():
    URL = "http://127.0.01:2024"

    client = get_client(url=URL)

    assistants = await client.assistants.search()
    assistants

    thread = await client.threads.create()

    input_msg = HumanMessage(content="Multiply 2 and 3")
    # mode of streaming values
    # async for event in client.runs.stream(thread['thread_id'],
    #                                         assistant_id="agent",
    #                                         input={"messages": [input_msg]},
    #                                         stream_mode='values'
    #                                         ):

    #     # print(event)
    #     # if event.event == 'values':
    #     #     print(event.data['messages'][0]['content'])

    #     msgs = event.data.get('messages', None)
    #     if msgs:
    #         if isinstance(convert_to_messages(msgs)[-1].content, List):
    #             print(convert_to_messages(msgs)[-1].content[0]['text'])
    #         else :
    #             print(convert_to_messages(msgs)[-1].content)
    #     print('-'*25)

    # stream mode of messages check the events types and other information in

    def format_tool_calls(tool_calls):
        if tool_calls:
            formatted_calls = []
            for call in tool_calls:
                formatted_calls.append(
                    f"Tool Call ID : {call['id']}, Function : {call['name']}, Arguments: {call['args']}"
                )
            return "\n".join(formatted_calls)
        return "No tool calls."

    async for event in client.runs.stream(
        thread["thread_id"],
        assistant_id="agent",
        input={"messages": [input_msg]},
        stream_mode="messages",
    ):
        # print(f"Event name: {event.event}, Data : {event.data}")

        # There are three types events emited by the event metadata, messages/complete, metadata/partial
        # Handle metadata events
        if event.event == "metadata":
            print(f"Metadata: Run ID - {event.data['run_id']}")
            print("-" * 50)

        # handle partial message events
        elif event.event == "messages/partial":
            for data_item in event.data:
                if "role" in data_item and data_item["role"] == "user":
                    print(f"Human: {data_item['content']}")
                else:
                    tool_calls = data_item.get("tool_calls", [])
                    invalid_tool_calls = data_item.get("invalid_tool_calls", [])
                    content = data_item.get("content", "")
                    response_metadata = data_item.get("response_metadata", {})

                    if content:
                        print(f"AI : {content[0]['text']}")

                    elif tool_calls:
                        print(f"AI : {content[0]['text']}")

                    elif invalid_tool_calls:
                        print("Invalid Tool calls :")
                        print(format_tool_calls(invalid_tool_calls))

                    elif response_metadata and response_metadata.get("finish_reason"):
                        print(
                            f"Response Metadata: Finish Reason - {response_metadata['finish-reason']}"
                        )
        print("-" * 40)


if __name__ == "__main__":
    # asyncio.run(astreammode())
    asyncio.run(streamAPI())
