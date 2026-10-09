# from typing_extensions import TypedDict
import warnings

warnings.filterwarnings("ignore")
import asyncio
from operator import add
from typing import Annotated, Any, List, TypedDict

import wikipedia
from dotenv import load_dotenv
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph_sdk import get_client
from typing_extensions import TypedDict

load_dotenv()

# class State(TypedDict):
#     state: List[str]


# class ReturnNodeValue:

#     def __init__(self, node_name):
#         self._value = node_name

#     def __call__(self, state: State) -> Any:
#         print(f"Adding {self._value} to {state['state']}")
#         return {"state": [self._value]}


# builder = StateGraph(State)

# builder.add_node("a", ReturnNodeValue("Node A"))
# builder.add_node("b", ReturnNodeValue("Node B"))
# builder.add_node("c", ReturnNodeValue("Node C"))
# builder.add_node("d", ReturnNodeValue("Node D"))

# builder.add_edge(START, "a")
# builder.add_edge("a", "b")
# builder.add_edge("b", "c")
# builder.add_edge("c", "d")
# builder.add_edge("d", END)

# graph = builder.compile()

# print(f"Graph final state : {graph.invoke({"state": []})}")

# print(f"Modifiying the graph to execute the nodes in parallel using fan out and fan in :")

# builder.add_edge(START, "a")
# builder.add_edge("a","b")
# builder.add_edge("a","c")
# builder.add_edge("b","d")
# builder.add_edge("c","d")
# builder.add_edge("d", END)

# graph = builder.compile()

# print("As there is no reducer to apply parallel state management logic let's use try except :")

# try :
#     print(f"Graph final State : {graph.invoke({"state" : []})}")
# except InvalidUpdateError as e:
#     print(f"The error caught is  : {e}")

# class State(TypedDict):
#     state: Annotated[List[str], add]

# builder = StateGraph(State)

# builder.add_node("a", ReturnNodeValue("Node A"))
# builder.add_node("b", ReturnNodeValue("Node B"))
# builder.add_node("b2", ReturnNodeValue("Node B2"))
# builder.add_node("c", ReturnNodeValue("Node C"))
# builder.add_node("d", ReturnNodeValue("Node D"))

# builder.add_edge(START, "a")
# builder.add_edge("a","b")
# builder.add_edge("a","c")
# builder.add_edge("b","d")
# builder.add_edge("c","d")
# builder.add_edge("d", END)

# graph = builder.compile()

# print(f"Here is the final state of the graph : {graph.invoke({"state": []})}")

# builder.add_edge("b", "b2")
# builder.add_edge(["c","b2"], "d")
# builder.add_edge("d", END)

# graph =  builder.compile()

# print(f"Parallel node execution with multiple nodes on one edge : {graph.invoke({"state" : []})}")

# def sorting_reducer(left, right):

#     if not isinstance(left, list):
#         left = [left]

#     if not isinstance(right, list):
#         right = [right]

#     return sorted(left+right, reverse=False)

# class State(TypedDict):
#     state: Annotated[List[str], sorting_reducer]

# builder = StateGraph(State)

# builder.add_node("a", ReturnNodeValue("Node A"))
# builder.add_node("b", ReturnNodeValue("Node B"))
# builder.add_node("b2", ReturnNodeValue("Node B2"))
# builder.add_node("c", ReturnNodeValue("Node C"))
# builder.add_node("d", ReturnNodeValue("Node D"))

# builder.add_edge(START, "a")
# builder.add_edge("a", "b")
# builder.add_edge("a", "c")
# builder.add_edge("b", "b2")
# builder.add_edge(["b2", "c"], "d")
# builder.add_edge("d", END)

# graph = builder.compile()

# print(f"Sorting reducer to sort the state updates as received : {graph.invoke({"state":[]})}")

# llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.4)
# wikipedia.set_user_agent("LanGraphAgent/1.0(info-en@wikimedia.org)")


# class State(TypedDict):
#     question: str
#     answer: str
#     context: Annotated[list, add]


# def search_web(state):
#     """Retrieve web docs"""

#     tavily_search = TavilySearch(max_results=2)
#     data = tavily_search.invoke({"query": state["question"]})
#     results = data.get("results", data)
#     formatted_search_docs = """\n\n---\n\n""".join(
#         [
#             f"<Document href={doc['url']} />/n {doc['content']} /n</Document>"
#             for doc in results
#         ]
#     )
#     return {"context": [formatted_search_docs]}


# def search_wikipedia(state):
#     """Retrieve docs from wikipedia"""

#     search_docs = WikipediaLoader(query=state["question"], load_max_docs=2).load()
#     formatted_search_docs = "\n\n---\n\n".join(
#         [
#             f"<Document source={doc.metadata["source"]}, page={doc.metadata.get("page", "")} /> \n {doc.page_content} \n <Document/>"
#             for doc in search_docs
#         ]
#     )
#     return {"context": [formatted_search_docs]}


# def answer_question(state):

#     context = state["context"]
#     question = state["question"]

#     answer_template = """Answer the question {question} using the context {context}"""
#     answer_instructions = answer_template.format(question=question, context=context)

#     answer = llm.invoke(
#         [SystemMessage(content=answer_instructions)]
#         + [HumanMessage(content="Answer the question using the context")]
#     )

#     return {"answer": answer}


# builder = StateGraph(State)

# builder.add_node("search", search_web)
# builder.add_node("wiki", search_wikipedia)
# builder.add_node("answer", answer_question)

# builder.add_edge(START, "search")
# builder.add_edge(START, "wiki")
# builder.add_edge("search", "answer")
# builder.add_edge("wiki", "answer")
# builder.add_edge("answer", END)

# workflow = builder.compile()

# response = workflow.invoke({"question": "Latest tool of the LangChain ecosystem"})

# print(response["answer"].content[0]['text'])

client = get_client(url="http://127.0.0.1:2024")


async def main():
    thread = await client.threads.create()
    input_question = {"question": "LangChain ecosystem modules"}

    async for event in client.runs.stream(
        thread["thread_id"],
        assistant_id="parallelization",
        input=input_question,
        stream_mode="values",
    ):
        if event.data is not None:
            answer = event.data.get("answer", None)
            if answer:
                print(answer["content"][0]["text"])


if __name__ == "__main__":
    asyncio.run(main())
