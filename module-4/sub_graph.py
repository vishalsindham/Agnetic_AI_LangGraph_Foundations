from operator import add
from typing import Annotated, List, Optional

from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class Log(TypedDict):
    id: str
    question: str
    docs: Optional[List]
    answer: str
    grade: Optional[int]
    grader: Optional[str]
    feedback: Optional[str]


class FailureAnalysisState(TypedDict):
    cleaned_logs: List[Log]
    failures: List[Log]
    fa_summary: str
    processed_logs: List[str]


class FailureAnalysisOutputState(TypedDict):
    fa_summary: str
    processed_logs: List[str]


def get_failures(state):
    """Get summary of failures."""
    cleaned_logs = state["cleaned_logs"]
    failures = [log for log in cleaned_logs if 'grade' in log]
    return {"failures": failures}


def generate_summary(state):
    """Generate summary of failures."""
    failures = state["failures"]
    # write the logic for summary here
    fa_summary = "Summary of failure logs, a dummy data."
    return {
        "fa_summary": fa_summary,
        "processed_logs": [
            f"Failure analysis on log : {failure['id']}" for failure in failures
        ],
    }


fa_builder = StateGraph(
    state_schema=FailureAnalysisState, output_schema=FailureAnalysisOutputState
)

fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", generate_summary)

fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_summary")
fa_builder.add_edge("generate_summary", END)


class QuestionSummarizationState(TypedDict):
    cleaned_logs: List[Log]
    qs_summary: str
    qs_report: str
    processed_logs: List[str]


class QuestionSummarizationOutputState(TypedDict):
    qs_summary: str
    processed_logs: List[str]


def generate_summary(state):
    cleaned_logs = state["cleaned_logs"]
    summary = "Dummy data as summary"
    return {
        "qs_summary": summary,
        "processed_logs": [f"summary on log {log['id']}" for log in cleaned_logs],
    }


def send_to_slack(state):
    qs_summary = state["qs_summary"]
    # Add report logic
    qs_report = "Question summary report"
    return {"qs_report": qs_report}


qs_builder = StateGraph(
    state_schema=QuestionSummarizationState,
    output_schema=QuestionSummarizationOutputState,
)

qs_builder.add_node("generate_summary", generate_summary)
qs_builder.add_node("send_to_slack", send_to_slack)

qs_builder.add_edge(START, "generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", END)


class EntryGraphState(TypedDict):
    raw_logs: List[Log]
    cleaned_logs: Annotated[List[Log], add]
    fa_summary: str
    qs_report: str
    processed_logs: Annotated[List[int], add]


def cleaned_logs(state):
    raw_logs = state["raw_logs"]
    cleaned_logs = raw_logs
    return {"cleaned_logs": cleaned_logs}


entry_builder = StateGraph(EntryGraphState)

entry_builder.add_node("clean_logs", cleaned_logs)
entry_builder.add_node("fa_sub_graph", fa_builder.compile())
entry_builder.add_node("qs_sub_graph", qs_builder.compile())

entry_builder.add_edge(START, "clean_logs")
entry_builder.add_edge("clean_logs", "qs_sub_graph")
entry_builder.add_edge("clean_logs", "fa_sub_graph")
entry_builder.add_edge("qs_sub_graph", END)
entry_builder.add_edge("fa_sub_graph", END)

workflow = entry_builder.compile()

# display(Image(workflow.get_graph().draw_mermaid_png(output_file_path="sub_graph.png")))


# Dummy logs
question_answer = Log(
    id="1",
    question="How can I import ChatOllama?",
    answer="To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'",
)

question_answer_feedback = Log(
    id="2",
    question="How can I use Chroma vector store?",
    answer="To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).",
    grade=0,
    grader="Document Relevance Recall",
    feedback="The retrieved documents discuss vector stores in general, but not Chroma specifically",
)

raw_logs = [question_answer, question_answer_feedback]

print(workflow.invoke({"raw_logs": raw_logs}))
