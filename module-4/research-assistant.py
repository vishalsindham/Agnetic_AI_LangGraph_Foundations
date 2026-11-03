import asyncio
import operator
from typing import Annotated, List

import wikipedia
from dotenv import load_dotenv
from IPython.display import Markdown
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     get_buffer_string)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
# llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


class Analyst(BaseModel):

    affiliation: str = Field(description=" Basic affiliation of the anlayst. ")

    role: str = Field(description=" Role of the analyst in the context of the topic. ")

    name: str = Field(description=" Name of the analyst. ")

    description: str = Field(
        description=" Description of the anlayst focus, concerns and motives. "
    )

    @property
    def persona(self):
        return f"\nName : {self.name} \nRole : {self.role} \nDescription : {self.description} \nAffilication : {self.affiliation}"


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description=" Comprehesive list of analysts with their roles and affiliation. "
    )


class GenerateAnalystsState(TypedDict):
    topic: str  # topic on which research is to be done
    max_analysts: int  # Number of analysts
    human_analyst_feedback: str  # human feedback
    analysts: List[Analyst]  # analysts asking questions


analyst_instructions = """You are tasked with creating a set of AI analyst personas. Follow this instructions carefully.

1. First, review the research topic :
{topic}

2. Examine any editioral feedback that has been optionally provided to guide creation of the analysts:
{human_analyst_feedback}

3. Determing the most interesting themes based upon documents and / or feedback above.

4. Pick the top {max_analysts} themes.

5. Assign one theme to each theme."""


def create_analysts(state: GenerateAnalystsState):
    """Generate a set of analysts"""

    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    #  llm which will return the analysts as per the scheme passed
    structured_llm = llm.with_structured_output(Perspectives)

    # build the system message
    system_msg = analyst_instructions.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts,
    )

    # make the llm call to generate the analysts with the required persona's
    perspectives = structured_llm.invoke(
        [SystemMessage(content=system_msg)]
        + [HumanMessage(content="Generate the analysts with the requirements shared")]
    )

    # Updating the state with the generated analysts
    return {"analysts": perspectives.analysts}


def human_analyst_feedback(state: GenerateAnalystsState):
    """No-op node a place holder for the human analysts feedback"""
    pass


def should_continue(state: GenerateAnalystsState):
    """Where the execution should flow as per the state."""

    human_analyst_feedback = state.get("human_analyst_feedback", None)
    if human_analyst_feedback:
        # if there is feedback return and re generate the analysts.
        return "create_analysts"

    return END


builder = StateGraph(GenerateAnalystsState)

# add nodes
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_analyst_feedback)

# build edges
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges(
    "human_feedback", should_continue, ["create_analysts", END]
)
memory = MemorySaver()

generate_analysts = builder.compile(
    interrupt_before=["human_feedback"], checkpointer=memory
)


class InterviewState(MessagesState):
    max_num_turns: int  # Number of conversations turn
    context: Annotated[list, operator.add]  # Source of docs
    analyst: Analyst  # Analyst asking questions
    interview: str  # Interview transcript
    sections: str  # Final key we duplicate in outer space for Send() API


class SearchQuery(BaseModel):
    search_query: str = Field(None, description=" search query for retrieval.")


question_instructions = """You are an analyst tasked with interviewing an expert to learn about a specific topic.

You goal is to boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising and non-obvious.

2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals : {goals}

Begin by introducing yourself using a name that fits your personna, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.

When you are statisfied with your understanding, complete the interview with: "Thank you so much for the help!"

Remember to stay in character throughout your response, reflecting the personna and goals provided to you.
"""


def generate_question(state: InterviewState):
    """Node to generate a question"""

    analyst = state["analyst"]
    msgs = state["messages"]

    sys_msg = question_instructions.format(goals=analyst)
    question = llm.invoke([SystemMessage(content=sys_msg)] + msgs)

    # return the response
    return {"question": question}


max_web_search_results = int(
    input("Enter the number of web search results you want to retrieve :")
)
tavily_search = TavilySearch(max_results=max_web_search_results)

wikipedia.set_user_agent("LanGraphAgent/1.0(info-en@wikimedia.org)")

# Search query prompt
search_instructions = SystemMessage(
    content=f"""You will be given a conversation between an analyst and an expert.

You goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.

First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query."""
)


def search_web(state: InterviewState):
    """Retrieve docs from web search"""

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions] + state["messages"])

    data = tavily_search.invoke({"query": search_query.search_query})
    search_docs = data.get("results", data)

    formatted_search_docs = "\n\n--\n\n".join(
        [
            f"<Document href='{doc['url']}'/>/n {doc['content']} /n</Document>"
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


def search_wikipedia(state: InterviewState):
    """Retrieve docs from wikipedia."""

    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions] + state["messages"])

    # Search Docs
    max_wiki_doc = int(input("Enter the maximum wiki documents you want to load : "))
    search_docs = WikipediaLoader(
        query=search_query.search_query, load_max_docs=max_web_search_results
    ).load()

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f"<Document source='{doc.metadata.get('source', '')}' page='{doc.metadata.get('page','')}'/>\n{doc.page_content}\n</Document>"
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


answer_instructions = """You are an expert being interviewed by an analyst.

Here is analyst area of focus: {goals}.

Your goal is to answer a question posed by the interviewer.

To answer this question, use this context:

{context}

When answering question, follow this guidelines:

1. Use only the information provided in the context.

2. Do not introduce external information or make assumptions beyond what is explicitly mentioned in the context.

3. The context contains sources at the topic of each individual document.

4. Include this sources in your answer next to any relevant statements. For example, for source #1 use [1].

5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc.

6. If the source is : '<Document source="assistant/docs/llama3_1.pdf" page="7/>'  then just list:

[1] assistant/docs/llama3_1.pdf, page 7

And skip the addition of the brackets as well as the Document source preamble in your citation.
"""


def generate_answer(state: InterviewState):
    """Node to answer a question"""

    # Get state
    analyst = state["analyst"]
    msgs = state["messages"]
    context = state["context"]

    # Answer question
    sys_msg = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=sys_msg)] + msgs)

    # Name the message as coming from an expert
    answer.name = "expert"

    # Append it to state
    return {"messages": [answer]}


def save_interview(state: InterviewState):
    """Save interview"""

    messages = state["messages"]

    interview = get_buffer_string(messages)

    return {"interview": interview}


def route_messages(state: InterviewState, name: str = "expert"):
    """Route between question and answer."""

    # Get messages
    messages = state["messages"]
    conversation_turns = int(
        input(
            "Enter the maximum conversation turns each analyst should have with the expert :"
        )
    )
    max_num_turns = state.get("max_num_turns", conversation_turns)

    # Check the number of expert responses
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than max turns
    if num_responses >= max_num_turns:
        return "save_interview"

    # This router is run after each question-answer pair
    # Get the last question asked if it signal the end of dicussion.
    last_question = messages[-2]
    if "Thank you so much for the help!" in last_question.content[0]:
        return "save_interview"
    return "ask_question"


section_writer_instructions = """You are an expert technical writer.

Your task is to create a short,easily digestible section based on a set of source documents.

1. Analyze the content of the source documents.
- The name of each source document is at the start of the document, with the <Document tag.

2. Create a report structure using markdown formatting:
- Use ## for section title.
- Use ### for sub-section headers.

3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst:
{focus}

5. For the summary section:
- Set up summary with the general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a number list of sources documents, as you use them
- Do not mention the name of experts or interviewers
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents.

6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.met.com/blog/llama-2-1
[4] https://ai.met.com/blog/llama-2-1

There should be no redundant sources. It should simply be:

[3] https://ai.met.com/blog/llama-2-1

8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed
"""


def write_section(state: InterviewState):
    """Node to answer a question"""

    # Get state
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]

    # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
    sys_msg = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke(
        [SystemMessage(content=sys_msg)]
        + [HumanMessage(content=f"Use this source to write your section : {context}")]
    )

    # Append it to state
    return {"sections": [section.content[0]["text"]]}


# Add nodes and edges
interview_builder = StateGraph(InterviewState)

interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wiki", search_wikipedia)
interview_builder.add_node("answer", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)


# graph flow
interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wiki")
interview_builder.add_edge("search_web", "answer")
interview_builder.add_edge("search_wiki", "answer")
interview_builder.add_conditional_edges(
    "answer", route_messages, ["ask_question", "save_interview"]
)
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

# Interview
memory = MemorySaver()
interview_graph = interview_builder.compile(checkpointer=memory).with_config(
    run_name="Conduct Interviews"
)

# view
interview_graph.get_graph().draw_mermaid_png(output_file_path="conduct_interview.png")


class ResearchGraphState(TypedDict):
    topic: str  # Research topic
    max_analysts: int  # Number of analysts
    human_analyst_feedback: str  # Human feedback
    analysts: List[Analyst]  # Analyst asking questions
    sections: Annotated[list, operator.add]  # Send() API Key
    introduction: str  # Introduction for the final report
    content: str  # Content for the final report
    conclusion: str  # Conclusion for the final report
    final_report: str  # Final report


def initiate_all_interviews(state: ResearchGraphState):
    """This is the 'map' step where we run each interview sub-graph using Send API."""

    # Check if human feedback
    human_analyst_feedback = state.get("human_analyst_feedback")
    if human_analyst_feedback:
        return "create_analysts"
    else:
        topic = state.get("topic", "")
        return [
            Send(
                "conduct_interview",
                {
                    "analyst": analyst,
                    "messages": [
                        HumanMessage(
                            content=f"So you said you were writing a article on {topic}.?"
                        )
                    ],
                },
            )
            for analyst in state["analysts"]
        ]


report_writer_instructions = """You are technical writer creating a report on this overall topic:

{topic}

You have a team of analysts. Each analyst has done two things:

1. They conducted an interview with an expert on a specific sub-topic.
2. They wrote up their finding into a memo.

Your task:
1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the centeral ideas from all of the memos.
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:

1. Use markdown formatting.
2. Include no pre-amble for the report.
3. Use no sub-heading.
4. Start your report with a single title header: ## Insights
5. Do not mention any analyst names in your report.
6. Preserve any citiations in the memos, which will be annotated in brackets, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
8. List your sources in order and do not repeat.

[1] Source 1
[2] Source 2

Here are the memos from your analysts to build your report from:{context}

"""


def write_report(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    response = llm.invoke(
        [SystemMessage(content=formatted_str_sections)]
        + [HumanMessage(content=f"Write a report based upon these memos.")]
    )
    return {"content": response.content[0]["text"]}


intro_conclusion_instructions = """You are a technical writer finishing report on {topic}

You will be given all of the sections of the report.

Your job is to write a crisp and compelling introduction or conclusion.

Include no-preamble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting.

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header.

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {formatted_str_sections}
"""


def write_introduction(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    instructions = intro_conclusion_instructions.format(
        topic=topic, formatted_str_sections=formatted_str_sections
    )
    introduction = llm.invoke(
        [instructions] + [HumanMessage(content=f"Write the report introduction")]
    )
    introduction, _ = introduction.content[0]["text"].split("## Conclusion")
    return {"introduction": introduction}


def write_conclusion(state: ResearchGraphState):

    # Fetching sections
    topic = state["topic"]
    sections = state["sections"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report
    instructions = intro_conclusion_instructions.format(
        topic=topic, formatted_str_sections=formatted_str_sections
    )
    conclusion = llm.invoke(
        [instructions] + [HumanMessage(content=f"Write the report conclusion")]
    )
    _, conclusion = conclusion.content[0]["text"].split("## Conclusion")
    return {"conclusion": conclusion}


def finalize_report(state: ResearchGraphState):
    """This is the 'reduce' step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion"""
    # Save the full report
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("## Sources")
        except:
            sources = None
    else:
        sources = None
    final_report = (
        state["introduction"]
        + "\n\n---\n\n"
        + content
        + "\n\n----\n\n"
        + state["conclusion"]
    )
    if sources is not None:
        final_report += "\n\n## Sources\n\n" + sources
    return {"final_report": final_report}


# Add nodes and edges
builder = StateGraph(ResearchGraphState)

builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_analyst_feedback)
builder.add_node("conduct_interview", interview_builder.compile())
builder.add_node("write_report", write_report)
builder.add_node("write_introduction", write_introduction)
builder.add_node("write_conclusion", write_conclusion)
builder.add_node("finalize_report", finalize_report)

# Logic flow
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges(
    "human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"]
)
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_conclusion")
builder.add_edge(
    ["write_introduction", "write_report", "write_conclusion"], "finalize_report"
)
builder.add_edge("finalize_report", END)

# Compile
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
graph.get_graph().draw_mermaid_png(output_file_path="research_workflow.png")


async def async_exe():
    """Needed async function to execute astream invocations."""
    topic = "LangGraph adoption for startups"
    num_analysts = int(input("Enter the number of analysts : "))
    max_analysts = num_analysts
    thread = {"configurable": {"thread_id": "1"}}

    # for event in generate_analysts.stream(
    #     {"topic": topic, "max_analysts": max_analysts}, thread, stream_mode="values"
    # ):
    #     # Review
    #     analysts = event.get("analysts", "")
    #     if analysts:
    #         for analyst in analysts:
    #             print(
    #                 f"\nName : {analyst.name} \nAffiliation : {analyst.affiliation} \nRole : {analyst.role} \nDescription : {analyst.description}\n"
    #             )

    # current_state = generate_analysts.get_state(thread)

    # human_feedback_input = input("Above are the anlaysts enter your feedback : ")
    # generate_analysts.update_state(
    #     thread,
    #     {
    #         # "human_feedback": "Add some one who is the end user of the product for consumer point of view"
    #         "human_analyst_feedback": human_feedback_input
    #     },
    #     as_node="human_feedback",
    # )

    # print(f"\n\n Updated analysts \n\n")
    # for event in generate_analysts.stream(None, thread, stream_mode="updates"):
    #     analysts = event.get("create_analysts", {}).get("analysts", "")
    #     if analysts:
    #         for analyst in analysts:
    #             print(
    #                 f"\nName : {analyst.name} \nRole : {analyst.role} \nDescription : {analyst.description} \nAffiliation : {analyst.affiliation}"
    #             )
    # human_feedback_input = input("Any more changes you want to make : ")
    # if str(human_analyst_feedback).lower() == "approved":
    #     for event in generate_analysts.stream(None, thread, stream_mode="updates"):
    #         analysts = event.get("create_analysts", {}).get("analysts", "")
    #         if analysts:
    #             for analyst in analysts:
    #                 print(
    #                     f"\nName : {analyst.name} \nRole : {analyst.role} \nDescription : {analyst.description} \nAffiliation : {analyst.affiliation}"
    #                 )
    #         print("---Node---")
    #         node_name = next(iter(event.keys()))
    #         print(node_name)

    # final_analysts = generate_analysts.get_state(thread)
    # final_analysts = final_analysts.values.get("analysts", "")
    # if final_analysts:
    #     for analyst in analysts:
    #         print(
    #             f"\nName : {analyst.name} \nRole : {analyst.role} \nDescription : {analyst.description} \nAffiliation : {analyst.affiliation}"
    #         )

    # msgs = [HumanMessage(f"So you said you were writing an article on {topic}?")]
    # response = interview_graph.invoke(
    #     {"analyst": final_analysts[0], "messages": msgs, "max_num_turns": 2},
    #     config=thread,
    # )

    # print(response["sections"][0])

    for event in graph.stream(
        {"topic": topic, "max_analysts": max_analysts}, thread, stream_mode="values"
    ):
        print("---Node---")
        node_name = next(iter(event.keys()))
        print(node_name)

    final_state = graph.get_state(thread)
    report = final_state.values.get("final_report")
    print(report)


async def main():
    await async_exe()


if __name__ == "__main__":
    asyncio.run(main())
