from langchain_community.utilities import SearxSearchWrapper
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.topic_research.prompts import (
    generate_page_instruction,
    generate_planning_instruction,
    generate_quires_instructions,
    generate_section_instruction,
)
from src.agents.topic_research.states import (
    FinalAnswer,
    OverallState,
    Query,
    QueryGenerationState,
    Report,
    WritingAnswerState,
)
from src.config import hp
from src.model import GeminiClient

llm = GeminiClient().get_chat_model()


def generate_query(state: OverallState) -> dict:
    llm_with_structed_output = llm.with_structured_output(QueryGenerationState)

    formatted_prompt = generate_quires_instructions.format(
        research_topic=state.get("research_topic")
    )
    result: dict = llm_with_structed_output.invoke(formatted_prompt)

    return {"queries": result.queries}


def continue_to_web_search(state: OverallState):
    queries = state.get("queries", [])
    if queries:
        return [Send("web_search", {"query": query}) for query in queries]
    else:
        return END


def web_search(state: Query) -> dict:
    search = SearxSearchWrapper(searx_host=hp.searx_host)
    web_result = search.results(state["query"], num_results=hp.searx_num_results)
    if web_result:
        return {"web_results": web_result}
    else:
        return {"web_results": []}


def planner(state: OverallState):
    research_topic = state.get("research_topic", "")

    web_answers = [
        f'片段{idx+1}:{item["snippet"]}'
        for idx, item in enumerate(state.get("web_results", []))
    ]
    formatted_prompt = generate_planning_instruction.format(
        research_topic=research_topic, web_answers=web_answers
    )

    llm_with_structed_output = llm.with_structured_output(Report)
    result: dict = llm_with_structed_output.invoke(formatted_prompt)
    sections: list = result.sections

    return {"sections": sections}


def continue_to_writing(state: OverallState):
    sections = state.get("sections", [])
    sends = [
        Send(
            "writer",
            {
                "idx": section.idx,
                "title": state["research_topic"],
                "heading": section.heading,
                "content": section.content,
                "purpose": section.purpose,
                "relation": section.relation,
            },
        )
        for section in sections
    ]
    return sends


def writer(state):
    title = state["title"]
    purpose = state["purpose"]
    content = state["content"]

    formatted_prompt = generate_section_instruction.format(
        title=title, purpose=purpose, content=content
    )

    llm_with_structed_output = llm.with_structured_output(WritingAnswerState)
    result: dict = llm_with_structed_output.invoke(formatted_prompt)

    return {"section_contents": [result.section_content]}


def Summarizer(state: OverallState):

    formatted_prompt = generate_page_instruction.format(
        section_contents=state.get("section_contents")
    )

    llm_with_structed_output = llm.with_structured_output(FinalAnswer)
    result: dict = llm_with_structed_output.invoke(formatted_prompt)

    return {"final_answer": result.final_answer}


builder = StateGraph(OverallState)

builder.add_node("generate_query", generate_query)
builder.add_node("web_search", web_search)
builder.add_node("planner", planner)
builder.add_node("writer", writer)
builder.add_node("summarizer", Summarizer)

builder.add_edge(START, "generate_query")
builder.add_conditional_edges(
    "generate_query",
    continue_to_web_search,
    ["web_search", END],
)
builder.add_edge("web_search", "planner")

builder.add_conditional_edges("planner", continue_to_writing, ["writer"])

builder.add_edge("writer", "summarizer")
builder.add_edge("summarizer", END)
graph = builder.compile()
