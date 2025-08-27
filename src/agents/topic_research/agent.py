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
    PaperState,
    Query,
    QueryGenerationState,
    Report,
    SectionState,
    WebAnswerState,
    WebResearchState,
    WritingAnswerState,
)
from src.config import hp
from src.model import GeminiClient

llm = GeminiClient().get_chat_model()


def generate_query(state: OverallState) -> list[Send]:
    llm_with_structed_output = llm.with_structured_output(QueryGenerationState)

    formatted_prompt = generate_quires_instructions.format(
        research_topic=state.research_topic
    )
    result: dict = llm_with_structed_output.invoke(formatted_prompt)

    return [Send("web_search", {"query": query}) for query in result.queries]


def web_search(state: Query) -> dict:
    search = SearxSearchWrapper(searx_host=hp.searx_host)
    results = search.results(state.query, num_results=hp.searx_num_results)
    if results:
        return {"results": results}
    else:
        return {"results": []}


def planner(state: WebAnswerState):
    web_answers = state.results
    formatted_prompt = generate_planning_instruction.format(web_answers=web_answers)

    llm_with_structed_output = llm.with_structured_output(Report)
    result: dict = llm_with_structed_output.invoke(formatted_prompt)
    sections: list = result["sections"]

    sends = [
        Send(
            "writer",
            {
                "title": result["title"],
                "summary": result["summary"],
                "idx": section["idx"],
                "heading": section["heading"],
                "content": section["content"],
            },
        )
        for section in sections
    ]
    return sends


def writer(state: SectionState):
    title = state.title
    summary = state.summary
    current_heading = state.heading
    current_content = state.content

    formatted_prompt = generate_section_instruction.format(
        title=title,
        summary=summary,
        current_heading=current_heading,
        current_content=current_content,
    )

    llm_with_structed_output = llm.with_structured_output(WritingAnswerState)
    result: dict = llm_with_structed_output.invoke(formatted_prompt)

    return {"section_content": result["section_content"]}


def Summarizer(state: PaperState):

    formatted_prompt = generate_page_instruction.format(
        section_contents=state.section_contents
    )

    llm_with_structed_output = llm.with_structured_output(FinalAnswer)
    result: dict = llm_with_structed_output.invoke(formatted_prompt)

    return {"final_answer": result["final_answer"]}


builder = StateGraph(OverallState)

builder.add_node("generate_query", generate_query)
builder.add_node("web_search", web_search)
builder.add_node("planner", planner)
builder.add_node("writer", writer)
builder.add_node("summarizer", Summarizer)


builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "web_search")
builder.add_edge("web_search", "planner")
builder.add_edge("planner", "writer")
builder.add_edge("writer", "summarizer")
builder.add_edge("summarizer", END)
graph = builder.compile()

graph.invoke({"research_topic": "苹果公司2024财年营收与iPhone销量增长的比较分析"})
