from operator import add
from typing import Annotated

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, field_validator


class OverallState(BaseModel):
    research_topic: str
    queries: list[str] = Field(default_factory=list)
    web_results: Annotated[list[dict], add] = Field(default_factory=list)
    report_plan: dict = Field(default_factory=dict)
    section_contents: Annotated[list[str], add] = Field(default_factory=list)
    final_answer: str = ""


class WebResearchState(BaseModel):
    research_topic: str

    # @field_validator("number_queries", mode="before")
    # @classmethod
    # def validate_number_queries(cls, value):
    #     if not isinstance(value, int) or value <= 0:
    #         raise ValueError("查询个数必须是正整数")
    #     return value


class QueryGenerationState(BaseModel):
    queries: list[str]
    rationale: str


class Query(BaseModel):
    query: str


class WebAnswerState(BaseModel):
    results: Annotated[list[dict], add]


class Report(BaseModel):
    title: str
    summary: str
    sections: list[dict]


class SectionState(BaseModel):
    title: str
    summary: str
    idx: int
    heading: str
    content: str


class WritingAnswerState(BaseModel):
    section_content: str
    rationale: str


class PaperState(BaseModel):
    section_contents: Annotated[list[str], add]


class FinalAnswer(BaseModel):
    final_answer: str
    rationale: str
