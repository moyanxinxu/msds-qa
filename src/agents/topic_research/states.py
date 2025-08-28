from operator import add
from typing import Annotated, TypedDict

from pydantic import BaseModel


class OverallState(TypedDict):
    research_topic: str
    queries: list[str]
    web_results: Annotated[list[dict], add]
    sections: list[dict]
    section_contents: Annotated[list[str], add]
    final_answer: str

class QueryGenerationState(BaseModel):
    queries: list[str]
    rationale: str


class Query(BaseModel):
    query: str


class SectionState(BaseModel):
    idx: int
    heading: str
    content: str
    purpose: str
    relation: str


class Report(BaseModel):
    title: str
    summary: str
    sections: list[SectionState]


class WritingAnswerState(BaseModel):
    section_content: str
    rationale: str


class FinalAnswer(BaseModel):
    final_answer: str
    rationale: str
