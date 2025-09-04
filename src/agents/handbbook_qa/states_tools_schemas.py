from typing import Annotated, TypedDict

from langgraph.graph import add_messages
from pydantic import BaseModel


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]

    query: str

    is_need_retriver: bool
    standard_query: str

    knowledge_result: str
    answer: str


class isNeedRetriverOutput(BaseModel):
    is_need_retriver: bool
    rationale: str
    standard_query: str
