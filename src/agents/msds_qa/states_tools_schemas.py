from typing import Annotated, TypedDict

from copilotkit import CopilotKitState
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class InputState(CopilotKitState):
    pass


class OutputState(CopilotKitState):
    messages: Annotated[list, add_messages]


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    is_need: bool
    rationale: str
    chems: list[str]
    chem: str


class isNeedSearchNrccInputModel(BaseModel):
    is_need: bool = Field(description="是否需要查询NRCC数据库")
    rationale: str = Field(description="判断用户是否需要查询NRCC数据库的理由")
    chems: list[str] = Field(description="用户需要查询的化学品名称的列表")
