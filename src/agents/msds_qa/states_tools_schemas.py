import operator
from typing import Annotated, TypedDict

from copilotkit import CopilotKitState
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class InputState(CopilotKitState):
    query: Annotated[str, add_messages]


class OutputState(CopilotKitState):
    messages: Annotated[list, add_messages]


class msdsOverallState(InputState, OutputState):
    # messages: Annotated[list, add_messages]
    chem_infos: Annotated[list, operator.add]
    current_chem_infos: list[str]

class isNeedSearchNrccModel(BaseModel):
    is_need: bool = Field(description="是否需要查询NRCC数据库")
    rationale: str = Field(description="判断用户是否需要查询NRCC数据库的理由")
    chems: list[str] = Field(description="用户需要查询的化学品名称的列表")


class chemsNrcc(TypedDict):
    is_need: bool
    chems: list


class chemNrcc(TypedDict):
    chem: str
