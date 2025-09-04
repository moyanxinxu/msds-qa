from typing import Literal
from uuid import uuid4

from langchain.schema import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents.handbbook_qa.prompts import is_need_retriver_prompt
from src.agents.handbbook_qa.states_tools_schemas import (
    OverallState,
    isNeedRetriverOutput,
)
from src.config import hp
from src.core import ToolSet
from src.db import FaissDB
from src.model import OllamaClient

llm_client = OllamaClient()

chat_model = llm_client.get_chat_model()
embed_model = llm_client.get_embed_model()


def isNeedRetriver(state: OverallState) -> dict:
    sys_message = SystemMessage(content=is_need_retriver_prompt)
    llm = chat_model.with_structured_output(isNeedRetriverOutput)
    result = llm.invoke([sys_message] + state["messages"])

    return {
        "is_need_retriver": result.is_need_retriver,
        "standard_query": result.standard_query,
        "rationale": result.rationale,
    }


def continue_to_kg(state: OverallState) -> Literal["rag", "final_answer"]:
    if state["is_need_retriver"]:
        return "rag"
    else:
        return "final_answer"


def rag_node(state: OverallState) -> dict:
    faiss_retriever = ToolSet.get_faiss_retriever_tool(
        FaissDB(hp.knowledge_space, embed_model).get_db(),
        "faiss_retriever",
        "用于检索新工艺设计，包括工艺原理及特点、物流数据表、物料平衡、装置概况、装置规模、装置组成、装置特点、工艺原理、工艺流程说明、操作变量分析、操作程序、事故处理原则",
    )

    tool_node = ToolNode([faiss_retriever])
    # knowledge_result = faiss_retriever.invoke(state["standard_query"])

    id_uuid = uuid4()

    tool_call = {
        "name": "faiss_retriever",
        "args": {"query": state["standard_query"]},
        "id": str(id_uuid),
        "type": "tool_call",
    }

    ai_message = [AIMessage(content="", tool_calls=[tool_call])]
    tool_message = tool_node.invoke({"messages": ai_message})

    tool_message = ai_message + tool_message["messages"]
    return {
        "knowledge_result": tool_message[-1].content,
        "messages": tool_message,
    }


def final_answer(state: OverallState) -> dict:

    result = chat_model.invoke(state["messages"])

    return {"answer": result.content, "messages": result}


builder = StateGraph(OverallState)
builder.add_node("is_need_retriver", isNeedRetriver)
builder.add_node("rag", rag_node)
builder.add_node("final_answer", final_answer)

builder.add_edge(START, "is_need_retriver")
builder.add_conditional_edges(
    "is_need_retriver",
    continue_to_kg,
    ["rag", "final_answer"],
)

builder.add_edge("rag", "final_answer")
builder.add_edge("final_answer", END)

graph = builder.compile()
