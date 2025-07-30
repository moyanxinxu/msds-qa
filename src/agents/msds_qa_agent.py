import json
from typing import Literal

from copilotkit import CopilotKitState
from copilotkit.langgraph import copilotkit_emit_state
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from src.config import hp
from src.core import ToolSet
from src.db import FaissDB
from src.model import GeminiClient, OllamaClient, SiliconflowClient

assert load_dotenv()
client = GeminiClient()
chat_model = client.get_chat_model()

tools = [
    ToolSet.get_nrcc_chem_info_tool(),
    ToolSet.get_faiss_retriever_tool(
        FaissDB(hp.knowledge_space, SiliconflowClient().get_embed_model()).get_db(),
        "faiss_retriever",
        "用于检索薄膜操作、磺化装置各组件维护保养方法、IKA旋转蒸发仪清洁消毒指南的检索工具等等实验室器件知识库",
    ),
]


class AgentState(CopilotKitState):
    language: str
    agent_name: str
    chem_info: dict
    file_path: str


async def chat_node(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["tool_node", "__end__"]]:
    system_message = SystemMessage(
        content=f"You are a helpful assistant. You name is {state.get('agent_name')}, you should answer questions in {state.get('language')} language. You should responsed in MarkDown format"
    )

    llm_with_tool = chat_model.bind_tools(tools)

    response = await llm_with_tool.ainvoke([system_message, *state["messages"]], config)
    if (
        isinstance(state["messages"][-1], ToolMessage)
        and state["messages"][-1].name == "ChemInfoRetriever"
    ):
        state["chem_info"] = json.loads(state["messages"][-1].content)
    if isinstance(response, AIMessage) and response.tool_calls:
        return Command(goto="tool_node", update={"messages": response})
    return Command(goto=END, update={"messages": response})


app = StateGraph(AgentState)
app.add_node("chat_node", chat_node)
app.add_node("tool_node", ToolNode(tools))
app.add_edge("tool_node", "chat_node")
app.set_entry_point("chat_node")

checkpointer = MemorySaver()
graph = app.compile(checkpointer=checkpointer)
