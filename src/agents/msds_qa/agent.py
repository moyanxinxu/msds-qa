from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.msds_qa.prompts import isNeedQueryMsds
from src.agents.msds_qa.states_tools_schemas import (
    chemNrcc,
    chemsNrcc,
    isNeedSearchNrccModel,
    msdsOverallState,
)
from src.core import ToolSet
from src.model import GeminiClient, OllamaClient, SiliconflowClient

assert load_dotenv()
client = GeminiClient()
chat_model = client.get_chat_model()


def is_need_query_nrcc(state: msdsOverallState, config: RunnableConfig) -> chemsNrcc:
    llm = chat_model.with_structured_output(isNeedSearchNrccModel)
    formatted_prompt = isNeedQueryMsds.format(query=state["messages"][-1].content)
    result = llm.invoke(formatted_prompt)
    return {"is_need": result.is_need, "chems": result.chems}


def continue_to_query_nrcc(state: chemsNrcc, config: RunnableConfig):
    is_needed = state.get("is_need", False)
    chems = state.get("chems", [])

    if is_needed and chems:
        sends = [Send("search_at_nrcc", {"chem": chem}) for chem in chems]
    else:
        sends = "finalize_answer"
    return sends


def search_at_nrcc(state: chemNrcc, config: RunnableConfig):
    nrcc_search_engine = ToolSet.get_nrcc_chem_info_tool()

    chem_name = {"chem_name": state["chem"]}
    chem_info = nrcc_search_engine.invoke(chem_name)

    id_uuid = str(uuid4())
    tool_call = {
        "name": "ChemInfoRetriver",
        "args": chem_name,
        "id": id_uuid,
        "type": "tool_call",
    }

    ai_message = AIMessage(content="", tool_calls=[tool_call])
    tool_message = ToolMessage(content=chem_info, tool_call_id=id_uuid)

    return {"chem_infos": [chem_info], "messages": [ai_message, tool_message]}


def finalize_answer(state: msdsOverallState, config: RunnableConfig):
    chem_infos = state.get("chem_infos", [])
    messages = state.get("messages", [])

    result = chat_model.invoke(messages)

    return {"messages": result}


builder = StateGraph(msdsOverallState)

builder.add_node("is_need_query_nrcc", is_need_query_nrcc)
builder.add_node("search_at_nrcc", search_at_nrcc)
builder.add_node("finalize_answer", finalize_answer)

builder.add_edge(START, "is_need_query_nrcc")

builder.add_conditional_edges(
    "is_need_query_nrcc", continue_to_query_nrcc, ["search_at_nrcc", "finalize_answer"]
)

builder.add_edge("search_at_nrcc", "finalize_answer")
builder.add_edge("finalize_answer", END)


checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    print(
        graph.invoke(
            {
                "messages": [
                    {"role": "user", "content": "我想坐船去外蒙古，帮我指一条明路"}
                ]
            },
            # {"messages": [{"role": "user", "content": "我想查甲苯和乙炔的象形图"}]},
            config={"configurable": {"thread_id": 42}},
        )
    )
