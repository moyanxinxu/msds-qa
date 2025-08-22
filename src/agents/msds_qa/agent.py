from uuid import uuid4

from colorama import Fore
from copilotkit.langgraph import copilotkit_customize_config
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send

from src.agents.msds_qa.prompts import isNeedQueryMsds
from src.agents.msds_qa.states_tools_schemas import (
    InputState,
    NrccQuery,
    OutputState,
    isNeedSearchNrccInputModel,
    msdsOverallState,
)
from src.core import ToolSet
from src.model import GeminiClient, OllamaClient, SiliconflowClient

assert load_dotenv()
client = GeminiClient()
chat_model = client.get_chat_model()


def is_need_query_nrcc(
    state: msdsOverallState,
    config: RunnableConfig,
) -> isNeedSearchNrccInputModel:
    config = copilotkit_customize_config(config, emit_messages=False)
    llm = chat_model.with_structured_output(isNeedSearchNrccInputModel)

    system_message = SystemMessage(content=isNeedQueryMsds)
    messages = [system_message] + state.get("messages", [])

    result = llm.invoke(messages, config=config)

    return result


def continue_to_query_nrcc(state: isNeedSearchNrccInputModel, config: RunnableConfig):
    is_needed = state.is_need
    chems = state.chems

    if is_needed and chems:
        sends = [Send("search_at_nrcc", {"chem": chem}) for chem in chems]
    else:
        sends = "finalize_answer"
    return sends


def search_at_nrcc(state: NrccQuery, config: RunnableConfig):
    nrcc_search_engine = ToolNode([ToolSet.get_nrcc_chem_info_tool()])

    id_uuid = uuid4()
    tool_call = {
        "name": "ChemInfoRetriever",
        "args": {"chem_name": state["chem"]},
        "id": str(id_uuid),
        "type": "tool_call",
    }

    ai_message = [AIMessage(content="", tool_calls=[tool_call])]
    tool_message = nrcc_search_engine.invoke({"messages": ai_message})

    messages = ai_message + tool_message["messages"]

    return {"messages": messages}


def finalize_answer(state: msdsOverallState, config: RunnableConfig):
    # chem_infos = state.get("chem_infos", [])
    messages = state.get("messages", [])
    result = chat_model.invoke(messages)

    return {"messages": result}


builder = StateGraph(msdsOverallState, input=InputState, output=OutputState)

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
        Fore.GREEN
        + graph.invoke(
            # {
            #     "messages": [
            #         {"role": "user", "content": "我想坐船去外蒙古，帮我指一条明路"}
            #     ]
            # },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "我想查甲苯和乙炔的象形图",
                    },
                ]
            },
            config={"configurable": {"thread_id": 42}},
        )["messages"][-1].content
    )
