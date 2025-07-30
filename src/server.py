import os

import uvicorn
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from dotenv import load_dotenv
from fastapi import FastAPI

from src.agents import msds_qa_graph, predictive_state_updates_graph

assert load_dotenv()

app = FastAPI()
sdk = CopilotKitRemoteEndpoint(
    agents=[
        LangGraphAgent(
            name="msds-qa-agent", description="msds-qa-agent", graph=msds_qa_graph
        ),
        LangGraphAgent(
            name="predictive_state_updates",
            description="predictive_state_updates",
            graph=predictive_state_updates_graph,
        ),
    ]
)

add_fastapi_endpoint(app, sdk, "/copilotkit")


def main():
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
