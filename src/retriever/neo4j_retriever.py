from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.schema import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.db import Neo4jDB


class Neo4jQuery(BaseModel):
    query: str = Field(..., description="要回答的问题")


class Neo4jRetriever(BaseTool):
    name: str = "neo4j_retriever"
    description: str = "用于从Neo4j数据库中检索信息的工具"
    args_schema: Type[BaseModel] = Neo4jQuery

    def __init__(self, db: Neo4jDB, **kwargs) -> None:
        super().__init__(**kwargs)
        self._db = db

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> list[Document]:
        return self._db.get_relevant_chunks(query=query)

    # async def _arun(
    #     self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    # ) -> list[Document]:
    #     return self._db.get_relevant_chunks(query=query)
