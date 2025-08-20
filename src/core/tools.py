from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import BaseTool, Tool
from langchain_core.vectorstores import VectorStore

from src.db import Neo4jDB
from src.retriever import ChemInfoRetriever, Neo4jRetriever


class ToolSet:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_faiss_retriever_tool(db: VectorStore, name: str, description: str) -> Tool:
        """创建一个检索工具"""
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.15},
        )
        retriever = create_retriever_tool(
            retriever=retriever,
            name=name,
            description=description,
        )
        return retriever

    @staticmethod
    def get_neo4j_retriever_tool(
        db: Neo4jDB,
        name: str = "neo4j_retriever",
        description: str = "用于从Neo4j数据库中检索信息的工具",
    ) -> BaseTool:
        """创建一个Neo4j检索工具"""
        return Neo4jRetriever(db=db, name=name, description=description)

    @staticmethod
    def get_nrcc_chem_info_tool() -> BaseTool:
        """nrcc化学品信息检索工具"""
        return ChemInfoRetriever()
