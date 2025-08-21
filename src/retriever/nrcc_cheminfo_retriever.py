from typing import Type

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.toolkits import ChemicalsDataSearchEngine, ChemInfoModel


class ChemInfoQuery(BaseModel):
    chem_name: str = Field(..., description="要查询的化学品名称")


class ChemInfoRetriever(BaseTool):
    name: str = "ChemInfoRetriever"
    description: str = (
        "向用户提供包括化学品基本信息，标签要素，理化特性（危险性类别，GHS警示词，象形图，危险性说明等），危害信息，应急处置措施（急救措施，泄漏应急措施，灭火方法等），安全技术说明书，msds等信息的检索工具。"
    )

    search_engine: ChemicalsDataSearchEngine = ChemicalsDataSearchEngine()
    args_schema: Type[BaseModel] = ChemInfoQuery

    def _run(self, chem_name: str) -> ToolMessage:
        """执行化学品信息查询"""
        chem_id = self.search_engine.get_idenDataId(chem_name)

        if chem_id:
            chem_info = self.search_engine.get_chemInfo(chem_id)
            return ChemInfoModel(**chem_info).get_formated_info()
        return None
