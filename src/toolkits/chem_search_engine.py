import os
from typing import Union

import requests
from pydantic import BaseModel, Field

from .funcs import GHSS, parallel_map


class ChemicalsDataSearchEngine:
    def __init__(self) -> None:
        # 国家危险化学品安全公共服务互联网平台
        self.home_url = "https://whpdj.mem.gov.cn"

        self.base_url = "https://whpdj.mem.gov.cn/internet/common/chemical"

        # 查询化学品列表
        self.get_chem_id_url = self.base_url + "/queryChemicalList"
        # 根据化学品唯一标识符查询化学品信息
        self.get_chem_info_url = self.base_url + "/queryChemicalById"

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15",
            "Content-Type": "application/json",
        }

        self.file_dir = "/root/Documents/msds-qa/assets"

        self.no_msds_chemicals = []

    def get_idenDataId(self, chemName: str) -> Union[str, None]:
        """
        精准匹配获取化学品的唯一标识符

        :param chemName: 化学品名称

        :return: 化学品的唯一标识符
        """

        payload = {
            "status": "1",
            "chemName": str(chemName),
            "chemCas": "",
            "chemEnglishName": "",
            "isFuzzy": "0",
            "page": {"current": "1", "size": 10},
        }

        response = requests.post(
            self.get_chem_id_url,
            json=payload,
            headers=self.headers,
        )
        json_object = response.json()

        if json_object["obj"] is not None and json_object["obj"]["records"]:
            # 返回最后一个记录的idenDataId
            # 可能存在多个记录，取最后一个
            return json_object["obj"]["records"][-1]["idenDataId"]
        else:
            return None

    def get_all_ChemNames(self) -> list[str]:
        """
        获取所有化学品的名称

        :return: 化学品的名称列表
        """

        payload = {
            "status": "1",
            "chemName": "",
            "chemCas": "",
            "chemEnglishName": "",
            "isFuzzy": "0",
            "page": {"current": "1", "size": 7173},
        }

        response = requests.post(
            self.get_chem_id_url,
            json=payload,
            headers=self.headers,
        )
        json_object = response.json()

        return parallel_map(
            lambda record: record["chemName"],
            json_object["obj"]["records"],
            max_workers=10,
            enable_tqdm=True,
        )

    def get_chemInfo(self, idenDataId: str) -> dict:
        """
        根据化学品的唯一标识符获取化学品信息

        :param idenDataId: 化学品的唯一标识符

        :return: 化学品信息
        """

        payload = {"idenDataId": idenDataId, "status": "1"}

        response = requests.post(
            self.get_chem_info_url,
            json=payload,
            headers=self.headers,
        )

        # 接口返回的json对象存在值为None的情况，和BaseModel的字段类型不匹配
        json_object = response.json()["obj"]

        for k, v in json_object.items():
            if v is None:
                json_object[k] = ""
        return json_object

    def get_fileInfo(self, response: dict) -> dict:
        """
        获取化学品的安全文件名称

        :param response: 化学品信息的响应

        :return: 包含文件名称和下载地址的字典
        :rtype: dict
        """

        ret = {
            "safetyFileName": response["safetyFileName"],
            "safetyFileUrl": (
                self.home_url + response["safetyFileUrl"]
                if response["safetyFileUrl"]
                else None
            ),
        }
        return ret

    def download_msds_by_name(self, chemName: str) -> None:
        """
        下载化学品的安全数据表

        :param chemName: 化学品名称
        """

        idenDataId = self.get_idenDataId(chemName)

        cond1 = idenDataId is not None
        if cond1:
            chem_info = self.get_chemInfo(idenDataId)
            file_info = self.get_fileInfo(chem_info)

            cond2 = file_info["safetyFileUrl"] is not None

        if cond1 and cond2:
            file_name = file_info["safetyFileName"].split("@")[0]
            file_path = os.path.join(self.file_dir, f"{file_name}.pdf")

            if not os.path.exists(file_path):
                response = requests.get(
                    file_info["safetyFileUrl"],
                    headers=self.headers,
                    allow_redirects=True,
                )
                if response.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(response.content)

                print(f"Download Successfully")
        else:
            self.no_msds_chemicals.append(chemName)

    def download_all_msds(self) -> None:
        """
        下载化学品的安全数据表

        :param chemName: 化学品名称
        """

        chemNames = self.get_all_ChemNames()

        parallel_map(
            self.download_msds_by_name,
            chemNames,
            max_workers=10,
            enable_tqdm=True,
        )

        with open("/root/Documents/msds-qa/scripts/no_msds_chemicals.txt", "a") as f:
            for chem in self.no_msds_chemicals:
                f.write(f"{chem}\n")
        # for chemName in tqdm(chemNames):
        #     self.download_msds_by_name(chemName)

    def test_get_idenDataId(self) -> str:
        chemName = "氟化铵"
        idenDataId = self.get_idenDataId(chemName)
        return idenDataId

    def test_get_chemInfo(self) -> dict:
        idenDataId = "B2BFE2FE-225C-43F6-972D-32335472CAF9"
        chem_info = self.get_chemInfo(idenDataId)
        return chem_info

    def test_get_fileInfo(self) -> dict:
        idenDataId = "B2BFE2FE-225C-43F6-972D-32335472CAF9"
        chem_info = self.get_chemInfo(idenDataId)
        file_info = self.get_fileInfo(chem_info)
        return file_info

    def test_download_msds_by_name(self) -> None:
        chemName = "氟化铵"
        self.download_msds_by_name(chemName)


class ChemInfoModel(BaseModel):
    """
    化学品信息数据类
    """

    model_config = {"extra": "ignore"}
    # idenDataId: str = Field(..., description="化学品的唯一标识符")
    chemName: str = Field(..., description="化学品名称")
    chemEnglishName: str = Field(..., description="化学品英文名称")
    chemCas: str = Field(..., description="化学品CAS号")
    chemAlias: str = Field(..., description="化学品别名")
    ph: str = Field(..., description="pH值")
    meltPoint: str = Field(..., description="熔点")
    boilPoint: str = Field(..., description="沸点")
    relativeDensity: str = Field(..., description="相对密度")
    relativeVaporDensity: str = Field(..., description="相对蒸气密度")
    vaporPressure: str = Field(..., description="蒸气压")
    combustionHeat: str = Field(..., description="燃烧热")
    limitTemp: str = Field(..., description="临界温度")
    limitPress: str = Field(..., description="临界压力")
    octMatModulus: str = Field(..., description="辛醇-水分配系数")
    flashPoint: str = Field(..., description="闪点")
    autoIgnitionTemp: str = Field(..., description="自燃温度")
    exploLowerLimit: str = Field(..., description="爆炸下限")
    exploUpperLimit: str = Field(..., description="爆炸上限")
    breakdownTemp: str = Field(..., description="分解温度")
    viscosity: str = Field(..., description="粘度")
    solubilty: str = Field(..., description="溶解度")
    density: str = Field(..., description="密度")
    specialDanger: str = Field(..., description="燃烧与爆炸危险性")
    physcialChemDanger: str = Field(..., description="活性反应")
    healthHazard: str = Field(..., description="中毒表现")
    careerContactLimit: str = Field(..., description="职业接触限值")
    environmentHazard: str = Field(..., description="环境危害")
    firstMeasure: str = Field(..., description="急救措施")
    leakageMeasure: str = Field(..., description="泄漏应急措施")
    adviceProjectExtinguish: str = Field(..., description="灭火方法")
    avoidMater: str = Field(..., description="避免接触的物质")
    acuteToxicity: str = Field(..., description="毒性")
    riskCategory: str = Field(..., description="危险性类别")
    riskDesc: str = Field(..., description="危险性说明")
    warnWord: str = Field(..., description="GHS警示词")
    pictograms: Union[str, list[str]] = Field(..., description="象形图")
    apperanceShape: str = Field(
        ..., description="外观与性状"
    )  # 接口字段为apperanceShape，非appearanceShape
    recordList: Union[list[dict], dict[str, list]] = Field(
        ..., description="化学品目录"
    )

    @staticmethod
    def get_descriptions() -> list[str]:
        """获取所有描述"""
        return [
            # "化学品的唯一标识符",
            "化学品名称",
            "化学品英文名称",
            "化学品CAS号",
            "化学品别名",
            "pH值",
            "熔点",
            "沸点",
            "相对密度",
            "相对蒸气密度",
            "蒸气压",
            "燃烧热",
            "临界温度",
            "临界压力",
            "辛醇-水分配系数",
            "闪点",
            "自燃温度",
            "爆炸下限",
            "爆炸上限",
            "分解温度",
            "粘度",
            "溶解度",
            "密度",
            "燃烧与爆炸危险性",
            "活性反应",
            "中毒表现",
            "职业接触限值",
            "环境危害",
            "急救措施",
            "泄漏应急措施",
            "灭火方法",
            "避免接触的物质",
            "毒性",
            "危险性类别",
            "危险性说明",
            "GHS警示词",
            "象形图",
            "外观与性状",
            "化学品目录",
        ]

    def get_formated_info(self) -> dict[str, str]:
        """
        获取格式化的化学品信息

        1. 替换换行和回车符为空格
        2. 对象形图进行中文描述转换

        :return: 格式化后的化学品信息字典
        """
        dic = {}

        # 1. 替换换行和回车符为空格
        for description, value in zip(
            self.get_descriptions(), list(self.model_dump().values())
        ):
            if isinstance(value, str):
                dic[description] = value.strip().replace("\n", " ").replace("\r", " ")
            else:
                dic[description] = value
        in_records, not_in_records = [], []  # 在哪些目录、不在哪些目录

        for record in dic["化学品目录"]:
            if record["isExist"] == "1":
                in_records.append(record["contentName"])
            else:
                not_in_records.append(record["contentName"])
        dic["化学品目录"] = {
            "该化学品在以下化学品目录中为危险化学品": in_records,
            "该化学品在以下化学品目录中未被记录": not_in_records,
        }

        # 2. 对象形图词进行中文描述转换
        dic["象形图"] = [
            GHSS.get(pictogram, pictogram) for pictogram in dic["象形图"].split(",")
        ]
        return dic


if __name__ == "__main__":
    search_engine = ChemicalsDataSearchEngine()
    idenDataId = search_engine.test_get_idenDataId()
    chem_info = search_engine.get_chemInfo(idenDataId)
    file_info = search_engine.get_fileInfo(chem_info)

    print("化学品信息查询结果:")
    print(f"化学品的唯一标识符: {idenDataId}")
    print(f"化学品信息: {chem_info['chemName']}")
    print(f"安全文件名称: {file_info['safetyFileName']}")
    print(f"安全文件下载地址: {file_info['safetyFileUrl']}")
    print(f"{ChemInfoModel(**chem_info).get_formated_info()}")
    # search_engine.download_all_msds()
