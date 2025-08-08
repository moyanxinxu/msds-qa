import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Callable

from json_repair import json_repair
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


GHSS: dict[str, str] = {
    "GHS01": "爆炸品",
    "GHS02": "易燃品",
    "GHS03": "氧化剂",
    "GHS04": "压缩气体",
    "GHS05": "腐蚀品",
    "GHS06": "急性毒性",
    "GHS07": "有害品",
    "GHS08": "健康危害",
    "GHS09": "环境危害",
}

# def handler_exception(func):
#     """
#     异常处理修饰器
#     """

#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except Exception as e:
#             logging.error(f"Error in {func.__name__}: {e}")
#             raise

#     return wrapper


def test_it(func):  #

    @wraps(func)
    def wrapper(*args, **kwargs):

        result = func(*args, **kwargs)

        logging.info(f"{func.__name__}() return {result}")
        return result

    return wrapper


def check_db_exists(func: Callable):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_db_exists:
            raise ValueError("数据库不存在，无法执行此操作")
        return func(self, *args, **kwargs)

    return wrapper


def parallel_map(
    func: Callable, container: list, max_workers=10, enable_tqdm: bool = False
):
    """
    并行映射函数，使用线程池执行函数

    :param func: 要执行的函数

    :param container: 要处理的容器

    :param max_workers: 最大工作线程数

    :return: 函数执行结果列表
    """

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if enable_tqdm:
            results = list(
                tqdm(
                    executor.map(func, container),
                    total=len(container),
                    desc="Loading",
                    colour="green",
                )
            )
        else:
            results = list(executor.map(func, container))
    return results


def get_files_from_kb_space(kb_path: str) -> list[str]:
    """
    获取指定路径下的所有文件

    :param kb_path: 知识库路径

    :return: 知识文件列表
    """
    from src.parser import FileChecker

    file_checker = FileChecker()
    if not os.path.exists(kb_path):
        raise ValueError(f"路径 {kb_path} 不存在")

    files = []
    for file in os.listdir(kb_path):
        file_path = os.path.join(kb_path, file)
        if file_checker.is_file_valid(file_path):
            files.append(file_path)

    return files


def get_json_from_str(text: str) -> dict | None:
    """
    从字符串中提取JSON数据

    :param text: 包含JSON数据的字符串。
    :return: 解析后的JSON对象（字典），如果解析失败则返回None。
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            repaired_json = json_repair.loads(json_str)
            return repaired_json if isinstance(repaired_json, dict) else None
        except Exception as E:
            logging.error(f"Failed to repair JSON: {E}")
            return None
