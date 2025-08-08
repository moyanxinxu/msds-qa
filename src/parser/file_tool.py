import os

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader


class FileChecker:

    def __init__(self) -> None:
        (
            self.supported_suffixes,
            self.supported_processors,
            self.supported_suffix2processor,
        ) = self.get_supported_suffix2processor()

    def get_suffix(self, file_path: str) -> str:
        """
        获取文件的后缀名

        :param file_path: 文件路径

        :return: 文件后缀名
        """
        return os.path.splitext(file_path)[-1].lower()

    def is_suffix_valid(self, file_path: str, suffix: list[str]) -> bool:
        """
        检查文件后缀是否有效

        :param file_path: 文件路径
        :param suffix: 期望的文件后缀

        :return: 如果文件后缀有效，返回 True，否则返回 False
        """
        if self.get_suffix(file_path) in suffix:
            return True
        else:
            return False

    def is_prefix_valid(self, file_path: str, prefix: str) -> bool: ...

    def get_supported_suffix2processor(self):
        suffix2processor = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            "doc": Docx2txtLoader,
        }

        return (
            list(suffix2processor.keys()),
            list(suffix2processor.values()),
            suffix2processor,
        )

    def is_file_valid(self, file_path: str) -> bool:
        """
        检查文件是否存在且后缀有效

        :param file_path: 文件路径

        :return: 如果文件存在且后缀有效，返回 True，否则返回 False
        """
        if not os.path.exists(file_path):
            return False
        if self.is_suffix_valid(file_path, self.supported_suffixes):
            return True
        else:
            return False


class FileProcessorTool:
    def __init__(self):
        self.file_checker = FileChecker()

    def get_file_processors(self, file_paths: list[str]):
        suffixes = [os.path.splitext(file_path)[-1].lower() for file_path in file_paths]
        processors = [
            self.file_checker.supported_suffix2processor[suffix] for suffix in suffixes
        ]
        return processors
