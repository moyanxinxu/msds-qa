import itertools

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader

from src.config import hp
from src.toolkits import parallel_map


class DocxParser:
    def __init__(self, files: list[str] | str):
        self.files: list[str] = files if isinstance(files, list) else [files]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=hp.max_chunk_size
        )
        self.loader = Docx2txtLoader

    def invoke(self) -> list[Document]:

        def load_and_format(file) -> list[Document]:
            docs = self.loader(file).load_and_split(self.text_splitter)
            return docs

        documents = list(
            itertools.chain.from_iterable(
                parallel_map(
                    load_and_format,
                    self.files,
                    max_workers=10,
                    enable_tqdm=True,
                )
            )
        )
        return documents
