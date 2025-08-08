import itertools
import os

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from src.config import hp
from src.toolkits import parallel_map

# class PdfParser:
#     def __init__(self, files: list[str]):
#         self.loader = PyPDFLoader
#         self.files: list[str] = self._check_pdf_files(files)

#     def _check_pdf_files(self, files: list[str] | str) -> list[str]:

#         files = [files] if isinstance(files, str) else files
#         return files

#     def invoke(self) -> list[Document]:
#         documents = [self.loader(file).load_and_split(self.text_splitter) for file in self.files]  # fmt: skip
#         documents = list(itertools.chain.from_iterable(documents))
#         return documents


class MsdsParser:
    def __init__(self, files: list[str] | str):
        self.files: list[str] = files if isinstance(files, list) else [files]
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=hp.max_chunk_size)  # fmt: skip
        self.loader = PyPDFLoader

    def format_context(self, context: Document) -> Document:
        source = context.metadata["source"]
        file_name = os.path.basename(source)
        context.page_content = f"<{file_name}>\n: {context.page_content}"
        return context

    def invoke(self) -> list[Document]:

        def load_and_format(file) -> list[Document]:
            docs = self.loader(file).load_and_split(self.text_splitter)
            return [self.format_context(doc) for doc in docs]

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
