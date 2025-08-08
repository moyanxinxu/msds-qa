import logging
import os
import shutil
from typing import Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.base import VectorStoreRetriever
from tqdm import trange

from src.config import hp
from src.toolkits import check_db_exists, parallel_map


class FaissDB:

    def __init__(self, db_path: str, embed_model: Embeddings, **kwargs) -> None:
        self.db_path: str = db_path
        self.embed_model: Embeddings = embed_model
        self.documents: list[Document] = kwargs.get("documents", [])
        self.db_exists = self.is_db_exists()
        self.db: FAISS = self.load_db()

    def __post_init__(self):
        del self.documents

    def is_db_exists(self) -> bool:
        """检查FAISS数据库是否存在"""
        return os.path.exists(self.db_path) and os.path.isdir(self.db_path)

    def create_db(self, documents: list[Document]) -> FAISS:
        """创建FAISS数据库"""
        try:
            db = FAISS.from_documents(documents[: hp.max_batch_size], self.embed_model)

            for idx in trange(
                hp.max_batch_size,
                len(documents),
                hp.max_batch_size,
                colour="green",
            ):
                db.add_documents(documents[idx : idx + hp.max_batch_size])
                db.save_local(self.db_path)

            os.makedirs(os.path.join(self.db_path, "files"), exist_ok=True)

            parallel_map(
                lambda file: shutil.copy(
                    file, os.path.join(self.db_path, "files", os.path.basename(file))
                ),
                list(set(doc.metadata["source"] for doc in documents)),
                max_workers=10,
                enable_tqdm=True,
            )

            logging.info(f"FAISS数据库已创建并保存到 {self.db_path}")
        except Exception:
            raise ValueError("无法创建FAISS数据库，请检查文档和嵌入模型是否正确")
        return db

    def load_db(self) -> FAISS:
        """加载FAISS数据库"""
        if self.db_exists:
            try:
                db = FAISS.load_local(
                    self.db_path, self.embed_model, allow_dangerous_deserialization=True
                )
                return db
            except Exception:
                raise ValueError(f"无法加载路径位于 {self.db_path} 的FAISS数据库")
        else:
            if not self.documents:
                raise ValueError("数据库不存在且未提供文档，无法创建FAISS数据库。")
            else:
                return self.create_db(self.documents)

    def get_db(self) -> FAISS:
        """获取FAISS数据库实例"""
        return self.db

    def save_db(self, db_path: Optional[str] = None):
        """保存FAISS数据库"""
        db_path = db_path if db_path else self.db_path
        try:
            self.db.save_local(db_path)
        except Exception:
            raise ValueError(f"无法保存FAISS数据库到路径 {db_path}")

    @check_db_exists
    def delete_db(self) -> None:
        """删除FAISS数据库"""
        try:
            shutil.rmtree(self.db_path)
            logging.info(f"FAISS数据库文件 {self.db_path} 已删除")

        except Exception:
            raise ValueError(f"无法删除FAISS数据库文件 {self.db_path}")

    def add_to_db(self, documents: list[Document]) -> None:
        """将文档添加到FAISS数据库"""
        try:
            self.db.add_documents(documents)
            self.save_db(self.db_path)
        except Exception:
            raise ValueError("无法将文档向量化并添加到FAISS数据库")

    def get_retriever(self, k: int = 5) -> VectorStoreRetriever:
        """获取检索器"""
        return self.db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.15},
        )


if __name__ == "__main__":
    from langchain_ollama import OllamaEmbeddings

    embed_model = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url="http://192.168.215.3:11434",
    )

    real_documents_data = [
        {
            "content": "无线路由器能发射无线信号，允许设备连接到互联网。",
            "source": "无线路由器说明书.docx",
        },
        {
            "content": "设备连接到无线路由器后，可以访问互联网。",
            "source": "无线路由器说明书.docx",
        },
        {
            "content": "访问无线网后可以休闲娱乐，观看视频、浏览网页等。",
            "source": "科技生活.pdf",
        },
    ]

    real_docs = [
        Document(page_content=d["content"], metadata={"source": d["source"]})
        for d in real_documents_data
    ]

    db = FaissDB(
        db_path="/root/Documents/msds-qa/kb",
        embed_model=embed_model,
        documents=real_docs,
    )

    db.delete_db()
