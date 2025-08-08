from src.config import hp
from src.db import FaissDB
from src.model import OllamaClient
from src.parser import DocxParser
from src.toolkits import get_files_from_kb_space

if __name__ == "__main__":
    client = OllamaClient()
    files = get_files_from_kb_space(hp.knowledge_file_path)
    docx_parser = DocxParser(files)
    documents = docx_parser.invoke()

    db = FaissDB(
        db_path="/root/Documents/msds-qa/kb",
        embed_model=client.get_embed_model(),
        documents=documents,
    )

    db.get_db()
