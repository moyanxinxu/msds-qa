from dataclasses import dataclass

from dotenv import load_dotenv

assert load_dotenv("/root/Documents/msds-qa/.env")

@dataclass
class hp:
    ollama_host: str = "http://192.168.215.3:11434"
    # ollama_chat_model: str = "qwen3:0.6b"
    # ollama_chat_model: str = "gemma3:1b"
    # ollama_chat_model: str = "qwen3:1.7b"
    ollama_chat_model: str = "qwen2.5:3b"

    ollama_embedding_model: str = "nomic-embed-text:latest"

    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    # siliconflow_chat_model: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    # siliconflow_chat_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    siliconflow_chat_model: str = "Qwen/Qwen3-8B"
    siliconflow_embedding_model: str = "BAAI/bge-m3"

    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "models/embedding-001"

    max_batch_size: int = 32
    max_chunk_size: int = 256

    knowledge_space: str = "/root/Documents/msds-qa/kb"
    knowledge_file_path: str = "/root/Documents/msds-qa/assets"

    neo4j_bolt_url: str = "bolt://192.168.215.3:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"

    max_retry: int = 1

    searx_host: str = "http://192.168.215.2:8080"
    searx_num_results: int = 5
