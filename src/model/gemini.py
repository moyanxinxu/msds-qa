import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from src.config import hp


class GeminiClient:
    def __init__(self, base_url: str = hp.gemini_base_url) -> None:
        self.base_url = base_url
        self.chat_model = self.get_chat_model()
        self.embed_model = self.get_embed_model()

    def get_chat_model(self, chat_model_name: str = hp.gemini_chat_model) -> ChatOpenAI:
        chat_model = ChatOpenAI(
            model=chat_model_name,
            base_url=self.base_url,
            api_key=SecretStr(os.getenv("GOOGLE_API_KEY", "None")),
        )
        return chat_model

    def get_embed_model(
        self, embed_model_name: str = hp.gemini_embedding_model
    ) -> OpenAIEmbeddings:
        embed_model = OpenAIEmbeddings(
            model=embed_model_name,
            base_url=self.base_url,
            api_key=SecretStr(os.getenv("GOOGLE_API_KEY", "None")),
        )
        return embed_model
