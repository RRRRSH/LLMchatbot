from typing import List
from langchain_core.embeddings import Embeddings
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os
import streamlit as st

""" _ = load_dotenv(find_dotenv())    # read local .env file """

class AliyunEmbeddings(Embeddings):
    """`Aliyun Embeddings` embedding models."""
    
    def __init__(self, model: str = "text-embedding-v4", dimensions: int = 1024):
        """
        实例化 AliyunEmbeddings
        """

        self.client = OpenAI(
            api_key=st.secrets["DASHSCOPE_API_KEY"] if "DASHSCOPE_API_KEY" in st.secrets else os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = "text-embedding-v4"
        self.dimensions = dimensions
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.
        """
        result = []
        # 假设 batch size 为 64
        for i in range(0, len(texts), 64):
            batch_texts = texts[i:i+64]
            embeddings = self.client.embeddings.create(
                model=self.model,
                input=batch_texts,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            result.extend([data.embedding for data in embeddings.data])
        return result
    
    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.
        """
        return self.embed_documents([text])[0]