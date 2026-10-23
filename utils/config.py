# utils/config.py
import os

DASHSCOPE_EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")
DASHSCOPE_EMBEDDING_DIM = int(os.getenv("DASHSCOPE_EMBEDDING_DIM", "1024"))
DASHSCOPE_EMBEDDING_BATCH = int(os.getenv("DASHSCOPE_EMBEDDING_BATCH", "10"))

DB_DIR = os.getenv("DB_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cn_rag_knowledge")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")