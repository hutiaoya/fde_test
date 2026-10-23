from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os, time

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    check_embedding_ctx_length=False,
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

print("⏳ 正在调用 Embedding API...")
start = time.time()
result = embeddings.embed_query("测试中文向量化")
print(f"✅ 成功！耗时 {time.time()-start:.2f}s，向量维度: {len(result)}")