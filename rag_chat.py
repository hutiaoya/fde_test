# rag_chat.py
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from openai import OpenAI
from langchain_openai import ChatOpenAI

# ⚠️ 注意：这里复用 Day2 的自定义 Embeddings 类
# 为避免代码冗余，建议将 Day2 的 DashScopeEmbeddings 抽取到 utils/embeddings.py
# 此处为演示完整流程，直接导入或复制该类
# from ingest import DashScopeEmbeddings
from utils.ingest import DashScopeEmbeddings

load_dotenv()

# ================= 1. 初始化组件 =================
# 加载 Day2 生成的向量库
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=DashScopeEmbeddings(),
    collection_name="cn_rag_knowledge"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 检索 Top-3 相关块

# 初始化 DashScope 聊天模型 (继续使用 OpenAI SDK 兼容模式)
# ✅ 正确写法：使用 LangChain 的 ChatOpenAI

llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",       # ← 在这里指定模型
    temperature=0.7,         # RAG 场景建议 0.5~0.7，平衡准确性与流畅度
    max_tokens=1024          # 防止生成过长截断
)


# ================= 2. 构建 RAG Prompt =================
# 💡 核心：明确告诉模型“只根据提供的上下文回答”
RAG_PROMPT_TEMPLATE = """你是一个专业的知识库问答助手。请严格根据以下【参考上下文】回答用户问题。
如果上下文中没有相关信息，请直接回复“抱歉，知识库中未找到相关内容”，不要编造答案。

【参考上下文】:
{context}

【用户问题】: {question}

【回答】:"""



prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# ================= 3. 组装 RAG 链 (LCEL) =================
def format_docs(docs):
    """将检索到的多个文档块拼接为纯文本"""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# ================= 4. 交互式问答测试 =================
if __name__ == "__main__":
    print("🤖 RAG 知识库问答系统已就绪 (输入 'quit' 退出)")
    print("-" * 50)

    while True:
        query = input("\n💬 你的问题: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("👋 再见!")
            break
        if not query:
            continue

        print("🔍 正在检索并生成回答...")
        answer = rag_chain.invoke(query)
        print(f"\n✅ 回答:\n{answer}")