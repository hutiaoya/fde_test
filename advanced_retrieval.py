# advanced_retrieval.py
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from ingest import DashScopeEmbeddings

load_dotenv()

# ================= 1. 基础组件初始化 =================
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=DashScopeEmbeddings(),
    collection_name="cn_rag_knowledge"
)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.7
)

# ================= 2. 策略A: Multi-Query Retriever =================
# 💡 原理：让 LLM 把用户原始问题改写成 3 个不同角度的子查询，分别检索后合并去重
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    include_original=True  # 保留原始查询，防止改写丢失本意
)

# ================= 3. 策略B: HyDE (Hypothetical Document Embeddings) =================
# 💡 原理：先让 LLM 生成一个“假设性回答”，用这个回答的向量去检索，比直接用问题检索更精准
HYDE_PROMPT = ChatPromptTemplate.from_template(
    "请根据以下问题，写一段简短、专业的事实性回答（即使你不确定也要尝试）:\n{question}\n\n假设性回答:"
)

hyde_generator = HYDE_PROMPT | llm | StrOutputParser()


def hyde_retriever(query: str):
    """自定义 HyDE 检索逻辑"""
    # Step 1: 生成假设性文档
    hypothetical_doc = hyde_generator.invoke({"question": query})
    print(f"   🧠 HyDE 假设文档: {hypothetical_doc[:80]}...")

    # Step 2: 用假设文档的语义去检索（而非原始问题）
    return base_retriever.invoke(hypothetical_doc)


# ================= 4. 统一接口：按需选择检索策略 =================
def get_retriever(strategy="multi_query"):
    """
    获取指定策略的检索器
    strategy: "basic" | "multi_query" | "hyde"
    """
    if strategy == "multi_query":
        return multi_query_retriever
    elif strategy == "hyde":
        return RunnableLambda(hyde_retriever)
    else:
        return base_retriever


# ================= 5. 快速测试 =================
if __name__ == "__main__":
    test_query = "这份文档的核心结论是什么？"  # ← 换成你 PDF 中的一个模糊问题

    for strategy in ["basic", "multi_query", "hyde"]:
        print(f"\n{'=' * 60}")
        print(f"🔍 测试策略: {strategy.upper()}")
        print(f"📝 原始问题: {test_query}")

        retriever = get_retriever(strategy)
        docs = retriever.invoke(test_query)

        print(f"✅ 召回 {len(docs)} 个文档块:")
        for i, doc in enumerate(docs):
            content_preview = doc.page_content.replace("\n", " ")[:100]
            print(f"   [{i + 1}] {content_preview}...")