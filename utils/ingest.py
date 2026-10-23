# ingest.py - Day2 核心产出
# ingest.py
import os
import shutil
from dotenv import load_dotenv
import dashscope
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

load_dotenv()


# ================= 1. 自定义 DashScope Embeddings =================
class DashScopeEmbeddings(Embeddings):
    """完全绕过 LangChain OpenAI 封装，直接调用 DashScope 原生 SDK"""

    def __init__(self, model="text-embedding-v3", dimension=1024, batch_size=10):
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size  # ← DashScope 原生接口限制最大 10 条/批
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 严格清洗：确保传入的是非空字符串列表
        cleaned = [str(t).strip() for t in texts if t and str(t).strip()]
        if not cleaned:
            return []

        all_embeddings = []
        # ✅ 核心修复：按 batch_size 分批调用 API
        for i in range(0, len(cleaned), self.batch_size):
            batch = cleaned[i: i + self.batch_size]

            resp = dashscope.TextEmbedding.call(
                model=self.model,
                input=batch,
                dimension=self.dimension
            )

            if resp.status_code != 200:
                raise RuntimeError(f"DashScope Embedding Error: {resp.code} - {resp.message}")

            batch_embeddings = [item["embedding"] for item in resp.output["embeddings"]]
            all_embeddings.extend(batch_embeddings)

            print(f"   🔄 已向量化 {min(i + self.batch_size, len(cleaned))}/{len(cleaned)} 条")

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self.embed_documents([text])
        return result[0] if result else []

# ================= 2. 主流程 =================
if __name__ == "__main__":
    # 🧹 清理旧数据（维度/模型切换时必须执行）
    DB_DIR = "./chroma_db"
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print(f"🗑️  已清理旧的向量库: {DB_DIR}")

    # 📄 Step 1: 加载文档

    PDF_PATH = "docs/中华人民共和国民法典_20200528.pdf" # ← 确认此文件存在且为文字版PDF
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"❌ 找不到测试文档: {PDF_PATH}\n请确保 data/ 目录下有文字版 PDF 文件")

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"✅ 加载了 {len(docs)} 页文档")

    # ✂️ Step 2: 文本切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", " "]
    )
    chunks = splitter.split_documents(docs)

    # 二次清洗：移除空块
    valid_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
    print(f"✅ 切分为 {len(valid_chunks)} 个有效文本块 (原始 {len(chunks)} 块)")

    if not valid_chunks:
        raise ValueError("❌ 没有提取到任何有效文本，请检查 PDF 是否为扫描版/图片版")

    # 🔢 Step 3: 向量化 + 存入 Chroma
    embeddings = DashScopeEmbeddings()

    print("⏳ 正在向量化并写入 Chroma...")
    vectorstore = Chroma.from_documents(
        documents=valid_chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="cn_rag_knowledge"
    )

    # ✅ Step 4: 验证
    count = vectorstore._collection.count()
    print(f"\n🎉 Day2 完成！向量库中共有 {count} 条记录")
    print(f"📁 持久化目录: {os.path.abspath(DB_DIR)}")

'''
# 1. 加载文档
loader = unstructured("docs/GB+34459-2017.pdf")
loader = Docx2txtLoader("docs/中华人民共和国刑法_20201226.docx")  # ← 提前放一份测试PDF到 data/ 目录
loader = UnstructuredWordDocumentLoader("docs/中华人民共和国刑法_20201226.docx")  # ← 提前放一份测试PDF到 data/ 目录
loader = PyPDFLoader("docs/中华人民共和国民法典_20200528.pdf")  # ← 提前放一份测试PDF到 data/ 目录
docs = loader.load()
print(f"✅ 加载了 {len(docs)} 页文档")

# 2. 文本切分
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", ".", " "]  # ← 中文分隔符优先
)
chunks = splitter.split_documents(docs)
print(f"✅ 切分为 {len(chunks)} 个文本块")

# 3. 向量化 + 存入 Chroma
embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    check_embedding_ctx_length=False,  # 关键参数
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",  # ← 本地持久化目录
    collection_name="cn_rag_knowledge"
)

# 4. 验证
count = vectorstore._collection.count()
print(f"✅ 向量库中共有 {count} 条记录")
'''