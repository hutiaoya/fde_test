from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv  # 导入库
load_dotenv()   # 这行必须在 os.getenv() 之前

app = FastAPI(title="中文知识库问答系统", version="0.1.0")

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # [reference:13]
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
default_headers={"X-DashScope-Async": "disable"}

)


class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat")
async def chat(request: ChatRequest):
    response = client.chat.completions.create(
        model="qwen-flash",
        messages=[{"role": "user", "content": request.question}]
    )
    return ChatResponse(answer=response.choices[0].message.content)


'''在PyCharm终端执行：
uvicorn main:app --reload --port 8000
'''