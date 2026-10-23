# 以通义千问为例（如用GLM请参考智谱官方文档调整）

import os
from dotenv import load_dotenv  # 导入库
load_dotenv()   # 这行必须在 os.getenv() 之前

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
default_headers={"X-DashScope-Async": "disable"}

)

response = client.chat.completions.create(
    model="qwen-flash",
    messages=[{"role": "user", "content": "你好，请用一句话介绍自己"}]
)


print(response.choices[0].message.content)