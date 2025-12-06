import os
import logging
import time
import touch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from  openai import OpenAI, RateLimitError
import httpx
#try with light weight ML model, exclude transformers if not necessary
#from transformers import pipeline


logging.basicConfig(level=logging.INFO)
load_dotenv() # Load environment variables from .env file


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","").strip()
if not OPENAI_API_KEY:
    logging.warning("OpenAI API key is not set. Please configure it properly.")
else:
    logging.info(f"OpenAI API key loaded successfully. : {OPENAI_API_KEY[:5]}***")

client = OpenAI(api_key = OPENAI_API_KEY)
SYSTEM_PROMPT = """
你是一个温柔的心理陪伴助手。
- 提供情绪支持而不是医疗建议
- 不进行诊断
- 使用同理心、积极倾听和开放式提问
- 若用户处于危机（如自杀、自残），温柔提醒寻求现实帮助
"""

CRISIS_KEYWORDS = ["suicide", "自杀", "kill myself", "end my life", "自残"]
CRISIS_MESSAGE = (
    "我注意到你可能有严重的情绪危机，请立即联系专业机构寻求帮助：\n"
    "德国热线: TelefonSeelsorge 0800 111 0 111\n"
    "或联系当地心理咨询师。"
)


# Initialize FastAPI app
app = FastAPI(title = "Your Friend LIMO")

class UserInput(BaseModel):
    text: str

#OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
##try with light weight ML model, exclude transformers if not necessary
#  emotion_classifier = pipeline(
#     "text-classification",
#     model="j-hartmann/emotion-english-distilroberta-base",
#     return_all_scores=True
# )
#try with light weight ML model, exclude transformers if not necessary


def call_oepnai_with_retry(messages, max_retries=3, backoff_factor=2):
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                #model="gpt-3.5-turbo",
                model="gpt-4o-mini",
                messages=messages,
                #max_tokens=150,
                #temperature=0
                max_tokens=200,
                temperature=0.7
            )
            return response
        except RateLimitError:
            wait_time = backoff_factor * (2 ** attempt)
            logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            attempt += 1
    raise HTTPException(status_code=500, detail="Max retries exceeded for OpenAI API calls.")

# POST endpoint to generate SQL query
#@app.post("/chat")
# async def generate_sql(user_input: UserInput):
#     logging.info(f"Received prompt: {user_input.prompt}")

#     if not OPENAI_API_KEY:
#         raise HTTPException(status_code=500, detail="OpenAI API key is not configured.")

#     try:
#         response = call_oepnai_with_retry([
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": user_input.prompt}
#         ])
#         # obtain content
#         reply = getattr(response.choices[0].message, "content", "").strip()
#         if not reply:
#             reply = "Error: No content generated."
#     except Exception as e:
#         logging.error(f"OpenAI request failed: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

#     logging.info(f"Generated SQL query: {reply}")
#     return {"sql_query": reply}

#try with light weight ML model, exclude transformers if not necessary
# @app.post("/chat")
# async def chat(user_input: UserInput):
#     text = user_input.text.strip().lower()

#     # 危机检测
#     if any(word in text for word in CRISIS_KEYWORDS):
#         return {"emotion": "crisis", "reply": CRISIS_MESSAGE}

#     # 情绪分析
#     emotions = emotion_classifier(user_input.text)[0]
#     top_emotion = max(emotions, key=lambda x: x['score'])['label']

#     # 调用 OpenAI 生成心理陪伴回复
#     response = call_oepnai_with_retry([
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": f"{user_input.text}\n\n用户当前情绪可能是：{top_emotion}"}
#     ])

#     reply = getattr(response.choices[0].message, "content", "").strip()
#     if not reply:
#         reply = "抱歉，我暂时无法生成回复。"

#     logging.info(f"Emotion: {top_emotion}, Reply: {reply}")
#     return {"emotion": top_emotion, "reply": reply}
#try with light weight ML model, exclude transformers if not necessary
# -----------------------------
# GET /
# -----------------------------

# -----------------------------
# POST /chat Endpoint
# -----------------------------
@app.post("/chat")
async def chat(user_input: UserInput):
    text = user_input.text.strip().lower()

    # Crisis detection
    if any(word in text for word in CRISIS_KEYWORDS):
        return {"emotion": "crisis", "reply": CRISIS_MESSAGE}

    # Use OpenAI GPT to analyze emotion and generate reply
    prompt = f"""
用户输入文本：
{text}

请帮忙：
1. 简单分析用户当前情绪（用英语或中文回答一个词，如 happy, sad, anxious 等）。
2. 提供温柔心理陪伴回复。
请输出 JSON：
{{"emotion": "情绪标签", "reply": "温柔的心理陪伴回复"}}
"""

    response = call_oepnai_with_retry([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])

    try:
        content = getattr(response.choices[0].message, "content", "").strip()
        import json
        data = json.loads(content)
    except Exception as e:
        logging.error(f"Failed to parse OpenAI response: {e}")
        data = {"emotion": "unknown", "reply": "抱歉，我暂时无法生成回复。"}

    return data

# -----------------------------
# GET / Endpoint
# -----------------------------

@app.get("/")
async def home():
    return {"message": "API is running. Use POST /chat with JSON {'prompt': 'your text'}"}
