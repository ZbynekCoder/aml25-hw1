import os
from dotenv import load_dotenv
import logging

from lightrag import LightRAG, QueryParam
from lightrag.llm import zhipu_complete, zhipu_embedding
from lightrag.utils import EmbeddingFunc
from persona_ingest import ingest_persona_csv

WORKING_DIR = "./working_dir"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")
if api_key is None:
    raise Exception("Please set ZHIPU_API_KEY in your environment")


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=zhipu_complete,
    llm_model_name="glm-4-flash",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    embedding_func=EmbeddingFunc(
        embedding_dim=2048,
        max_token_size=8192,
        func=lambda texts: zhipu_embedding(texts),
    ),
)

ingest_persona_csv(rag, "../data/Synthetic-Persona-Chat/data/Synthetic-Persona-Chat_train-mini.csv")

q = "谁有一只名叫 Timothy 的乌龟？请引用设定回答。"
ctx = rag.query(q, QueryParam(mode="naive", only_need_context=True))
print("=== 检索上下文（Top-2） ===\n", ctx)

# 生成回答（基于上述两条设定）
ans = rag.query(q, QueryParam(mode="naive", response_type="Short Answer"))
print("=== 回答 ===\n", ans)