import csv
from typing import Dict, List
from src.lightrag import LightRAG

def ingest_persona_csv(rag: LightRAG, csv_path: str, encoding: str = "utf-8"):
    """
    将 Synthetic-Persona-Chat_test.csv 导入 LightRAG：
    - 每条人物设定作为独立 chunk
    - 在 content 前加标签 [U1]/[U2] 以及 conv 标识，便于检索和追踪
    - 不做实体抽取；直接落到 text_chunks/chunks_vdb
    """
    chunks: List[Dict[str, str]] = []

    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        conv_idx = 0
        for row in reader:
            conv_idx += 1
            # 列名兼容：可能是 user 1 personas / user 2 personas
            u1 = row.get("user 1 personas") or row.get("User 1 personas") or ""
            u2 = row.get("user 2 personas") or row.get("User 2 personas") or ""

            # 拆行；去掉空白
            u1_lines = [x.strip() for x in u1.splitlines() if x.strip()]
            u2_lines = [x.strip() for x in u2.splitlines() if x.strip()]

            # 为每条设定构造 chunk，source_id 用 conv 和角色标识
            for i, line in enumerate(u1_lines):
                content = f"[U1][conv={conv_idx}][idx={i+1}] {line}"
                chunks.append({"content": content, "source_id": f"conv{conv_idx}_U1_{i+1}"})
            for i, line in enumerate(u2_lines):
                content = f"[U2][conv={conv_idx}][idx={i+1}] {line}"
                chunks.append({"content": content, "source_id": f"conv{conv_idx}_U2_{i+1}"})

    if not chunks:
        print(f"[persona_ingest] 未从 {csv_path} 解析到人物设定")
        return

    # 用 LightRAG 的自定义 KG 插入，仅用 chunks（不会跑实体抽取）
    rag.insert_custom_kg({"chunks": chunks})
    print(f"[persona_ingest] 完成导入：{len(chunks)} 条人物设定")
