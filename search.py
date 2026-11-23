import json
import sys
import numpy as np
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer

if len(sys.argv) < 2:
    print("Method: python search.py <查询语句>")
    sys.exit(1)
query = sys.argv[1]

doc_embeddings = np.load("doc_embedding.npy")
with open("documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

model_dir = snapshot_download("Qwen/Qwen3-Embedding-0.6B")
model = SentenceTransformer(model_dir)

query_embedding = model.encode([query], prompt_name="query")
similarities = model.similarity(query_embedding, doc_embeddings).numpy()[0]

top_3 = np.argsort(-similarities)[:3]

print(f"\n查询：{query}")

for rank, index in enumerate(top_3, 1):
    doc = documents[index]
    score = similarities[index]

    print(f"Top {rank} 相似度为 {score:.3f}  下标 {index}")
    print(f"标题：{doc['title']}")
    print(f"链接：{doc['url']}")
    print(f"原文：{doc['text']}")

