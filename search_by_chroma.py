import sys
import chromadb
from chromadb.utils import embedding_functions
from modelscope import snapshot_download

if len(sys.argv) < 2:
    print("用法：python search.py <查询语句>")
    sys.exit(1)
query = sys.argv[1]

model_dir = snapshot_download("Qwen/Qwen3-Embedding-0.6B")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_dir,
)

client = chromadb.PersistentClient(path="./chroma")
collection = client.get_collection(
    name="example_product_docs",
    embedding_function=embed_fn,
)

results = collection.query(query_texts=[query], n_results=3)

docs = results["documents"][0]
metas = results["metadatas"][0]
dists = results["distances"][0]

print(f"\n查询：{query}")
for rank, (text, meta, dist) in enumerate(zip(docs, metas, dists), 1):
    # cosine 距离 = 1 - cosine 相似度
    similarity = 1 - dist
    print(f"  Top {rank} 相似度 {similarity:.3f}")
    print(f"    标题：{meta['title']}")
    print(f"    链接：{meta['url']}")
    print(f"    片段：{text}")

