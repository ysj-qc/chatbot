import os
import chromadb
from chromadb.utils import embedding_functions
from modelscope import snapshot_download
from openai import OpenAI

API_KEY = os.environ.get("API_KEY")
BASE_URL = os.environ.get("BASE_URL")

if not API_KEY or not BASE_URL:
    raise RuntimeError("Please set API_KEY & BASE_URL first before run this script!")

llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

model_dir = snapshot_download("Qwen/Qwen3-Embedding-0.6B")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_dir
)

client = chromadb.PersistentClient(path="./chroma")

collection = client.get_collection(
    name="example_product_docs",
    embedding_function=embed_fn
)

def retrieve(query, top_k = 3):
    #对一条查询返回前三名的文档
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return [
        {"title": m["title"], "url": m["url"], "text": t} 
        for m, t in zip(metas, docs)
    ]

def build_system_prompt(retrieved_docs):
    #检索到的片段+meta要拼接到上下文
    blocks = [f"[{d['title']}]({d['url']})\n({d['text']})" for d in retrieved_docs]
    context = "\n\n".join(blocks)

    return (
        "你是产品客服助手，请基于下面「参考资料」回答用户问题。\n"
        "如果参考资料里没有答案，明确告诉用户「文档中没有相关信息」，不要编造。\n"
        "回答末尾用 Markdown 链接列出引用的文档标题。\n\n"
        f"参考资料：\n{context}"
    )

print("请输入消息开始聊天，输入q以退出：")

while True:
    user_input = input("YOU: ")
    if user_input.strip() == "q":
        break

    retrieved = retrieve(user_input)
    print(f"    [检索] 命中 {[d['title'] for d in retrieved]}")

    messages = [
        {"role": "system", "content": build_system_prompt(retrieved)},
        {"role": "user", "content": user_input},
    ]

    response = llm_client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
        extra_body={"thinking": {"type": "disabled"}},
    )

    print(f"AI: {response.choices[0].message.content}\n")


