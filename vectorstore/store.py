import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============ Embedding 配置（支持环境变量） ============
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
DASHSCOPE_API_KEY = os.getenv(
    "DASHSCOPE_API_KEY",
    "sk-ws-H.EIYYXDP.4hX0.MEUCIQDqWcsB9X63I_PNZ9BKK6E4Rx3VKdfXJ0SKQa71FxBTLAIgEf_oVycY8p53UzvXC6fTk71c8G2KhK1IRo5v_6QkjsI",
)

embedding_model = DashScopeEmbeddings(
    model=EMBEDDING_MODEL,
    dashscope_api_key=DASHSCOPE_API_KEY,
)
def create_vector_db():
    loader = TextLoader(
        "./knowlege/learn.txt",
        encoding="utf-8"
    )
    document = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    # 切片完成的
    split_docs = splitter.split_documents(document)
    db = Chroma.from_documents(
        split_docs,
        embedding_model=embedding_model,
        persist_directory="./vectorstore/chroma_db",
    )
    return db
def search_knowledge(question):
    db = Chroma(
        persist_directory="./vectorstore/chroma_db",
        embedding_function=embedding_model,
    )
    docs  = db.similarity_search(
        question,
        k=3
    )
    return docs