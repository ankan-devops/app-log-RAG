import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from datetime import datetime

load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

def convert_to_epoch(timestamp):
    return int(datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp())

def init_milvus_vectorstore(expr):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", encode_kwargs={'normalize_embeddings': True, 'prompt': 'Represent this log message for searching application logs: '})
    # milvus = Milvus(embedding_function=embedding_model, collection_name="log_embeddings", auto_id=True, connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT})
    milvus = Milvus(embedding_function=embedding_model, collection_name="log_embeddings", auto_id=True, connection_args={"uri": "./demo.db"})
    retriever = milvus.as_retriever(search_type="mmr", search_kwargs={"k": 3, "expr": expr})
    return retriever

def milvus_retrieval(query, start_time, end_time):
    start = convert_to_epoch(start_time)
    end = convert_to_epoch(end_time)
    expr = f"timestamp > {start} and timestamp < {end}"
    res = init_milvus_vectorstore(expr).invoke(query)
    print (res)

milvus_retrieval("find errors in FHIR", "2024-10-21T10:00:00.000Z", "2024-10-21T20:00:00.000Z")