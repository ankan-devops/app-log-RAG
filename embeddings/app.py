import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from elasticsearch import Elasticsearch
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from langchain.schema import Document
from datetime import datetime


load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = os.getenv("ES_PORT", "41443")
ES_USER = os.getenv("ES_USER", "")
ES_PASS = os.getenv("ES_PASS", "")
ES_INDEX = os.getenv("ES_INDEX", "logstash-2024.10.21")

THREAD_SIZE = int(os.getenv("THREAD_SIZE", multiprocessing.cpu_count()))

c = 0


def convert_to_epoch(timestamp):
    return int(datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp())

def init_milvus_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", encode_kwargs={'normalize_embeddings': True, 'prompt': 'Represent this log message for searching application logs: '})
    #milvus = Milvus(embedding_function=embedding_model, collection_name="log_embeddings", auto_id=True, connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT})
    milvus = Milvus(embedding_function=embedding_model, collection_name="log_embeddings", auto_id=True, connection_args={"uri": "./demo.db"})
    return milvus

def preprocess_logs(log_data):
    log_message = log_data['_source'].get('log')
    log_level = log_data['_source'].get('log_level', 'INFO')
    timestamp = log_data['_source'].get('@timestamp')

    kubernetes_data = log_data['_source'].get('kubernetes', {})
    container_name = kubernetes_data.get('container_name', 'N/A')
    app = kubernetes_data.get('labels', {}).get('app', 'N/A')
    namespace = kubernetes_data.get('namespace_name', 'N/A')

    log_message_clean = log_message.replace(f'[{log_level}]', '').strip()

    full_log_message = (f"[{log_level}] {log_message_clean} | "
                        f"Container: {container_name}, app: {app}, Namespace: {namespace}, Timestamp: {timestamp}")

    timestamp_epoch = convert_to_epoch(timestamp)

    metadata = {
        "log_id": log_data['_id'],
        "timestamp": timestamp_epoch,
        "container_name": container_name,
        "app": app,
        "namespace": namespace
    }

    return Document(page_content=full_log_message, metadata=metadata)


def store_logs_in_milvus(docs, milvus_vectorstore):
    log_documents = [preprocess_logs(doc) for doc in docs]
    milvus_vectorstore.add_documents(log_documents)

    global c
    c += len(log_documents)
    print(f"Inserted {len(log_documents)} documents. Total: {c}")


def process_documents_parallel(documents, milvus_vectorstore, max_threads=THREAD_SIZE):
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for i in range(0, len(documents), max_threads):
            batch = documents[i:i + max_threads]
            futures.append(executor.submit(store_logs_in_milvus, batch, milvus_vectorstore))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f'Error processing document: {exc}')


if ES_USER and ES_PASS:
    es = Elasticsearch([f"http://{ES_USER}:{ES_PASS}@{ES_HOST}:{ES_PORT}/"])
else:
    es = Elasticsearch([f"http://{ES_HOST}:{ES_PORT}/"])

scroll_timeout = '2m'
scroll_size = 1000

milvus_vectorstore = init_milvus_vectorstore()

response = es.search(
    index=ES_INDEX,
    body={
        "size": scroll_size,
        "query": {
            "match_all": {}
        }
    },
    scroll=scroll_timeout
)

scroll_id = response['_scroll_id']
documents = response['hits']['hits']

while len(documents) > 0:
    process_documents_parallel(documents, milvus_vectorstore, max_threads=THREAD_SIZE)
    response = es.scroll(scroll_id=scroll_id, scroll=scroll_timeout)
    scroll_id = response['_scroll_id']
    documents = response['hits']['hits']
