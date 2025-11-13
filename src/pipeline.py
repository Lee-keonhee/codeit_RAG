'''
전체 RAG 파이프라인 통합
입력 : {
    "query": "사용자 질문",
    "filters": {...},  # optional
    "chat_history": [...]  # optional
}

출력 :{
    "answer": "LLM 생성 답변",
    "sources": [...],
    "retrieved_chunks": [...],
    "metadata": {...}
}

주요 함수:
class RAGPipeline:
    def __init__(self, config_path: str)
    def build_index(self, doc_folder: str, metadata_path: str)
    def query(self, question: str, filters: Dict, chat_history: List) -> Dict
    def run_pipeline(query: str, pipeline: RAGPipeline) -> Dict
'''
import os
from data_loader import load_documents
from preprocessor import process_all_documents
from embedder import embed_chunks
from vector_store import create_vector_store, save_vector_store, load_vector_store

class RAGPipeline:
    def __init__(self, config:dict):
        self.config = config

    def build_index(self,):
        documents = load_documents(self.config['data_dir'], self.config['metadata_path'])
        all_chunks = process_all_documents(documents, self.config)
        all_chunks = embed_chunks(all_chunks)

        if os.path.exists(os.path.join(self.config['vector_store_dir'], 'index.faiss')):
            vectorstore = load_vector_store(self.config['vector_store_dir'])
        else:
            vectorstore = create_vector_store(all_chunks)
            save_vector_store(vectorstore, self.config['vector_store_dir'])

