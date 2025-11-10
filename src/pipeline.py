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