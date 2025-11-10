'''
역할: Vector DB 구축 및 관리 (FAISS or Chroma)
입력 : [
    {
        "chunk_id": "RFP_001_chunk_0",
        "text": "청크1 텍스트...",
        "embedding": [0.123, -0.456, ...],
        "metadata": {...}
    }
]
출력 :Vector DB 인덱스 저장 (파일 시스템)
      검색 시: 유사도 높은 청크 리스트 반환

주요 함수:
def create_vector_store(chunks: List[Dict]) -> VectorStore
def save_vector_store(store: VectorStore, path: str)
def load_vector_store(path: str) -> VectorStore
def search(query_embedding: List[float], top_k: int, filters: Dict) -> List[Dict]
'''