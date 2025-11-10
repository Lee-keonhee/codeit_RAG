'''
쿼리 기반 관련 문서 검색
입력 : {
    "query": "국민연금공단 이러닝시스템 요구사항은?",
    "filters": {
        "발주기관": "국민연금공단"  # optional
    },
    "top_k": 5
}
출력 :[
    {
        "chunk_id": "RFP_001_chunk_3",
        "text": "관련 텍스트...",
        "score": 0.89,
        "metadata": {...}
    },
    ...
]

주요 함수:
def retrieve(query: str, vector_store: VectorStore, top_k: int, filters: Dict) -> List[Dict]
def rerank_results(results: List[Dict]) -> List[Dict]  # optional
'''