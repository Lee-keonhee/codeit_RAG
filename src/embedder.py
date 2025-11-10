
'''
텍스트 임베딩 생성 (OpenAI API 사용)
입력 : [
    {
        "chunk_id": "RFP_001_chunk_0",
        "text": "청크1 텍스트...",
        "metadata": {...}
    }
]

출력 : [
    {
        "chunk_id": "RFP_001_chunk_0",
        "text": "청크1 텍스트...",
        "embedding": [0.123, -0.456, ...],  # 1536차원 벡터:OpenAI text-embedding-3-small의 임베딩 차원
        "metadata": {...}
    }
]

주요 함수:
def get_embedding(text: str, model: str) -> List[float]
def embed_chunks(chunks: List[Dict], model: str) -> List[Dict]
'''