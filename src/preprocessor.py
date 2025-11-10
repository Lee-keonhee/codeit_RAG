
'''
텍스트 청킹 및 전처리
입력 : {
    "doc_id": "RFP_001",
    "text": "문서 전체 텍스트...",
    "metadata": {...}
}

출력 : [
    {
        "chunk_id": "RFP_001_chunk_0",
        "text": "청크1 텍스트...",
        "metadata": {
            "doc_id": "RFP_001",
            "발주기관": "국민연금공단",
            "사업명": "이러닝시스템 구축",
            "chunk_index": 0
        }
    },
    {
        "chunk_id": "RFP_001_chunk_1",
        "text": "청크2 텍스트...",
        "metadata": {...}
    }
]

주요 함수:
def clean_text(text: str) -> str
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]
def process_document(doc: Dict, config: Dict) -> List[Dict]
'''