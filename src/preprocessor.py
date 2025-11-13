
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
import re
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

def clean_text(text: str) -> str:
    """
    텍스트 정제

    Args:
        text: 원본 텍스트

    Returns:
        정제된 텍스트
    """
    # 1. null 바이트 제거
    text = text.replace('\x00', '')

    # 2. 제어 문자 제거 (개행, 탭은 유지)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # 3. 필요한 유니코드 문자만 유지
    # ASCII + 한글 + 로마숫자 + 원문자 + 특수기호
    # text = re.sub(r'[^\u0000-\u007F\uAC00-\uD7A3]+', '', text)
    text = re.sub(r'[^\u0000-\u007F\uAC00-\uD7A3\u2160-\u217F\u2460-\u24FF\u2022\u25A0-\u25FF]+', '', text)

    # 4. 개행 정리
    text = re.sub(r'\r\n', '\n', text)

    # 5. 공백 정리
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' +\n', '\n', text)
    text = re.sub(r' *\n *', '\n', text)

    # 6. 연속 개행 정리 (최대 3개)
    text = re.sub(r'\n{3,}', '\n\n\n', text)

    # 7. 앞뒤 공백 제거
    text = text.strip()

    return text


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    텍스트를 청크로 분할

    Args:
        text: 분할할 텍스트
        chunk_size: 청크 크기
        chunk_overlap: 청크 간 중첩 크기

    Returns:
        청크 리스트
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    chunks = splitter.split_text(text)
    return chunks


def process_document(document: Dict, config: Dict) -> List[Dict]:
    """
    문서를 처리하여 청크 리스트 생성

    Args:
        document: 문서 딕셔너리
            {
                'doc_id': str,
                'text': str,
                'metadata': dict
            }
        config: 설정 딕셔너리
            {
                'chunk_size': int,
                'chunk_overlap': int
            }

    Returns:
        청크 딕셔너리 리스트
    """
    # 설정 추출
    chunk_size = config['chunk_size']
    chunk_overlap = config['chunk_overlap']

    # 1. 텍스트 정제
    cleaned_text = clean_text(document['text'])

    # 2. 텍스트 청킹
    text_chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)

    # 3. 청크 딕셔너리 생성
    chunks = []
    for chunk_index, chunk_content in enumerate(text_chunks):
        chunk_dict = {
            'chunk_id': f"{document['doc_id']}_chunk_{chunk_index}",
            'text': chunk_content,
            'metadata': {
                **document['metadata'],  # 원본 metadata 복사
                'doc_id': document['doc_id'],
                'chunk_index': chunk_index,
                'total_chunks': len(text_chunks)
            }
        }
        chunks.append(chunk_dict)
    return chunks

def process_all_documents(documents: List[Dict], config: Dict) -> List[Dict]:
    """
    모든 문서 처리

    Args:
        documents: 문서 리스트
        config: 설정 딕셔너리

    Returns:
        전체 청크 리스트
    """
    all_chunks = []

    print(f"📄 Processing {len(documents)} documents...")

    for doc in tqdm(documents, desc="Chunking"):
        try:
            doc_chunks = process_document(doc, config)
            all_chunks.extend(doc_chunks)
        except Exception as e:
            print(f"\n❌ Error processing {doc.get('doc_id', 'unknown')}: {e}")
            continue

    print(f"✅ 총 {len(documents)}개 문서 → {len(all_chunks)}개 청크 생성")

    # 통계 출력
    avg_chunks = len(all_chunks) / len(documents) if documents else 0
    print(f"📊 문서당 평균 청크 수: {avg_chunks:.1f}")

    return all_chunks


if __name__ == '__main__':
    from data_loader import load_documents

    # 설정
    config = {
        'data_dir': '../data/raw/files',
        'metadata_path': '../data/raw/data_list.csv',
        'chunk_size': 1000,
        'chunk_overlap': 200,
    }

    # 문서 로드
    documents = load_documents(config['data_dir'], config['metadata_path'])

    # 전체 문서 처리 ⭐
    all_chunks = process_all_documents(documents, config)

    # 확인
    print(f"\n첫 번째 청크 확인:")
    print(f"chunk_id: {all_chunks[0]['chunk_id']}")
    print(f"text 길이: {len(all_chunks[0]['text'])}")
    print(f"metadata: {all_chunks[0]['metadata']}")