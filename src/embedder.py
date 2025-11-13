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

import os
import time
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def get_embedding(text: str, model: str = 'text-embedding-3-small') -> List[float]:
    '''
    text를 embedding모델을 활용하여 임베딩 생성

    '''
    response = client.embeddings.create(model=model,
                                        input=text)
    return response.data[0].embedding


def embed_chunks(chunks: List[Dict], model: str = 'text-embedding-3-small') -> List[Dict]:
    '''
    각 청크에 대한 임베딩 생성
    '''
    for chunk in tqdm(chunks, desc="임베딩 생성 중"):  # tqdm 활용
        chunk['embedding'] = get_embedding(chunk['text'], model)  # 's' 제거: embedding (단수)
        time.sleep(0.01)  # API rate limit 방지

    return chunks


if __name__ == '__main__':
    from data_loader import load_documents
    from preprocessor import process_all_documents

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

    all_chunks = embed_chunks(all_chunks)
    print(all_chunks[0])