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

from typing import List, Dict
import numpy as np
import faiss
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

def create_faiss_index(chunks: List[Dict]):
    '''
    청크들을 통해 VectorStore 생성
    '''
    # dim = len(chunks[0]['embedding'])
    dimension = 1536 # text-embedding-3-small의 차원

    # L2 거리 기반 인덱스
    index = faiss.IndexFlatL2(dimension)

    # 모든 청크에 대한 embedding 값을 numpy 형태로 저장
    embeddings = np.array([chunk['embedding'] for chunk in chunks])

    # 벡터 저장소에 추가
    index.add(embeddings)
    print(f'저장된 벡터의 개수 : {index.ntotal}')

    return index


def save_index_chunks(index: faiss.Index, chunks: List[Dict], path: str):
    '''
    FAISS index와 chunks를 파일로 저장
    '''
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)

    # 1. FAISS index 저장
    faiss.write_index(index, f"{path}.index")

    # 2. chunks 저장 (pickle 사용)
    with open(f"{path}.pkl", 'wb') as f:
        pickle.dump(chunks, f)

    print(f"Vector store 저장 완료: {path}")


def load_index_chunks(path:str):
    '''
    저장된 FAISS index와 chunks 불러오기
    '''
    try:
        # 1. FAISS index 불러오기
        index = faiss.read_index(f"{path}.index")

        # 2. chunks 불러오기
        with open(f"{path}.pkl", 'rb') as f:
            chunks = pickle.load(f)

        print(f"Vector store 불러오기 완료: {path}")
        print(f"저장된 벡터 개수: {index.ntotal}")

    except FileNotFoundError as e:
        # 예외 객체 e를 사용하여 상세 정보를 출력
        print("❌ 파일을 열 수 없습니다.")
        print(f"오류 상세: {e}")

    except Exception as e:
        # FileNotFoundError 외의 모든 일반 오류를 잡고 상세 정보를 출력
        print("❌ 알 수 없는 오류가 발생했습니다.")
        print(f"오류 타입: {type(e)}")
        print(f"오류 상세: {e}")

    return index, chunks

def create_vector_store(chunks:List[Dict]):
    '''
    이미 임베딩된 청크로 LangChain VectorStore 생성
    '''
    # 1. 텍스트, 임베딩, 메타데이터 분리
    texts = [chunk['text'] for chunk in chunks]
    embeddings_list = [chunk['embedding'] for chunk in chunks]
    metadatas = [chunk.get('metadata', {}) for chunk in chunks]

    # 2. 임베딩 모델 초기화 (검색 시 쿼리 임베딩용)
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )

    # 3. FAISS VectorStore 생성
    text_embedding_pairs = list(zip(texts, embeddings_list))
    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=embedding_model,
        metadatas=metadatas
    )

    return vectorstore

def save_vector_store(vectorstore, path:str):
    '''
    LangChain VectorStore를 로컬에 저장
    '''
    vectorstore.save_local(path)
    print(f"VectorStore 저장 완료: {path}")

def load_vector_store(path: str) -> FAISS:
    '''
    저장된 VectorStore 불러오기
    '''
    # 임베딩 모델 초기화 (불러올 때도 필요)
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )

    # VectorStore 불러오기
    vectorstore = FAISS.load_local(
        path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True  # pickle 역직렬화 허용
    )

    print(f"VectorStore 불러오기 완료: {path}")

    return vectorstore

if __name__ == '__main__':
    from data_loader import load_documents
    from preprocessor import process_all_documents
    from embedder import embed_chunks

    # 설정
    config = {
        'data_dir': '../data/raw/files',
        'metadata_path': '../data/raw/data_list.csv',
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'vector_store_dir': '../vector_store',
        # 'vector_store_path': '../vector_store/vector_stor',

    }

    # 문서 로드
    documents = load_documents(config['data_dir'], config['metadata_path'])

    # 전체 문서 처리 ⭐
    all_chunks = process_all_documents(documents, config)

    all_chunks = embed_chunks(all_chunks)

    # index = create_faiss_index(all_chunks)

    # save_index_chunks(config['vector_store_path'])

    # index, chunks = load_index_chunks(config['vector_store_path'])

    vectorstore = create_vector_store(all_chunks)
    # dir_name = os.path.dirname(config['vector_store_path'])
    dir_name = config['vector_store_dir']
    save_vector_store(vectorstore, dir_name)
    vectorstore = load_vector_store(dir_name)

    print(vectorstore)

