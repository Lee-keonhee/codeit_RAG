import os, json, yaml
from typing import Dict, List, Any, Optional
from data_loader import load_documents
from preprocessor import process_all_documents
from embedder import embed_chunks
from vector_store import create_vector_store, save_vector_store, load_vector_store
from retriever import retrieve
from generator_test import RAGGenerator

class RAGPipeline:
    """
    전체 RAG 파이프라인 총괄
    build_index() → 문서 → 청크 → 임베딩 → VectorStore 저장
    query() → 검색 + LLM 답변
    """
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.top_k = self.config.get('retrieval', {}).get('top_k', 5)
        self.vector_store_path = self.config.get('paths', {}).get('vector_store_dir', '../vector_store')
        self.vector_store = None # 로드된 벡터스토어를 저장할 변수
        
        self.generator_test = RAGGenerator(
            model=self.config.get('openai', {}).get('chat_model', 'gpt-5-mini'),
            response_type=self.config.get("openai", {}).get("response_type", "detailed")
        )

    def _load_config(self, config_path: Optional[str]):
        if config_path is None:
            return {}

        if not os.path.exists(config_path):
            abs_path = os.path.abspath(config_path)
            raise FileNotFoundError(f"Config 파일이 존재하지 않습니다: {abs_path}")

        # YAML or JSON 자동 판별
        ext = os.path.splitext(config_path)[1].lower()

        if ext in [".yaml", ".yml"]:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        elif ext == ".json":
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            raise ValueError("Config 파일은 .yaml/.yml/.json 만 지원합니다.")

    def build_index(self, doc_folder: str = 'data_dir', metadata_path: str = 'metadata_path'):
        try:
            index_file_path = os.path.join(self.vector_store_path, 'index.faiss')

            # 기존 인덱스 로드 시도
            if os.path.exists(index_file_path):
                print(f"기존 벡터스토어 로드: {self.vector_store_path}")
                self.vector_store = load_vector_store(self.vector_store_path) 
                return

            # 인덱스 생성 및 저장
            print("새 인덱스 생성")
            
            # config 딕셔너리에서 경로 키를 사용하여 실제 경로를 가져옵니다.
            documents = load_documents(self.config['paths'][doc_folder], self.config['paths'][metadata_path])
            all_chunks = process_all_documents(documents, self.config)
            all_chunks = embed_chunks(all_chunks)

            self.vector_store = create_vector_store(all_chunks)
            save_vector_store(self.vector_store, self.vector_store_path)
            print(f"새 벡터스토어 생성 및 저장 완료: {self.vector_store_path}")

        except Exception as e:
            print(f"인덱스 생성 중 오류 발생: {e}")

    def load_index(self):
        store_dir = self.vector_store_path

        if not os.path.exists(store_dir):
            raise FileNotFoundError(f"VectorStore 경로가 존재하지 않습니다: {store_dir}")

        print(f"VectorStore 불러오는 중: {store_dir}")
        self.vector_store = load_vector_store(store_dir)

    def query(self, question: str, filters: Optional[Dict] = None):
        """
        전체 RAG Query 처리
        """
        if self.vector_store is None:
            self.load_index()
            
        print("\n검색 실행 중...")
        
        retrieved_chunks = retrieve(
            question,
            self.vector_store,
            self.top_k,
            filters
        )
        print(f"    ➜ 검색된 청크: {len(retrieved_chunks)}개")

        
        print("LLM 답변 생성 중...")
        
        llm_result = self.generator_test.generate(
            query=question,
            retrieved_chunks=retrieved_chunks,
            use_history=True
        )
        
        return {
            "answer": llm_result["answer"],
            "sources": llm_result["sources"],
            "retrieved_chunks": retrieved_chunks,
            "metadata": llm_result["metadata"]
        }

def get_filter_from_user():
    print("검색에 사용할 메타데이터 값을 입력하세요(엔터시 건너뜀)")
    filters = {}
    filters['사업명'] = input("사업명: ") or None
    filters['사업 금액'] = parse_numeric_filter(input("사업 금액: ")) or None
    filters['발주 기관'] = input("발주 기관: ") or None
    filters['파일명'] = input("파일명: ") or None
    filters['사업 요약'] = input("사업 요약: ") or None

def parse_numeric_filter(user_input: str):
    """
    사용자 문자열 (예: '>20000') -> dict 형태 (예: {'$gt': 20000})
    """
    user_input = user_input.strip().replace(' ', '')
    op_map = {
        '>=': '$gte',
        '<=': '$lte',
        '>': '$gt',
        '<': '$lt',
        '=': '$eq'
    }
    # 우선 >=, <= 같은 긴 연산자부터 체크
    for symbol, mongo_op in op_map.items():
        if user_input.startswith(symbol):
            num_part = user_input[len(symbol):]
            try:
                value = int(num_part)
                return {mongo_op: value}
            except ValueError:
                return None  # 숫자로 변환 실패 시
    # 숫자만 입력한 경우 eq로 처리
    if user_input.isdigit():
        return {'$eq': int(user_input)}
    return None

# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    try:
        pipe = RAGPipeline(config_path="../config/config_test.yaml")
        
        user_filters = get_filter_from_user() #메타데이터 필터 UI 넣는거 아닌이상 꼬리 무는 질문에서 불편
        while True:
            question = input("질문(q 입력시 종료): ")
            if question.lower() == 'q':
                break
                
            response = pipe.query(question, user_filters)
            
            print("\n====== 결과 ======")
            print("🤖 AI:", response["answer"])
            print("참고문서:", response["sources"])
    
    except FileNotFoundError as e:
        print(f"\n[테스트 오류] {e}. 파일을 확인하거나 Mock 데이터를 사용하여 테스트하십시오.")
    except Exception as e:
        print(f"\n[테스트 오류] 예상치 못한 에러 발생: {e}")