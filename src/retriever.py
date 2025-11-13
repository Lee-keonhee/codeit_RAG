from typing import List, Dict, Optional
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from vector_store import load_vector_store

# 환경 변수 로드
#load_dotenv()

# ============================
# 기본 검색 함수
# ============================

def retrieve( query: str, vector_store_path: str, 
              top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
    """
    쿼리 기반 관련 문서 검색
    입력:
        query: 사용자 입력 질의 (예: "국민연금공단 이러닝시스템 요구사항은?")
        vector_store_path: 저장된 VectorStore 디렉터리 경로
        top_k: 상위 검색 개수
        filters: 메타데이터 필터 (예: {"발주기관": "국민연금공단"})
    출력:
        List[Dict]: 유사도 높은 청크 리스트
    """

    # 벡터스토어 로드
    vectorstore = load_vector_store(vector_store_path)

    # 메타데이터 필터 적용
    # LangChain FAISS는 where 필터 지원 안 함 → 직접 필터링 필요
    docs = vectorstore.similarity_search_with_score(query, k=top_k * 2)
    
    results = []
    for i, (doc, score) in enumerate(docs):
        meta = getattr(doc, "metadata", {})
        
        # 필터 조건 확인
        if filters:
            match = all(meta.get(k) == v for k, v in filters.items())
            
            if not match:
                continue

        doc_id = meta['doc_id']
        chunk_index = meta['chunk_index']
        
        results.append({
            "chunk_id": f"{doc_id}_chunk_{chunk_index}", #doc_1_chunk_12 형태
            "text": doc.page_content,
            "score": float(score),
            "metadata": meta
        })

    # Reranking 
    results = rerank_results(query, results)

    # top_k 반환
    return results[:top_k]


# ============================
# Re-ranking 함수
# ============================

def rerank_results(query: str, results: List[Dict]) -> List[Dict]:
    """
    CrossEncoder를 이용한 문맥 기반 재정렬
    """
    if not results:
        return results

    try:
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, r["text"]) for r in results]
        scores = model.predict(pairs)

        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)

        results = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return results

    except Exception as e:
        print(f"Re-ranking 중 오류 발생: {e}")
        return results




# ============================
# FAISS 필터링 헬퍼 함수
# ============================

def _apply_faiss_filter(meta: Dict, filters: Dict) -> bool:
    """
    FAISS의 Post-query Filtering에서 메타데이터를 숫자 비교 및 정확히 일치시키는 함수.
    """
    for key, condition in filters.items():
        doc_value = meta.get(key)
        
        # 1. 값이 아예 없는 경우 (필터링 실패)
        if doc_value is None:
            return False

        # 2. 정확한 값 일치 (문자열 또는 숫자)
        if isinstance(condition, (str, int, float)):
            if str(doc_value) != str(condition):
                return False
        
        # 3. 숫자 범위 비교 (Dict 형태, 예: {"$gt": 100000000})
        elif isinstance(condition, dict):
            try:
                # 메타데이터 값을 float으로 변환 시도
                doc_num = float(doc_value)
            except (ValueError, TypeError):
                # 숫자로 변환 불가능하면 해당 필터 조건은 불일치로 처리
                return False

            for op, filter_num in condition.items():
                if op == "$gt" and not (doc_num > filter_num):
                    return False
                elif op == "$lt" and not (doc_num < filter_num):
                    return False
                elif op == "$gte" and not (doc_num >= filter_num):
                    return False
                elif op == "$lte" and not (doc_num <= filter_num):
                    return False
                elif op == "$eq" and not (doc_num == filter_num):
                    return False
                # 다른 연산자는 무시 (추후 확장 가능)
        
        # 4. 지원하지 않는 필터 조건 형식
        else:
            return False

    return True # 모든 필터 조건을 통과함


# ============================
# 실행 예시
# ============================
"""
if __name__ == "__main__":
    query_input = {
        "query": "국민연금공단 이러닝시스템 요구사항은?",
        "filters": {"발주 기관": "국민연금공단"},
        "top_k": 5
    }

    vector_store_path = "../vector_store"  # vector_store.py에서 저장한 경로

    results = retrieve(
        query=query_input["query"],
        vector_store_path=vector_store_path,
        top_k=query_input["top_k"],
        filters=query_input["filters"]
    )

    for r in results:
        print("-" * 80)
        print(f"[Score] {r['rerank_score']:.4f}")
        print(f"[Text]\n{r['text'][:300]}...")
"""