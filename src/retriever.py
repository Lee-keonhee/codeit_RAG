from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS
from vector_store import load_vector_store

# ============================
# 기본 검색 함수
# ============================

def retrieve( query: str, vectorstore: FAISS,
              top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
    """
    쿼리 기반 관련 문서 검색, 
    입력:
        query: 사용자 입력 질의 (예: "국민연금공단 이러닝시스템 요구사항은?")
        vectorstore: 이미 로드된 VectorStore 객체를 사용
        top_k: 상위 검색 개수
        filters: 메타데이터 필터 (예: {"발주기관": "국민연금공단"})
    출력:
        List[Dict]: 유사도 높은 청크 리스트
    """

    # 이전 로드 로직 제거 (RAGPipeline에서 이미 로드하여 vectorstore에 전달했으므로)
    # 유사도 검색
    docs = vectorstore.similarity_search_with_score(query, k=top_k * 5)
    
    results = [] 
    #Post-query 필터링
    for i, (doc, score) in enumerate(docs):
        meta = getattr(doc, "metadata", {})
        
        # 필터 조건을 만족하는지 확인
        match = True
        if filters:
            # 헬퍼 함수를 사용하여 숫자 비교 및 정확한 값 필터링 수행
            match = _apply_faiss_filter(meta, filters)
            if not match:
                continue # 조건 불일치 시 건너뛰기

        # 필터 조건을 통과한 경우에만 결과에 추가
        if match:
            doc_id = meta['doc_id']
            chunk_index = meta['chunk_index']
        
            results.append({
                "chunk_id": f"{doc_id}_chunk_{chunk_index}", #doc_1_chunk_12 형태
                "text": doc.page_content,
                "score": float(score), #유사도 점수
                "metadata": meta
            })

        # 필터링 후 top_k 개수를 확보했는지 확인 후 중단
        if len(results) >= top_k:
            break
            
    #print(f"DEBUG: Re-ranking 시작. 대상 문서 개수: {len(results)}개.")

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
            r["rerank_score"] = float(s) # 기존 유사도 스코어와는 별개로 재정렬 스코어 추가

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