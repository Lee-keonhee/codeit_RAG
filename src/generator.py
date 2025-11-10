'''
LLM을 통한 답변 생성 (OpenAI API)
입력 : {
    "query": "국민연금공단 이러닝시스템 요구사항은?",
    "retrieved_chunks": [
        {"text": "...", "metadata": {...}},
        ...
    ],
    "chat_history": [...]  # optional
}
출력 :{
    "answer": "국민연금공단의 이러닝시스템 요구사항은 다음과 같습니다...",
    "sources": ["RFP_001_chunk_3", "RFP_001_chunk_5"],
    "metadata": {
        "model": "gpt-4o-mini",
        "tokens_used": 450
    }
}

주요 함수:
def create_prompt(query: str, context: List[str], chat_history: List) -> str
def generate_answer(prompt: str, model: str, config: Dict) -> str
def format_response(answer: str, sources: List) -> Dict
'''