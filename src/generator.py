# 1. 설정 및 초기화
## 라이브러리 설치 및 임포트
!unzip /content/drive/MyDrive/pdf_files.zip
!pip install langchain pypdf tiktoken openai

import os
import json
import re
import hashlib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from functools import lru_cache
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from openai.types.chat import ChatCompletion
from dotenv import load_dotenv
import tiktoken

## 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. 데이터 준비
## PDF 로딩 함수
def load_pdf(file_path: str) -> str:
    """PDF 파일에서 텍스트 전체를 추출합니다."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or "" # 텍스트가 없는 페이지는 스킵
        return text
    except Exception as e:
        print(f"PDF 로딩 오류 ({os.path.basename(file_path)}): {e}")
        return ""

## 텍스트 청킹
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # 1000 글자 단위로 자르기
    chunk_overlap=100,  # 100 글자씩 겹치게 자르기 (문맥 유지)
    length_function=len,
    is_separator_regex=False,
)

## 파일 처리 실행
file_directory = "/content/pdf_files/" # PDF 파일이 있는 폴더 경로

all_chunks = [] # 모든 청크(Chunk)를 저장할 리스트
processed_file_count = 0

# 지정된 디렉토리가 존재하는지 확인
if not os.path.exists(file_directory):
    print(f"폴더를 찾을 수 없습니다: {file_directory}")
else:
    # 폴더 내의 모든 파일을 순회
    for file_name in os.listdir(file_directory):

        # .pdf 파일만 대상으로 함
        if file_name.endswith(".pdf"):
            file_path = os.path.join(file_directory, file_name)

            # 1. Loading: PDF에서 텍스트 추출
            print(f"[PDF] 로딩 중: {file_name}")
            raw_text = load_pdf(file_path)

            if not raw_text:
                print(f"텍스트 추출 실패: {file_name}")
                continue

            # 2. Chunking: 긴 텍스트를 청크로 분할
            chunks = text_splitter.split_text(raw_text)

            print(f"  -> {file_name}에서 {len(chunks)}개의 청크 생성됨.")

            # 3. 메타데이터 추가
            for i, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source_id": file_name, # Generator가 출처 표시에 사용
                        "chunk_index": i
                    }
                })

            processed_file_count += 1

print(f"총 {processed_file_count}개 PDF에서 {len(all_chunks)}개 청크 생성")

# 3. Generator 유틸리티
## 토큰 제한 관리
def truncate_chunks_by_tokens(
    chunks: List[Dict],
    max_tokens: int = 3000,
    model: str = "gpt-4o-mini"
) -> List[Dict]:
    """컨텍스트가 너무 길어지지 않도록 토큰 수를 제한"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # 모델이 없으면 기본 인코딩 사용
        encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    selected_chunks = []

    for chunk in chunks:
        chunk_tokens = len(encoding.encode(chunk['text']))
        if total_tokens + chunk_tokens > max_tokens:
            break
        selected_chunks.append(chunk)
        total_tokens += chunk_tokens

    logger.info(f"토큰 제한: {len(chunks)}개 청크 중 {len(selected_chunks)}개 선택 (총 {total_tokens} 토큰)")
    return selected_chunks

## 답변 품질 검증
def validate_answer(answer: str, chunks: List[Dict]) -> Dict[str, Any]:
    """답변이 실제 문서에 기반했는지 검증"""
    validation = {
        "has_content": len(answer.strip()) > 10,
        "is_refusal": any(phrase in answer for phrase in [
            "찾을 수 없습니다",
            "정보가 없습니다",
            "제공된 문서에서"
        ]),
        "answer_length": len(answer)
    }
    return validation

## 후처리 (Post-processing)
def post_process_answer(answer: str) -> str:
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    answer = answer.strip()
    return answer

## 캐싱 유틸리티 (성능 최적화)
def get_query_hash(query: str, chunk_ids: List[str]) -> str:
    content = query + "".join(sorted(chunk_ids))
    return hashlib.md5(content.encode()).hexdigest()

answer_cache = {}

# 4. 프롬프트
## 프롬프트 템플릿 정의
SYSTEM_PROMPTS = {
    "default": """당신은 '입찰메이트'의 B2G 입찰 전문 컨설턴트입니다.
당신의 임무는 제공된 '참고 문서 조각'만을 근거로 사용자의 질문에 대해 전문적이고 간결하게 답변하는 것입니다.

지침:
- 답변은 반드시 제공된 '참고 문서 조각'의 내용에 기반해야 합니다.
- 문서에 내용이 없는 경우, '제공된 문서에서는 해당 정보를 찾을 수 없습니다.'라고 명확히 답변하세요.
- 답변 시 어떤 문서 조각을 참고했는지 본문에 명시할 필요는 없습니다.
- 숫자/날짜/금액은 정확히 인용해야 합니다.
- 애매한 표현 지양, 명확한 사실을 전달해야 합니다.
- '입찰메이트'의 전문 컨설턴트로서 친절하고 명확한 톤을 유지하세요.""",

    "technical": """당신은 기술 제안서 작성 전문가입니다.
제공된 문서의 기술 요구사항을 분석하여 구체적이고 기술적인 답변을 제공합니다.
시스템 아키텍처, 성능 요구사항, 기술 스택 등을 중심으로 답변하세요.""",

    "summarize": """제공된 입찰 문서를 요약하는 전문가입니다.
핵심 내용을 간결하게 정리하되, 중요한 세부사항은 누락하지 마세요."""
}

## 프롬프트 메시지 생성
def create_prompt_messages(
    query: str,
    retrieved_chunks: List[Dict],
    prompt_type: str = "default",
    chat_history: Optional[List[Dict]] = None
) -> List[Dict[str, str]]:
    """RAG 파이프라인을 위한 OpenAI messages 리스트 생성"""

    # 시스템 프롬프트
    system_prompt = SYSTEM_PROMPTS.get(prompt_type, SYSTEM_PROMPTS["default"])
    messages = [{"role": "system", "content": system_prompt}]

    # 컨텍스트 포맷팅 (Retrieved Chunks)
    context_str = "--- 참고 문서 조각 ---\n"
    for i, chunk in enumerate(retrieved_chunks):
        source_id = chunk.get('metadata', {}).get('source_id', f'문서조각_{i+1}')
        context_str += f"[문서 ID: {source_id}]\n"
        context_str += f"{chunk.get('text', '')}\n\n"
    context_str += "----------------------\n"

    # 대화 내역 추가
    if chat_history:
        messages.extend(chat_history)

    # 사용자 질문 + 컨텍스트
    user_prompt = (
        f"{context_str}\n"
        f"위의 '참고 문서 조각'을 바탕으로 다음 질문에 답변해 주세요:\n"
        f"질문: {query}"
    )
    messages.append({"role": "user", "content": user_prompt})

    return messages

## LLM 답변 생성 (기본)
def generate_answer(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    config: Dict[str, Any]
) -> ChatCompletion:
    """OpenAI API를 호출하여 LLM의 답변을 받습니다."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            **config
        )
        return completion
    except Exception as e:
        logger.error(f"OpenAI API 오류 발생: {e}")
        raise

## LLM 답변 생성 (로깅 포함)
def generate_answer_with_logging(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    config: Dict[str, Any]
) -> ChatCompletion:
    """로깅이 포함된 답변 생성"""
    start_time = datetime.now()

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            **config
        )

        latency = (datetime.now() - start_time).total_seconds() * 1000
        logger.info({
            "timestamp": start_time.isoformat(),
            "model": model,
            "tokens_used": completion.usage.total_tokens if completion.usage else 0,
            "latency_ms": latency,
            "status": "success"
        })

        return completion

    except Exception as e:
        logger.error({
            "timestamp": start_time.isoformat(),
            "error": str(e),
            "status": "failed"
        })
        raise

## 응답 포맷팅
def format_response(
    completion: ChatCompletion,
    retrieved_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """ChatCompletion 객체를 최종 JSON 형식으로 포맷팅"""

    # 1. 답변 추출 및 후처리
    answer = completion.choices[0].message.content or "답변을 생성하지 못했습니다."
    answer = post_process_answer(answer)

    # 2. 소스 추출
    source_ids = []
    for chunk in retrieved_chunks:
        source_id = chunk.get('metadata', {}).get('source_id', 'Unknown Source')
        if source_id not in source_ids:
            source_ids.append(source_id)
    source_ids.sort()

    # 3. 메타데이터
    response_metadata = {
        "model": completion.model,
        "tokens_used": completion.usage.total_tokens if completion.usage else 0,
        "finish_reason": completion.choices[0].finish_reason
    }

    # 4. 답변 검증
    validation = validate_answer(answer, retrieved_chunks)
    response_metadata["validation"] = validation

    # 5. 최종 출력
    final_output = {
        "answer": answer,
        "sources": source_ids,
        "metadata": response_metadata
    }

    return final_output

## 캐싱 적용 답변 생성 (통합)
def generate_with_cache(
    client: OpenAI,
    query: str,
    chunks: List[Dict],
    model: str,
    config: Dict[str, Any],
    prompt_type: str = "default",
    chat_history: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """캐싱을 적용한 답변 생성"""

    # 캐시 키 생성
    chunk_ids = [c['metadata']['source_id'] for c in chunks]
    cache_key = get_query_hash(query, chunk_ids)

    # 캐시 확인
    if cache_key in answer_cache:
        logger.info(f"캐시 히트: {cache_key[:8]}...")
        return answer_cache[cache_key]

    # 실제 생성
    logger.info(f"캐시 미스: 새로운 답변 생성 중...")
    messages = create_prompt_messages(query, chunks, prompt_type, chat_history)
    completion = generate_answer_with_logging(client, messages, model, config)
    final_response = format_response(completion, chunks)

    # 캐시 저장
    answer_cache[cache_key] = final_response

    return final_response

# 5. 대화 관리
## 대화 관리자
class ConversationManager:
    def __init__(self, max_history: int = 5):
        self.history = []
        self.max_history = max_history

    def add_turn(self, user_msg: str, assistant_msg: str):
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": assistant_msg})

        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

    def get_history(self):
        return self.history

    def clear_history(self):
        self.history = []

## 실행
# .env 파일에서 환경변수 로드
load_dotenv()

# 환경변수에서 API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")

# API 키 존재 여부 확인
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)

# 입력 데이터
user_query = "국민연금공단 이러닝시스템 요구사항은?"

mock_retrieved_chunks = [
    {
        "text": "국민연금공단 이러닝시스템은... 콘텐츠 개발 관리 요구사항으로 SCORM v1.2 표준을 준수해야 함...",
        "metadata": {"source_id": "RFP_001_chunk_3", "file_name": "국민연금공단_RFP.hwp"}
    },
    {
        "text": "또한, 시스템은 1일 최대 5,000명의 동시 접속자를 처리할 수 있어야 하며, 응답 시간은 3초 이내여야 함.",
        "metadata": {"source_id": "RFP_001_chunk_5", "file_name": "국민연금공단_RFP.hwp"}
    }
]

# Generation 설정
model_to_use = "gpt-5-nano" # gpt-5 > gpt-5-mini > gpt-5-nano
generation_config = {
    "temperature": 1,
    "max_completion_tokens": 1024,
    "top_p": 1.0,
}

# 방법 1: 기본 실행
print("방법 1: 기본 실행")
messages_list = create_prompt_messages(
    query=user_query,
    retrieved_chunks=mock_retrieved_chunks
)

completion_object = generate_answer_with_logging(
    client=client,
    messages=messages_list,
    model=model_to_use,
    config=generation_config
)

final_response = format_response(
    completion=completion_object,
    retrieved_chunks=mock_retrieved_chunks
)

print("최종 출력 (JSON)")
print(json.dumps(final_response, indent=4, ensure_ascii=False))

# 방법 2: 캐싱 적용 실행
print("방법 2: 캐싱 적용 실행")

# 토큰 제한 적용
truncated_chunks = truncate_chunks_by_tokens(mock_retrieved_chunks, max_tokens=2000)

# 캐싱 적용 생성
cached_response = generate_with_cache(
    client=client,
    query=user_query,
    chunks=truncated_chunks,
    model=model_to_use,
    config=generation_config
)

print("캐싱 적용 결과")
print(json.dumps(cached_response, indent=4, ensure_ascii=False))

# 같은 쿼리 재실행 (캐시 히트 확인)
print("같은 쿼리 재실행 (캐시 확인)")
cached_response_2 = generate_with_cache(
    client=client,
    query=user_query,
    chunks=truncated_chunks,
    model=model_to_use,
    config=generation_config
)
