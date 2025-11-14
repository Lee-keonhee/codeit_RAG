import os
import json
import re
import hashlib
import logging
import time
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. 모델 및 생성 설정 최적화
# ============================================================================

class ModelConfig:
    """비용과 성능을 고려한 모델 설정"""
    
    # 모델별 특성
    MODELS = {
        "gpt-5": {
            "cost_per_1k_input": 0.0025,
            "cost_per_1k_output": 0.010,
            "context_window": 128000,
            "best_for": "복잡한 입찰 분석"
        },
        "gpt-5-mini": {
            "cost_per_1k_input": 0.00015,
            "cost_per_1k_output": 0.0006,
            "context_window": 128000,
            "best_for": "일반 질의응답"
        },
        "gpt-5-nano": {
            "cost_per_1k_input": 0.0005,
            "cost_per_1k_output": 0.0015,
            "context_window": 16385,
            "best_for": "간단한 정보 추출"
        }
    }
    
    @classmethod
    def get_optimal_model(cls, query_complexity: str = "medium") -> str:
        """쿼리 복잡도에 따른 최적 모델 선택"""
        if query_complexity == "high":
            return "gpt-5"
        elif query_complexity == "low":
            return "gpt-5-nano"
        else:
            return "gpt-5-mini"  # 기본값: 비용 대비 성능 최적
    
    @classmethod
    def get_generation_config(cls, response_type: str = "detailed") -> Dict[str, Any]:
        """응답 타입별 최적화된 생성 설정"""
        configs = {
            "detailed": {
                "max_completion_tokens": 4000
            },
            "concise": {
                "max_completion_tokens": 500
            },
            "creative": {
                "max_completion_tokens": 1000
            }
        }
        return configs.get(response_type, configs["detailed"])


# ============================================================================
# 2. 프롬프트 최적화
# ============================================================================

class OptimizedPrompts:
    """프롬프트 템플릿 최적화"""
    
    SYSTEM_PROMPT = """당신은 '입찰메이트'의 B2G 입찰 전문 AI 어시스턴트입니다.

**핵심 원칙:**
1. 제공된 [참고 문서]의 내용만 사용
2. 문서에 없는 내용은 "문서에서 확인할 수 없습니다" 명시
3. 숫자, 날짜, 금액은 정확히 인용
4. 간결하고 구조화된 답변

**답변 형식:**
- 핵심 내용을 bullet point로 정리
- 중요 키워드는 **볼드** 처리
- 애매한 표현 지양

**톤:** 전문적이지만 이해하기 쉽게"""

    @staticmethod
    def create_user_prompt(query: str, chunks: List[Dict], chat_history: Optional[List[Dict]] = None) -> str:
        """토큰 효율적인 사용자 프롬프트 생성"""
        
        # 컨텍스트 구성 (최소한의 메타데이터만 포함)
        context_parts = []
        seen_sources = set()
        
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get('metadata', {}) if isinstance(chunk, dict) else {}
            chunk_id = meta.get('chunk_id')
            text = chunk.get('text', '').strip()
            
            # 중복 소스 표시 최소화
            if chunk_id not in seen_sources:
                context_parts.append(f"[{chunk_id}]")
                seen_sources.add(chunk_id)
            
            context_parts.append(text)
            context_parts.append("")  # 청크 간 구분
        
        context_str = "\n".join(context_parts)
        
        # 대화 히스토리 요약 (있는 경우)
        history_str = ""
        if chat_history and len(chat_history) > 0:
            # 최근 2턴만 포함하여 토큰 절약
            recent_history = chat_history[-4:]  # user + assistant 2쌍
            history_lines = []
            for msg in recent_history:
                role = "사용자" if msg["role"] == "user" else "AI"
                content = msg["content"][:100]  # 100자로 제한
                history_lines.append(f"{role}: {content}")
            history_str = "\n이전 대화:\n" + "\n".join(history_lines) + "\n\n"
        
        # 최종 프롬프트 (토큰 절약형)
        return f"""{history_str}[참고 문서]
{context_str}

질문: {query}

위 문서를 바탕으로 답변해주세요."""

    @staticmethod
    def estimate_tokens(text: str, model: str = "gpt-5-mini") -> int:
        """텍스트의 토큰 수 추정"""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


# ============================================================================
# 3. 토큰 사용량 최적화
# ============================================================================

class TokenOptimizer:
    """토큰 사용량 최적화 유틸리티"""
    
    @staticmethod
    def truncate_chunks_smart(
        chunks: List[Dict],
        query: str,
        max_tokens: int = 4000,
        model: str = "gpt-5-mini"
    ) -> List[Dict]:
        """쿼리 관련성과 토큰 제한을 고려한 청크 선택"""
        
        # 간단한 키워드 기반 관련성 스코어링
        query_keywords = set(query.lower().split())
        
        scored_chunks = []
        for chunk in chunks:
            text = chunk.get('text', '').lower()
            # 키워드 매칭 스코어
            matches = sum(1 for kw in query_keywords if kw in text)
            
            # 추가: rerank_score가 있으면 가중치 반영
            rerank = float(chunk.get('rerank_score') or chunk.get('metadata', {}).get('rerank_score') or 0)
            score = matches + rerank
            scored_chunks.append((score, chunk))
        
        # 관련성 높은 순으로 정렬
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # 토큰 제한까지 선택
        selected = []
        total_tokens = 0
        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        for score, chunk in scored_chunks:
            chunk_text = chunk.get('text', '')
            chunk_tokens = len(encoding.encode(chunk_text))
            if total_tokens + chunk_tokens <= max_tokens:
                selected.append(chunk)
                total_tokens += chunk_tokens
            else:
                # 토큰 초과시 청크를 줄여서라도 일부 포함하는 로직은 현재 비활성
                continue
        
        logger.info(f"토큰 최적화: {len(chunks)}개 중 {len(selected)}개 선택 ({total_tokens} tokens)")
        return selected


# ============================================================================
# 4. 향상된 대화 관리자
# ============================================================================

class EnhancedConversationManager:
    """대화 맥락을 유지하는 향상된 대화 관리자"""
    
    def __init__(self, max_history: int = 3, max_tokens_per_turn: int = 300):
        self.history = []
        self.max_history = max_history
        self.max_tokens_per_turn = max_tokens_per_turn
        self.conversation_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        
    def add_turn(self, user_msg: str, assistant_msg: str, sources: List[str] = None):
        """대화 턴 추가 (토큰 제한 적용)"""
        
        # 토큰 제한을 위해 메시지 요약
        user_msg_truncated = self._truncate_message(user_msg)
        assistant_msg_truncated = self._truncate_message(assistant_msg)
        
        self.history.append({
            "role": "user",
            "content": user_msg_truncated,
            "timestamp": datetime.now().isoformat(),
            "full_content": user_msg  # 필요시 참조용
        })
        
        self.history.append({
            "role": "assistant",
            "content": assistant_msg_truncated,
            "sources": sources or [],
            "timestamp": datetime.now().isoformat()
        })
        
        # 히스토리 길이 제한
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]
            
        logger.info(f"대화 턴 추가 (총 {len(self.history)//2}턴)")
    
    def _truncate_message(self, message: str) -> str:
        """메시지를 토큰 제한에 맞게 축약"""
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(message)
        
        if len(tokens) > self.max_tokens_per_turn:
            truncated_tokens = tokens[:self.max_tokens_per_turn]
            return encoding.decode(truncated_tokens) + "..."
        return message
    
    def get_history_for_prompt(self) -> List[Dict[str, str]]:
        """프롬프트에 포함할 히스토리 반환 (간결화)"""
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.history
        ]
    
    def get_context_summary(self) -> str:
        """대화 맥락 요약"""
        if not self.history:
            return "새로운 대화"
        
        user_turns = [msg for msg in self.history if msg["role"] == "user"]
        return f"{len(user_turns)}개 질문 진행 중"
    
    def clear_history(self):
        """대화 히스토리 초기화"""
        self.history = []
        logger.info(f"대화 {self.conversation_id} 히스토리 초기화")

# ============================================================================
# 5. 메인 Generator 클래스
# ============================================================================

class RAGGenerator:
    """최적화된 RAG Generator"""
    
    def __init__(
        self,
        model: Optional[str] = None,
        response_type: str = "detailed"
    ):
        # API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model or ModelConfig.get_optimal_model("medium")
        self.config = ModelConfig.get_generation_config(response_type)
        self.conversation_manager = EnhancedConversationManager()
        self.prompts = OptimizedPrompts()
        self.token_optimizer = TokenOptimizer()
        
        # 메트릭 추적
        self.metrics = {
            "total_queries": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "avg_latency_ms": 0.0
        }
        
        logger.info(f"Generator 초기화: model={self.model}, config={self.config}")
    
    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        use_history: bool = True
    ) -> Dict[str, Any]:
        """최적화된 답변 생성"""
        
        start_time = time.time()
        self.metrics["total_queries"] += 1
        
        try:
            # 1. 토큰 최적화: 청크 선택
            optimized_chunks = self.token_optimizer.truncate_chunks_smart(
                chunks=retrieved_chunks,
                query=query,
                max_tokens=4000,
                model=self.model
            )
            
            # 2. 프롬프트 생성
            chat_history = self.conversation_manager.get_history_for_prompt() if use_history else None
            
            messages = [
                {"role": "system", "content": self.prompts.SYSTEM_PROMPT},
            ]
            
            if chat_history:
                messages.extend(chat_history)
            
            user_prompt = self.prompts.create_user_prompt(
                query=query,
                chunks=optimized_chunks,
                chat_history=chat_history
            )
            messages.append({"role": "user", "content": user_prompt})
            
            # 토큰 수 추정
            estimated_tokens = sum(
                self.prompts.estimate_tokens(msg["content"], self.model)
                for msg in messages
            )
            logger.info(f"입력 토큰 추정: {estimated_tokens}")
            
            # 3. API 호출
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self.config
            )
            
            # 4. 결과 처리
            answer = completion.choices[0].message.content or "답변을 생성하지 못했습니다."
            answer = self._post_process(answer)
            
            # 5. 메트릭 업데이트
            tokens_used = completion.usage.total_tokens if completion.usage else 0
            latency_ms = (time.time() - start_time) * 1000
            cost = self._calculate_cost(completion.usage, self.model)
            
            self.metrics["total_tokens"] += tokens_used
            self.metrics["total_cost"] += cost
            self.metrics["avg_latency_ms"] = (
                (self.metrics["avg_latency_ms"] * (self.metrics["total_queries"] - 1) + latency_ms)
                / self.metrics["total_queries"]
            )
            
            # 6. 소스 추출
            sources = list(set(
                chunk.get('metadata', {}).get('파일명', 'Unknown')
                for chunk in optimized_chunks
            ))
            
            # 7. 대화 히스토리 업데이트
            if use_history:
                self.conversation_manager.add_turn(query, answer, sources)
            
            # 8. 최종 응답
            response = {
                "answer": answer,
                "sources": sorted(sources)p,
                "metadata": {
                    "model": self.model,
                    "tokens_used": tokens_used,
                    "latency_ms": round(latency_ms, 2),
                    "cost_usd": round(cost, 6),
                    "chunks_used": len(optimized_chunks),
                    "conversation_turns": len(self.conversation_manager.history) // 2
                }
            }
            
            logger.info(f"답변 생성 완료: {tokens_used} tokens, {latency_ms:.0f}ms, ${cost:.6f}")
            return response
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            return {
                "answer": "죄송합니다. 답변 생성 중 오류가 발생했습니다.",
                "sources": [],
                "metadata": {"error": str(e)}
            }
    
    def _post_process(self, answer: str) -> str:
        """답변 후처리"""
        # 과도한 줄바꿈 제거
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        # 앞뒤 공백 제거
        answer = answer.strip()
        return answer
    
    def _calculate_cost(self, usage: Any, model: str) -> float:
        """API 사용 비용 계산"""
        if not usage:
            return 0.0
        
        model_info = ModelConfig.MODELS.get(model, ModelConfig.MODELS["gpt-5-mini"])
        input_cost = (usage.prompt_tokens / 1000) * model_info["cost_per_1k_input"]
        output_cost = (usage.completion_tokens / 1000) * model_info["cost_per_1k_output"]
        return input_cost + output_cost
    
    def get_metrics(self) -> Dict[str, Any]:
        """현재까지의 사용 메트릭 반환"""
        return self.metrics.copy()
    
    def reset_conversation(self):
        """대화 초기화"""
        self.conversation_manager.clear_history()