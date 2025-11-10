# RAG 기반 RFP 문서 질의응답 시스템

## 📋 프로젝트 개요

공공입찰 RFP(제안요청서) 문서에서 핵심 정보를 빠르게 추출하고 질의응답할 수 있는 RAG(Retrieval-Augmented Generation) 시스템입니다.

### 🎯 프로젝트 목표

- **문제 정의**: 하루 수백 건씩 올라오는 수십 페이지 분량의 RFP 문서에서 필요한 정보를 빠르게 찾기
- **해결 방안**: RAG 시스템을 통한 자동 문서 검색 및 질의응답
- **기대 효과**: 컨설턴트의 문서 검토 시간 단축, 핵심 정보 빠른 파악

### 👥 팀 구성 및 역할

| 역할 | 담당자 | 주요 업무 |
|------|--------|-----------|
| Project Manager | OOO | 프로젝트 매니징, 성능 평가 |
| 데이터 처리 | OOO | 문서 로딩, 청킹 전략 설계 |
| Retrieval | OOO | 임베딩 생성, Vector DB 구축 |
| Generation | OOO | LLM 답변 생성, 프롬프트 엔지니어링 |

---

## 🛠️ 기술 스택

### Core
- **Language**: Python 3.10+
- **Framework**: Custom RAG Pipeline

### Document Processing
- **PDF**: pypdf
- **HWP**: olefile
- **Data**: pandas

### AI/ML
- **LLM**: OpenAI GPT-4o-mini
- **Embedding**: OpenAI text-embedding-3-small
- **Vector DB**: FAISS (CPU)

### Development
- **Environment**: Python venv
- **Notebook**: Jupyter
- **Testing**: pytest

---

## 📁 프로젝트 구조

```
rag-project/
├── config/
│   └── config.yaml              # 설정 파일
├── data/
│   ├── raw/                     # 원본 RFP 문서 (hwp, pdf)
│   └── processed/               # 전처리된 데이터
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # 문서 로딩
│   ├── preprocessor.py          # 전처리 및 청킹
│   ├── embedder.py              # 임베딩 생성
│   ├── vector_store.py          # Vector DB 관리
│   ├── retriever.py             # 검색 및 Retrieval
│   ├── generator.py             # LLM 답변 생성
│   └── pipeline.py              # 전체 RAG 파이프라인
├── utils/
│   ├── __init__.py
│   └── helpers.py               # 유틸리티 함수
├── notebooks/
│   └── experiments.ipynb        # 실험 및 테스트
├── tests/
│   └── test_pipeline.py         # 테스트 코드
├── requirements.txt             # 패키지 의존성
├── .env.example                 # 환경변수 예시
├── .gitignore
└── README.md
```

---

## 🚀 시작하기

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd rag-project

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
# .env.example을 복사하여 .env 파일 생성
cp .env.example .env

# .env 파일 편집
# OPENAI_API_KEY=your-api-key-here
```

### 3. 데이터 준비

```bash
# RFP 문서를 data/raw/ 폴더에 배치
# - PDF 파일
# - HWP 파일
# - data_list.csv (메타데이터)
```

---

## 💻 사용 방법

### 인덱스 구축

```python
from src.pipeline import RAGPipeline

# 파이프라인 초기화
pipeline = RAGPipeline(config_path='config/config.yaml')

# 문서 인덱싱
pipeline.build_index(
    doc_folder='data/raw',
    metadata_path='data/raw/data_list.csv'
)

# 인덱스 저장
pipeline.save_index('data/processed/vector_store')
```

### 질의응답

```python
# 인덱스 로드
pipeline = RAGPipeline.load('data/processed/vector_store')

# 질문하기
response = pipeline.query(
    question="국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 정리해 줘",
    filters={'발주기관': '국민연금공단'}  # 선택적 필터
)

print(response['answer'])
print(response['sources'])
```

### Jupyter Notebook으로 실험

```bash
jupyter notebook notebooks/experiments.ipynb
```

---

## 📊 성능 평가

### 평가 지표

- **검색 정확도**: 관련 문서 검색 정확도
- **답변 품질**: LLM 답변의 정확성, 완성도
- **응답 시간**: 질문부터 답변까지 소요 시간
- **메타데이터 필터링**: 특정 기관/사업 필터링 정확도

### 테스트 쿼리 예시

```python
test_queries = [
    "국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 정리해 줘",
    "기초과학연구원 극저온시스템 사업 요구에서 AI 기반 예측에 대한 요구사항이 있나?",
    "한국 원자력 연구원에서 선량 평가 시스템 고도화 사업을 발주했는데, 이 사업이 왜 추진되는지 목적을 알려 줘",
]
```

---

## 🔧 주요 기능

### 1. 문서 처리
- PDF, HWP 파일 자동 로딩
- 메타데이터(발주기관, 사업명, 예산 등) 연동
- 효율적인 청킹 전략

### 2. 검색 (Retrieval)
- OpenAI Embedding을 통한 의미 기반 검색
- FAISS를 활용한 빠른 벡터 검색
- 메타데이터 필터링 지원

### 3. 답변 생성 (Generation)
- OpenAI GPT를 활용한 자연어 답변 생성
- 컨텍스트 기반 정확한 답변
- 대화 히스토리 유지

### 4. 파이프라인
- End-to-End RAG 시스템
- 모듈화된 구조로 확장 가능
- 성능 모니터링 및 로깅

---

## 📈 실험 및 개선사항

### 완료된 작업

- [ ]

### 진행 중인 작업
- [ ] 기본 RAG 파이프라인 구축


### 향후 계획
- [ ] PDF/HWP 문서 로더 구현
- [ ] OpenAI API 연동
- [ ] 메타데이터 필터링
- [ ] 청킹 전략 최적화
- [ ] Retrieval 성능 개선
- [ ] 프롬프트 엔지니어링

### 추가 작업 계획
- [ ] 온프레미스 모델 적용 (HuggingFace)
- [ ] Re-ranking 구현
- [ ] Hybrid Search (키워드 + 의미 검색)
- [ ] 웹 인터페이스 구축

---

## 📝 참고 자료

### 문서
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [FAISS Documentation](https://faiss.ai/)
- [LangChain Documentation](https://python.langchain.com/)

### 논문
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

---


## 👨‍💻 팀원

- **팀장**: OOO ([@github-username](https://github.com/))
- **팀원1**: OOO ([@github-username](https://github.com/))
- **팀원2**: OOO ([@github-username](https://github.com/))
- **팀원3**: OOO ([@github-username](https://github.com/))

---

**Last Updated**: 2024-11-10