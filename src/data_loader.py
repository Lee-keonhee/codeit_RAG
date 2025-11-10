'''
HWP, PDF 파일 로딩 및 메타데이터 읽기
입력 : 문서 파일 경로 (.hwp, .pdf)
       메타데이터 CSV 파일 (data_list.csv)

출력 : {
    "doc_id": "RFP_001",
    "text": "문서 전체 텍스트...",
    "metadata": {
    "발주기관": "국민연금공단",
    "사업명": "이러닝시스템 구축",
    "예산": "500,000,000원",
    "file_path": "path/to/file.pdf"
    }
}

주요 함수:
def load_pdf(file_path: str) -> str
def load_hwp(file_path: str) -> str
def load_metadata(csv_path: str) -> pd.DataFrame
def load_documents(doc_folder: str, metadata_path: str) -> List[Dict]
'''