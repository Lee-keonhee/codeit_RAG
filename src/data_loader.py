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

import os
import pandas as pd
import gethwp
from pypdf import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    # 모든 페이지를 순회하며 텍스트를 추출합니다.
    for page in reader.pages:
        text += page.extract_text()
        text += "\n\n" # 페이지 구분을 위해 줄바꿈 추가    return text
    return text

def load_hwp(file_path):
    text = gethwp.read_hwp(file_path)
    return text


def load_metadata(file_path):
    df = pd.read_csv(file_path)
    return df


def load_documents(file_dir, metadata_file_path):
    # 문서 내용 저장 리스트
    documents = []

    # 메타데이터 데이터프레임 가져오기
    metadata_df = load_metadata(metadata_file_path)
    # 필요 데이터 column
    selected_column = ['사업명','사업 금액','발주 기관','파일명','사업 요약']

    # 각 문서의 텍스트 가져오기
    file_path_list = []
    file_name_list = []
    for root, dirs, files in os.walk(file_dir):
        data = {}
        file_name_list.extend(files)
        file_path_list.extend([os.path.join(root,file) for file in files])
    for idx, (file_name, file_path) in enumerate(zip(file_name_list, file_path_list)):
        if file_path.endswith('hwp'):
            text = load_hwp(file_path)
        elif file_path.endswith('pdf'):
            text = load_pdf(file_path)
        else:
            print('지원되는 파일 형식이 아닙니다')
            continue
        metadata = metadata_df[metadata_df['파일명']==file_name][selected_column].iloc[0].to_dict()

        metadata.update({'file_path':file_path})
        data = {'doc_id': f'doc_{idx}','text': text, 'metadata':metadata}
        documents.append(data)
    return documents

if __name__=='__main__':
    config = {'data_dir':'../data/raw/files',
              'metadata_path':'../data/raw/data_list.csv',
              'chunk_size':1000,
              'overlap':200,
             }
    data_dir = config['data_dir']
    metadata_path = config['metadata_path']

    documents = load_documents(data_dir, metadata_path)
    print(documents[0])
