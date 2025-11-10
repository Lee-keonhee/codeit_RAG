import os
from pyhwpx import Hwp
from typing import List
import shutil

def get_hwp_files(folder_path: str) -> List[str]:
    """지정된 폴더에서 HWP 및 HWPX 파일의 전체 경로 목록을 가져옵니다."""
    hwp_files = []
    pdf_files = []
    # 폴더 내의 모든 파일과 폴더를 반복합니다.
    for filename in os.listdir(folder_path):
        # 파일의 확장자를 소문자로 변환하여 확인합니다.
        if filename.lower().endswith(('.hwp', '.hwpx')):
            # 파일의 전체 경로를 생성하여 목록에 추가합니다.
            full_path = os.path.join(folder_path, filename)
            hwp_files.append(full_path)
        elif filename.lower().endswith('.pdf'):
            full_path = os.path.join(folder_path, filename)
            pdf_files.append(full_path)

    return hwp_files, pdf_files

def batch_hwp_to_pdf(data_dir: str, target_dir: str):
    """지정된 폴더 내의 모든 HWP/HWPX 파일을 PDF로 일괄 변환합니다."""

    # 목표 폴더가 없으면 생성합니다.
    os.makedirs(target_dir, exist_ok=True)
    # ❗️ Hwp 객체는 전체 작업 동안 한 번만 생성하고 종료하는 것이 효율적입니다.
    hwp = Hwp()

    # 1. 변환할 파일 목록 가져오기
    hwp_files, pdf_files = get_hwp_files(data_dir)

    if not hwp_files:
        print(f"⚠️ 폴더 내에 변환할 HWP/HWPX 파일이 없습니다: {data_dir}")
        hwp.quit()
        return

    print(f"총 {len(hwp_files)}개의 파일을 변환합니다...")

    for i, hwp_path in enumerate(hwp_files):
        # 2. PDF 출력 경로 설정
        # 확장자를 .pdf로 변경합니다.
        pdf_path = os.path.join(target_dir, os.path.splitext(os.path.basename(hwp_path))[0] + '.pdf')

        print(f"[{i+1}/{len(hwp_files)}] 변환 시작: {os.path.basename(hwp_path)}")

        try:
            # 3. 한글 파일 열기
            hwp.open(hwp_path, 'HWP', 'ForceOpen=True')

            # 4. PDF로 저장하기
            hwp.save_as(pdf_path, format="PDF")

            print(f"   ✅ 변환 완료: {os.path.basename(pdf_path)}")

        except Exception as e:
            print(f"   ❌ 오류 발생: {os.path.basename(hwp_path)} 변환 실패. 오류 내용: {e}")

    # 5. 모든 변환 작업이 끝난 후 한글 프로그램 종료
    hwp.quit()
    print("\n🎉 일괄 변환 작업이 모두 완료되었습니다.")

    if pdf_files:
        print("\n📄 기존 PDF 파일을 복사합니다.")




        for i, pdf_path in enumerate(pdf_files):
            # 목표 폴더에 복사될 파일의 전체 경로를 만듭니다.
            target_path = os.path.join(target_dir, os.path.basename(pdf_path))

            print(f"[{i + 1}/{len(pdf_files)}] 복사 시작: {os.path.basename(pdf_path)}")

            try:
                # shutil.copy2를 사용하여 파일 내용 및 메타데이터를 복사합니다.
                shutil.copy2(pdf_path, target_path)
                print(f"   ✅ 복사 완료: {os.path.basename(target_path)}")
            except Exception as e:
                print(f"   ❌ 오류 발생: {os.path.basename(pdf_path)} 복사 실패. 오류 내용: {e}")
    else:
        print("ℹ️ 복사할 기존 PDF 파일이 없습니다.")

# --- 사용 예시 ---

# 🛑 변환할 파일들이 들어있는 폴더 경로를 정확히 지정하세요.
data_folder = "..\\data\\raw\\files"
target_folder = "..\\data\\raw\\pdf_files"
batch_hwp_to_pdf(data_folder, target_folder)
#
# path = 'D:\project_rag\data\raw\files\한영대학_한영대학교 특성화 맞춤형 교육환경 구축 - 트랙운영 학사정보.pdf'
# print(os.path.splitext(os.path.basename(path))[0]+'.pdf')
#

