import csv
import json
import os
import re
from tqdm import tqdm

from tripcok_models.worker.ALP_utils.get_category import get_categories
from tripcok_models.worker.ALP_utils.reverse_geocoding import reverse_geocode
from tripcok_models.worker.ALP_utils.db_utils import *


class ALP:
    def __init__(self):
        self.db_ip = input("Enter DB IP (default: 127.0.0.1): ") or "127.0.0.1"
        self.db_port = input("Enter DB Port (default: 3306): ") or 3306
        self.db_user = input("Enter DB Username (default: root): ") or "root"
        self.db_pass = input("Enter DB Password (default: root): ") or "root"
        self.db_name = input("Enter DB Database Name (default: database): ") or "database"
        self.offset = 0
        self.read_files_path = '../csv_maker/batch'
        self.file_list = []
        self.progress_file = 'progress.json'

    def save_progress(self, progress):
        """진행 상태 저장"""
        if not isinstance(progress, dict):
            raise ValueError("progress는 딕셔너리 형식이어야 합니다.")
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f)

    def load_progress(self):
        """진행 상태 로드"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                raise ValueError("진행 상태 파일의 형식이 올바르지 않습니다.")
        return {}

    def read_csv(self):
        """CSV 파일 목록 읽기"""
        batch_path = self.read_files_path
        try:
            fetch_files = sorted(f for f in os.listdir(batch_path) if f.startswith('batch_') and f.endswith('.csv'))
            self.file_list = sorted(fetch_files, key=lambda x: int(re.search(r'\d+', x).group()))
            print(f"처리 할 파일 리스트: {len(self.file_list)}개")
        except FileNotFoundError:
            print(f"디렉토리에서 파일을 찾을 수 없습니다.: {batch_path}")

    def ask_user_choice(self):
        """사용자에게 처리된 곳부터 시작할지 묻기"""
        while True:
            choice = input("처리된 곳부터 실행하시겠습니까? (Y/n): ").strip().lower() or 'y'
            if choice in ['y', 'n']:
                return choice == 'y'
            print("잘못된 입력입니다. 'y' 또는 'n'을 입력하세요.")

    def process_csv(self):
        """CSV 파일 처리"""
        progress = self.load_progress()  # 진행 상태 로드
        if progress and not self.ask_user_choice():
            progress.clear()  # 처음부터 다시 시작

        connection = get_db_connection(self.db_ip, self.db_port, self.db_user, self.db_pass, self.db_name)
        try:
            with tqdm(total=len(self.file_list), desc="Processing Files", unit="file") as pbar:
                for file in self.file_list:
                    # 이미 완료된 파일은 건너뜀
                    if file in progress and progress[file] == -1:
                        pbar.update(1)
                        continue

                    start_row = progress.get(file, 0)  # 시작 행 번호 가져오기
                    file_path = os.path.join(self.read_files_path, file)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)

                            # 파일의 진행 상태를 확인하고 중단된 부분부터 읽기
                            for i, row in enumerate(reader):
                                if i < start_row:
                                    continue  # 이미 처리된 행은 건너뜀

                                data = {
                                    'title': row.get('title'),
                                    'contentId': row.get('contentid'),
                                    'category1': row.get('cat1'),
                                    'category2': row.get('cat2'),
                                    'category3': row.get('cat3'),
                                    'thumbnail': row.get('firstimage') if row.get('firstimage') else None,
                                    'longitude': float(row.get('mapx')),
                                    'latitude': float(row.get('mapy')),
                                    'address': reverse_geocode(row.get('mapy'), row.get('mapx')),
                                    'description': row.get('overview')
                                }

                                # 여행지 저장
                                place_id = save_to_db_place(connection, data)

                                print(place_id)

                                # Categories 가져오기
                                categories = get_categories(connection, data['category1'], data['category2'],
                                                            data['category3'])

                                # 카테고리 저장
                                if None not in categories:
                                    save_to_db_place_category(connection, place_id, categories)

                                # 이미지 저장
                                if data['thumbnail'] is not None:
                                    save_to_db_place_image(connection, place_id, data['thumbnail'])

                                # 현재 행 번호를 저장
                                progress[file] = i + 1
                                self.save_progress(progress)

                            # 파일 처리 완료 시 진행 상태에서 제거 대신 완료 표시
                            progress[file] = -1  # -1로 완료된 파일 표시
                            self.save_progress(progress)

                            pbar.update(1)

                    except Exception as e:
                        print(f"\n파일 처리 중 오류 발생: {file}, 오류: {e}")
                        break

            print("\n모든 파일 처리가 완료되었습니다!")

        except KeyboardInterrupt:
            print("\n작업이 중단되었습니다. 진행 상태가 저장됩니다.")
            self.save_progress(progress)
        finally:
            connection.close()


if __name__ == "__main__":
    alp = ALP()
    alp.read_csv()
    alp.process_csv()
