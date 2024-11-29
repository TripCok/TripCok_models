import csv
from os.path import dirname

from tripcok_models.worker.request_utils import *
import os


def main():
    path = __file__
    dirname = os.path.dirname(path)

    CSV_FILE = os.path.join(dirname, "category.csv")
    ADMIN_ID = int(input("관리자 계정의 고유번호를 입력해주세요 : "))
    API_URL = "http://localhost:8080/api/v1/place/category"

    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f)
        csv_list = list(reader)

    for row in csv_list:
        # 첫번째 카테고리 검사
        first_category_id = None

        # 첫번째 카테고리에 같은 이름이 있는지 없는지 검사
        first_search_data = {"depth": 0, "categoryName": row[0], }
        first_request = get_api_request(API_URL, "param", first_search_data)

        if len(first_request) == 0:
            # 첫번째 카테고리 생성
            api_request = post_api_request(API_URL, {"placeName": str(row[0]), "memberId": ADMIN_ID})
            first_category_id = api_request["id"]
            print(f"CREATE : category = {api_request['name']}")
        else:
            first_category_id = first_request[0]["id"]

        # 두번째 카테고리 검사
        second_category_id = None

        # 부모 카테고리 자식에 같은 이름이 있는지 없는지 검사
        second_search_data = {"depth": 1, "categoryName": row[1], "parentId": first_category_id}
        second_request = get_api_request(API_URL, "param", second_search_data)

        if len(second_request) == 0:
            # 두번째 카테고리 생성
            api_request = post_api_request(API_URL, {"parentId": first_category_id, "placeName": str(row[1]),
                                                     "memberId": ADMIN_ID})
            second_category_id = api_request["id"]
            print(f"CREATE : category = {api_request['name']}")
        else:
            second_category_id = second_request[0]["id"]

        # 세번째 카테고리 검사
        third_category_id = None

        # 부모 카테고리 자식에 같은 이름이 있는지 없는지 검사
        third_category_data = {"depth": 2, "categoryName": row[2], "parentId": second_category_id}
        third_request = get_api_request(API_URL, "param", third_category_data)

        if len(third_request) == 0:
            # 두번째 카테고리 생성
            api_request = post_api_request(API_URL, {"parentId": second_category_id, "placeName": str(row[2]),
                                                     "memberId": ADMIN_ID})
            second_category_id = api_request["id"]
            print(f"CREATE : category = {api_request['name']}")
        else:
            second_category_id = third_request[0]["id"]
