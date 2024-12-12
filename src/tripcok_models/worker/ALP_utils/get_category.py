def get_categories(db_conn, c1, c2, c3):
    """카테고리 ID를 순차적으로 가져오는 함수"""
    queries = [
        {
            "query": """
                SELECT place_category_id FROM place_category
                WHERE depth = 0 AND name = %s
            """,
            "params": (c1,),
        },
        {
            "query": """
                SELECT place_category_id FROM place_category
                WHERE depth = 1 AND parent_id = %s AND name = %s
            """,
            "params": (None, c2),  # 첫 번째 카테고리 ID로 업데이트됨
        },
        {
            "query": """
                SELECT place_category_id FROM place_category
                WHERE depth = 2 AND parent_id = %s AND name = %s
            """,
            "params": (None, c3),  # 두 번째 카테고리 ID로 업데이트됨
        },
    ]

    cursor = db_conn.cursor(dictionary=True)
    category_ids = []

    try:
        for index, query_info in enumerate(queries):
            if index > 0:  # 첫 번째 쿼리 이후에는 부모 ID 설정
                query_info["params"] = (category_ids[-1], query_info["params"][1])

            cursor.execute(query_info["query"], query_info["params"])
            result = cursor.fetchone()

            if not result:
                return []  # 결과가 없으면 빈 리스트 반환

            category_ids.append(result['place_category_id'])

        return category_ids
    finally:
        cursor.close()
