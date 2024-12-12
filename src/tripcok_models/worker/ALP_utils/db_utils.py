from datetime import datetime

import mysql.connector


def get_db_connection(db_ip, db_port, db_user, db_pass, db_name):
    """DB 연결 생성"""
    try:
        return mysql.connector.connect(
            host=db_ip,
            port=db_port,
            user=db_user,
            password=db_pass,
            database=db_name,
            charset='utf8mb4',  # MariaDB에서 호환 가능한 utf8mb4 설정
            collation='utf8mb4_general_ci'
        )
    except mysql.connector.Error as e:
        print(f"DB 연결 실패: {e}")
        raise


def save_to_db_place(conn, data):
    """데이터를 DB에 저장하고 저장된 고유 번호 반환"""
    query = """
    INSERT INTO place (name, ml_mapping_id, description, longitude, latitude, address, create_time, update_time)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (
        data['title'],
        data['contentId'],
        data['description'],
        data['longitude'],
        data['latitude'],
        data['address'],
        datetime.now(),
        datetime.now()
    )
    try:
        cursor = conn.cursor()
        cursor.execute(query, values)
        conn.commit()
        last_id = cursor.lastrowid
        return last_id
    except mysql.connector.Error as e:
        print(f"데이터 저장 실패: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()


def save_to_db_place_category(conn, place_id, categories):
    """place_id와 categories를 place_category 테이블에 저장"""
    query = """
    INSERT INTO place_category_mapping (place_id, category_id, create_time, update_time)
    VALUES (%s, %s, %s, %s)
    """

    try:
        cursor = conn.cursor()
        data = [(place_id, category, datetime.now(), datetime.now()) for category in categories]
        cursor.executemany(query, data)
        conn.commit()

        print(f"{place_id}의 카테고리 : {len(data)}개의 카테고리가 저장되었습니다.")
    except mysql.connector.IntegrityError as e:
        print(f"중복 삽입 방지: {e}")
    except mysql.connector.Error as e:
        print(f"카테고리 저장 실패: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()


def save_to_db_place_image(conn, place_id, image_url):
    query = """
        INSERT INTO place_image (place_id, file_name, image_path, image_type)
        VALUES (%s, %s, %s, %s)
        """

    cursor = conn.cursor()
    cursor.execute(query, (place_id, 'placeholder.jpg', image_url, 'T'))
    conn.commit()
    cursor.close()
