import requests


def post_api_request(api_uri, data):
    try:
        response = requests.post(api_uri, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API 요청 실패: {e}")
        return None


def get_api_request(api_uri: str, type: str, data: dict):
    try:
        if type == "json":
            response = requests.get(api_uri, json=data)
            response.raise_for_status()
            return response.json()
        elif type == "param":
            api_uri += '?'
            for k, v in data.items():
                api_uri += f'{k}={v}'
                api_uri += '&'

            response = requests.get(api_uri)
            response.raise_for_status()

            print(response.json())
            return response.json()

    except requests.exceptions.RequestException as e:
        print(f"API 요청 실패: {e}")
        return None
