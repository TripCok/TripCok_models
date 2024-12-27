# generates random query for FastAPI
import pickle
from semantic2 import CsvLoader
import os
import pandas as pd
import textwrap

QUERY_LENGTH = 5
CSV_PATH ='/home/nishtala/TripCok/TripCok_models/src/tripcok_models/csv_maker/batch/'

def main():
    csv_loader = CsvLoader(CSV_PATH)
    df = csv_loader.fetch_df()

    random_cids = df['contentid'].sample(QUERY_LENGTH).tolist()
    query = textwrap.dedent(f"""
    {{
        "contentids": [{", ".join(map(str, random_cids))}],
        "top_k": 5
    }}
    """)
    print(query)


if __name__ == "__main__":
    main()