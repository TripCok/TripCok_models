# unset PYTHONPATH
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pickle
import os
import pandas as pd
import warnings
import logging

from torch.utils.data import DataLoader
import torch

warnings.filterwarnings('ignore')
CSV_PATH ='/home/nishtala/TripCok/TripCok_models/src/tripcok_models/csv_maker/batch/'


class CsvLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        if not os.path.exists(csv_path):
            raise FileNotFoundError (f"\n[ERROR] BatchLoader failed to find any .csv files in {self.csv_path}.\n")
        
        self.df_list = sorted([
            entry.name for entry in os.scandir(csv_path)
            if entry.is_file() and entry.name.endswith('.csv')
        ])

    def fetch_df(self):
        data_frames = []
        for file in self.df_list:
            file_path = os.path.join(self.csv_path, file)
            try:
                df = pd.read_csv(file_path)[['contentid', 'title', 'overview']]
                data_frames.append(df)
            except FileNotFoundError:
                logging.info(f"[ERROR] could not find file {file}. Skipping...")
                continue
        
        return pd.concat(data_frames, ignore_index=True)
        


# need to make so that search() does not include the query itself as result
# maybe implement contentid? but memory is something to consider
class SemanticSearch:
    def __init__(self, model_name="jhgan/ko-sroberta-multitask"):
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.corpus = []    # list of dictionaries
        self.corpus_embeddings = None

    def add_corpus(self, contentids: pd.Series, overviews: pd.Series, batch_size: int=32):

        self.corpus = [
            {"contentid": cid, "overview": overview}
            for cid, overview in zip(contentids, overviews)
        ]

        data_loader = DataLoader(overviews_list, batch_size = batch_size)
        embeddings = []

        for batch in data_loader:
            batch embeddings = self.embedder.encode(batch, convert_to_tensor=True)
            embeddings.append(batch_embeddings)

        self.corpus_embeddings = torch.cat(embeddings, dim=0)

    # query will enter in form of contentid
    # grab corresponding overview -> run model -> exclude self
    def search(self, query, top_k=5):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, self.corpus_embeddings)[0]
        cos_scores = cos_scores.cpu()

        top_k = min(len(self.corpus), top_k+1)  # query exclusion
        top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
        results = []
        
        # re-implement self exclusion; dict will make this efficient
        for idx in top_results:
            contentid = self.corpus[idx]["contentid"]
            overview = self.corpus[idx]["overview"]
            score = float(cos_scores[idx])
            results.append({"contentid": contentid, "overview": overview, "score": score})
        
        return results


    def save_model(self, path):
        state = {
            'corpus': self.corpus, 
            'corpus_embeddings': self.corpus_embeddings,
            'model_name': self.model_name
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(state, f)
            
    @classmethod
    def load_model(cls, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)

        searcher = cls(model_name = state['model_name'])
        searcher.corpus = state['corpus']
        searcher.corpus_embeddings = state['corpus_embeddings']
        return searcher


def tester():
    # CSV 파일 읽기
    df = pd.read_csv('./batch_0.csv')

    # overview가 Unknown이 아닌 행만 필터링
    df = df[df['overview'] != 'Unknown']

    # corpus를 overview 컬럼으로 지정
    corpus = df['overview'].tolist()

    # contentid도 함께 저장
    contentids = df['contentid'].tolist()

    # 모델 초기화 및 코퍼스 추가
    searcher = SemanticSearch()
    searcher.add_corpus(corpus)

    # 검색 수행할 쿼리들
    queries = [
        "경기도 여주시의 해장국 음식점이다. 페럼 컨트리클럽에서 자동차로 5분 이내에 닿을 수 있는 가래울 해장국은 새벽 다섯 시  반부터 문을 열어 라운딩 전 아침 식사를 챙기기 좋은 식당이다. 가래울 해장국에서는 주인이 직접 만드는 정갈한 맛의 반찬과 여주 쌀로 지어진 맛있는 쌀밥을 맛볼 수 있다. 그뿐만 아니라 콩나물과 시래기가 듬뿍 들어간 가래울 해장국과 가래울 정식은 이곳의 대표 메뉴로 든든한 한 끼로 충분하다. 가래울 해장국은 골퍼들 사이에 이름난 여주 맛집이다.",
        "가래골농원 캠핑장은 포천시 창수면에 위치한 종현산 자락에 있다.  총면적 148,000m² 달하는 대지에 가족농원 캠핑장을 모토로 조성되었다. 텃밭과 동물농장이 구비돼 있으며, 펜션과 오토캠핑장으로 구성돼 있는 복합 캠핑장이다. 이곳은 테마별로 조성된 사이트를 가지고 있는데, 잣나무와 소나무가 우거져있는 소나무 사이트와 팔각정, 그리고 제한된 구역이긴 하지만 반려견 동반입장이 가능한 알프스 사이트가 있다. 캠핑장 입구에서부터 눈길을 끄는 인공연못과 조형물들은 잘 가꾸어진 정원 속에서 빛을 발한다. 뿐만 아니라 이곳은 놀거리와 체험거리가 많은 캠핑장이다. 아이들을 위한 미니농장이 있어, 양, 사슴, 토끼, 염소, 말, 공작새 등, 동물들에게 먹이주기 체험도 할 수 있고, 승마장에선 조랑말을 타보는 승마체험도 할 수 있다. 여름이면 25미터 길이의 워터슬라이드 미끄럼틀 수영장에서 짜릿한 물놀이도 즐길 수 있다.  또한 산책로가 잘 조성되어 있어 맑은 공기를 마시며 산책을 즐기기에도 좋다.",
    ]
    
    # 각 쿼리에 대해 검색 수행
    for query in queries:
        results = searcher.search(query, top_k=5)
        print("-" * 50)
        print("-" * 50)
        print(f"\nquery: {query}")
        print("-" * 50)
        print("-" * 50)
        for corpus_text, score in results:
            idx = corpus.index(corpus_text)
            print(f"contentid: {contentids[idx]}")
            print(f"overview: {corpus[idx]}")
            print(f"score: {score}")
            print("-" * 50)

    # 모델 저장
    # searcher.save_model("models/semantic_search.pkl")

    # 나중에 모델 불러오기
    # loaded_searcher = SemanticSearch.load_model("models/semantic_search.pkl")    


def main():
    model_name = "semantic_search.pkl"
    searcher = None
    
    if os.path.exists(model_path):
        searcher = SemanticSearch.load_model(model_path)
    else:
        csv_loader = CsvLoader(CSV_PATH)
        df = csv_loader.fetch_df()

        searcher = SemanticSearch()
        searcher.add_corpus(df['contentid'], df['overview'])

    # testing
    logging.info(f"[DEBUG] testing random query...")
    random_row = df.sample(1).iloc[0]
    query = random_row['overview']
    logging.info(f"[DEBUG] Testing with query: {query}")
    results = searcher.search(query, top_k=5)
    
    for overview, score in results:
        logging.log(f"[RESULT] Overview: {overview}\nScore: {score}\n{'-'*50}")




if __name__ == "__main__":
    main()
#    tester()