#from semantic2 import CsvLoader, SemanticSearch         # classes, relative import
#from semantic2 import testing_text, testing_cid         # func, relative import
import pickle

from semantic2 import CsvLoader, SemanticSearch, testing_text, testing_cid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import torch
import uvicorn
import logging

#import importlib

# --- Constants ---
MODEL_PATH = "semantic_search.pkl"
CSV_PATH ='/home/nishtala/TripCok/TripCok_models/src/tripcok_models/csv_maker/batch/'
EMBEDDINGS_PATH = "corpus_embeddings.pt"
CONTENTID_PATH = "contentid_map.pkl"

# --- App Initialization ---
app = FastAPI(title="Semantic Search API", version="1.0.0")
searcher = None
df = None

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 중요!!!! pickle class sanity
class CustomUnPickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "SemanticSearch":
            return SemanticSearch
        return super().find_class(module, name)


# --- Model Loading Function --- 
def load_model():
    global searcher, df
    try:
        for file in [MODEL_PATH, EMBEDDINGS_PATH, CONTENTID_PATH]:
            if not os.path.exists(file):
                raise FileNotFoundError(f"[ERROR] Required file '{file}' not found. Cannot proceed.")

        # --- Load Data and Model ---
        csv_loader = CsvLoader(CSV_PATH)
        df = csv_loader.fetch_df()

        with open(MODEL_PATH, 'rb') as f:
            searcher = CustomUnPickler(f).load()

        logging.info(f"[INFO] Successfully loaded model.")

    except Exception as e:
        logging.error(f"[ERROR] Failed to load model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# --- Pydantic Models ---
class SearchQuery(BaseModel):
    contentids: list[str]
    top_k: int = 5

# --- Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Semantic Search API!"}

@app.post("/search/")
async def search_contentids(query: SearchQuery):
    """
    Search for similar content IDs based on input content IDs.
    """
    try:
        results = searcher.search_cid(query.contentids, top_k=query.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Test Endpoints
@app.get("/test/text/")
async def test_text(query_num: int = 1):
    """
    Run a random query test with text input.
    """
    global df
    try:
        results: testing_text(searcher, df, query_num)
        return {
            "message": f"Ran {query_num} text queries successfully.",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/test/cid/")
async def test_cid(query_num: int = 1):
    """
    Run a random query test with content ID input.
    """
    global df
    try:
        results = testing_cid(searcher, df, query_num)
        return {
            "message": f"Ran {query_num} content ID queries successfully.",
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Startup Event to Load the Model ---
@app.on_event("startup")
async def startup_event():
    load_model()

# --- Entry Point for Debugging ---
if __name__ == "__main__":
    uvicorn.run("place_recom:app", host="0.0.0.0", port=8000, reload=False)
