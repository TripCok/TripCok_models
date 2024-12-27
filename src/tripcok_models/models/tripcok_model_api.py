import pickle
from tripcok_model import SemanticSearch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import torch
import uvicorn
import logging

MODEL_PATH = "semantic_search.pkl"

# --- App Initialization
app = FastAPI(title="Semantic Search API", version = "2.0.0")
recommender = None
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Required to read pickled SemanticSearch class
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "SemanticSearch":
            return SemanticSearch
        return super().find_class(module, name)

# -- Pydantic: QOL library for data validation & integration
class SearchQuery(BaseModel):
    contentids: list[int]
    top_k: int=5



# --- Model Loading Function
def load_model():
    global recommender
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"[ERROR] Required file '{file}' not found. Exiting...")

        with open(MODEL_PATH, 'rb') as f:
            recommender = CustomUnpickler(f).load()
        logging.info(f"[INFO] Successfully loaded model.")
    except Exception as e:
        logging.error(f"[ERROR] Failed to load model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model. Check logging info...")

# --- FastAPI Endpoints
@app.get("/")
async def root():
    return {"message": "Welcome to the Semantic Search API!"}

@app.post("/recommend/")
async def search_contentids(query: SearchQuery):
    """
    takes a list of contentids
    returns a list of dictionaries, in format {cid: score}
    """
    try:
        results = recommender.search_cid(query.contentids, top_k=query.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.on_event("startup")
async def startup_event():
    load_model()