import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

from config import FAISS_INDEX_FILE, LOCATIONS_DESCRIPTION_CSV, SENTENCE_TRANSFORMER_MODEL
from utils import search_similar_locations


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

faiss_metadata = pd.read_csv(LOCATIONS_DESCRIPTION_CSV)
faiss_index = faiss.read_index(FAISS_INDEX_FILE)
embedding_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
embedding_cache = {}


class LocationRequest(BaseModel):
    userId: str
    messages: list[dict]


@app.post("/api/find-matching-location")
async def root(request: LocationRequest):
    user_id = request.userId
    desc = request.messages[-1]['text']
    existing_embedding = embedding_cache.get(user_id)
    embedding = embedding_model.encode([desc], convert_to_numpy=True)
    if existing_embedding is not None:
        embedding += existing_embedding
        print('existing_embedding:', existing_embedding)
    embedding_cache[user_id] = embedding 
    locations = search_similar_locations(embedding, faiss_index, faiss_metadata, top_n=5)
    return locations
  
  
@app.post("/api/ask-bot-about-location")
async def root():
    print("API is called")
    time.sleep(1)
    print("Sending reply")
    return {"reply": "This is a random reply from the API."}
  
