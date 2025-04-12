import os
import pickle

import numpy as np
from openai import OpenAI
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def generate_single_description(location, client):
    response = client.chat.completions.create(
		model="deepseek-chat",
		messages=[
			{"role": "user", "content":
                f"""
                Generate a detailed and engaging description for {location}, including:
                The location of the destination (where it is situated)
                The historical or cultural significance of the destination
                Key features or landmarks of the destination (e.g., architectural, natural)
                What activities or experiences are available for visitors
                Any interesting facts, local traditions, or unique aspects about the destination"

                Keep your response between 100 and 200 words
                """
                }],
		max_tokens=1024,
		temperature=0.7,
		stream=False)
    return response.choices[0].message.content
    
    
def generate_descriptions(locations_csv, client, savename, rewrite=False):
    # given a list of destinations, return a list of descriptions"
    # loop through all rows in df
    df = pd.read_csv(locations_csv)
    df["description"] = "None"
    
    if os.path.exists(savename):
        df_existing = pd.read_csv(savename)
        # merge existing descriptions, only keeping rows from the original df
        # replace descriptions with existing descriptions in df_existing
        df = pd.merge(df, df_existing, on="location", how="left")
        # only keep descriptions from the existing df
        df["description"] = df["description_y"].fillna(df["description_x"])
        df = df.drop(columns=["description_x", "description_y"])
        
    for index, row in tqdm(df.iterrows(), total=len(df)):
        location = row["location"]
        description = row["description"]
        if not rewrite and description != "None":
            continue
        description = generate_single_description(location, client)
        df.at[index, "description"] = description
        df.to_csv(savename, index=False)
  
  
def generate_embeddings(
    locations_description_csv, 
    model_name='all-MiniLM-L6-v2', 
    index_file='faiss_index.bin', 
    overwrite=True,
    ):
    """
    Generates vector embeddings for a list of sentences and stores them using FAISS on disk.
    
    :param df: Pandas DataFrame with 'location' and 'description' columns
    :param model_name: Sentence-Transformer model name
    :param index_file: File path to store the FAISS index
    :param metadata_file: File path to store metadata (location, sentence list)
    :param overwrite: If False, only compute embeddings for new sentences
    """
    
    df = pd.read_csv(locations_description_csv)
    descriptions = df['description'].to_list()
    
    model = SentenceTransformer(model_name)
    
    # Generate embeddings for new sentences
    embeddings = model.encode(descriptions, convert_to_numpy=True)
    
    # Normalize embeddings for better FAISS performance
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    # Save updated FAISS index to file
    faiss.write_index(index, index_file)
    
    print(f"Stored {len(descriptions)} new embeddings in '{index_file}'.")


def process_locations(
    locations_csv, 
    locations_description_csv, 
    faiss_index_file, 
    sentence_transformer_model,
    rewrite=False,
    ):
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    print("Creating descriptions for locations...")
    generate_descriptions(
        locations_csv, 
        client, 
        locations_description_csv, 
        rewrite=rewrite,
        )
    
    print("Generating embeddings for locations...")
    generate_embeddings(
        locations_description_csv, 
        index_file=faiss_index_file, 
        model_name=sentence_transformer_model,
        overwrite=rewrite,
        )
 
 
if __name__ == "__main__":
    from config import (
        DEEPSEEK_API_KEY,
        LOCATIONS_CSV, 
        LOCATIONS_DESCRIPTION_CSV, 
        FAISS_INDEX_FILE, 
        SENTENCE_TRANSFORMER_MODEL
        )
    
    process_locations(
        LOCATIONS_CSV, 
        LOCATIONS_DESCRIPTION_CSV, 
        FAISS_INDEX_FILE,
        SENTENCE_TRANSFORMER_MODEL,
        rewrite=False,
        )