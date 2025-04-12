import faiss


def search_similar_locations(query_embedding, index, metadata_df, top_n=5):
    faiss.normalize_L2(query_embedding)
    # Search for the top N matches
    distances, indices = index.search(query_embedding, top_n)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(metadata_df):
            results.append({
                "name": metadata_df.iloc[idx]['location'],
                "description": metadata_df.iloc[idx]['description'],
                "score": 1 - dist})
    return results
