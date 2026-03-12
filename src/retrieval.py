import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Retrieve top k results for a query
def retrieve_top_k(query, preferred_category, df, embeddings, model, k):

    df_scored = compute_scores(query, preferred_category, df, embeddings, model)

    return df_scored.sort_values(by='final_score', ascending=False).head(k)

# Compute cosine similarity scores
def compute_scores(query, preferred_category, df, embeddings, model):

    # query embedding
    query_embedding = model.encode([query])
    query_embedding = normalize(query_embedding)
    
    scores = cosine_similarity(query_embedding, embeddings)[0]

    df_temp = df.copy()
    df_temp['biencoder_score'] = scores
    
    if preferred_category != "None":

        df_temp["cat_score"] = df_temp["category"].apply(lambda x: category_score(x, preferred_category))

        df_temp['final_score'] = 0.9 * df_temp['biencoder_score'] + 0.1 * df_temp['cat_score']

    else:
        df_temp['final_score'] = df_temp['biencoder_score']
    
    return df_temp

# Category-based scoring
def category_score(categories, preferred_category):
    category_list = categories.split("|")

    if preferred_category in category_list:
        return 1 / len(category_list)
    
    return 0

def rerank_with_cross_encoder(query, df, embeddings, bi_encoder, cross_encoder, k=20, preferred_category=None):
    """
    Two-stage retrieval pipeline.
    Stage 1 - Bi-encoder: fast candidate retrieval using cosine similarity.
    Stage 2 - Cross-encoder: precise reranking of top-k candidates.
    """
    
    # Bi-encoder Retrieval
    candidates = retrieve_top_k(query=query, preferred_category=preferred_category, df=df, embeddings=embeddings, model=bi_encoder, k=k)
    
    # Cross-encoder reranking
    pairs = [(query, text) for text in candidates["summary"]]
    cross_scores = cross_encoder.predict(pairs)
    
    candidates = candidates.reset_index(drop=True)
    candidates["cross_score"] = cross_scores
    
    return candidates.sort_values("cross_score", ascending=False)