import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def retrieve_top_k(query, preferred_category, df, embeddings, model, k):
    df_scored = compute_scores(query, preferred_category, df, embeddings, model)
    return df_scored.sort_values(by='bi_score', ascending=False).head(k)

def compute_scores(query, preferred_category, df, embeddings, model):

    query_embedding = model.encode([query])
    query_embedding = normalize(query_embedding)
    
    scores = cosine_similarity(query_embedding, embeddings)[0]

    df_temp = df.copy()
    df_temp['bi_score'] = scores
    
    if preferred_category != "None":
        df_temp["cat_score"] = df_temp["category"].apply(lambda x: category_score(x, preferred_category))
        df_temp['bi_score'] = 0.9 * df_temp['bi_score'] + 0.1 * df_temp['cat_score']
    
    return df_temp


def category_score(categories, preferred_category):
    category_list = categories.split("|")

    if preferred_category in category_list:
        return 1 / len(category_list)
    
    return 0


def rerank(query, candidates, cross_encoder):
    
    pairs = [(query, text) for text in candidates["combined"]]
    cross_scores = cross_encoder.predict(pairs)
    
    candidates = candidates.reset_index(drop=True)
    candidates["cross_score"] = cross_scores
    
    return candidates.sort_values("cross_score", ascending=False)