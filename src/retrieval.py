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
    df_temp['sbert_score'] = scores
    
    if preferred_category != "None":

        df_temp["cat_score"] = df_temp["category"].apply(lambda x: category_score(x, preferred_category))

        df_temp['final_score'] = 0.9 * df_temp['sbert_score'] + 0.1 * df_temp['cat_score']

    else:
        df_temp['final_score'] = df_temp['sbert_score']
    
    return df_temp

# Category-based scoring
def category_score(categories, preferred_category):
    category_list = categories.split("|")

    if preferred_category in category_list:
        return 1 / len(category_list)
    
    return 0