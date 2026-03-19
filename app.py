import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

# Retrieval function
from src.retrieval import rerank_with_cross_encoder

st.set_page_config(
    page_title="Scientific Literature Semantic Search",
    layout="wide"
)

# Loading data into cache for performance improvement
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_encoder():
    return CrossEncoder("cross-encoder/stsb-roberta-base")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/arxiv_papers.csv")

@st.cache_data
def load_embeddings():
    return np.load("data/processed/sbert_embeddings.npy")

st.title("Scientific Literature Semantic Search")
df = load_data()
model = load_model()
cross_encoder = load_encoder()
embeddings = load_embeddings()


# Category mapping
category_mapping = {
    "None": "None",
    "Machine Learning": "cs.LG",
    "Artificial Intelligence": "cs.AI",
    "Computer Vision": "cs.CV",
    "Natural Language Processing": "cs.CL",
    "Image and Video Processing": "eess.IV",
    "Cryptography and Security": "cs.CR"
}

# User input
query = st.text_input("Enter Query")
category_display = st.selectbox("Preferred Category", list(category_mapping.keys()))
year = st.selectbox("Prefered Year", ["None", "2020", "2019"])
k = st.slider("Number of results", 5, 20, 10)

category_code = category_mapping[category_display]

reverse_mapping = {v: k for k, v in category_mapping.items()}

if query:
    results = rerank_with_cross_encoder(query, df, embeddings, model, cross_encoder, category_code)
    results = results.head(k)

    if year != "None":
        results = results[results['published'] == int(year)]

    for _, row in results.iterrows():

            st.subheader(row["title"])
            st.write("Score:", round(row["cross_score"],3))
            st.write("Category:", ", ".join(reverse_mapping.get(x, x) for x in row["category"].split("|")))
            st.write("Published Year: ", row['published'])

            arxiv_id = row["id"].split("/")[-1]

            st.markdown(f"[{row['title']}](https://arxiv.org/abs/{arxiv_id})")

            st.write(f"Summary:\n{row['summary'][:500]}...")
            st.divider()


else: st.write("Enter a query to get results!")