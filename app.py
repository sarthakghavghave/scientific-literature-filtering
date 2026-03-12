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

@st.cache_resource
def load_model():
    return SentenceTransformer("allenai-specter")

@st.cache_resource
def load_encoder():
    return CrossEncoder("cross-encoder/stsb-roberta-base")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/arxiv_papers.csv")

@st.cache_data
def load_embeddings():
    return np.load("data/processed/specter_embeddings.npy")

st.title("Scientific Literature Semantic Search")
df = load_data()
model = load_model()
cross_encoder = load_encoder()
embeddings = load_embeddings()


query = st.text_input("Enter Query")
category = st.selectbox("Preferred Category", ["None","cs.LG","cs.CV","stat.ML","hep-ex","cs.CR","cs.NI","cs.IR","eess.IV","cs.SI"])
year = st.selectbox("Prefered Year", ["None", "2020", "2019"])
k = st.slider("Number of results", 1, 50, 10)

if query:
    results = rerank_with_cross_encoder(query, df, embeddings, model, cross_encoder, category)
    results = results.head(k)

    if year is not "None":
        results = results[results['published'] == year]

    for _, row in results.iterrows():

            st.subheader(row["title"])
            st.write("Score:", round(row["final_score"],3))
            st.write("Category:", ", ".join(row["category"].split("|")))
            st.write("Published Year: ", row['published'])

            arxiv_id = row["id"].split("/")[-1]

            st.markdown(f"[{row['title']}](https://arxiv.org/abs/{arxiv_id})")

            st.write(f"Summary:\n{row['summary'][:500]}...")
            st.divider()


else: st.write("Enter a query to get results!")