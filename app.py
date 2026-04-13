import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from src.retrieval import retrieve_top_k, rerank

st.set_page_config(
    page_title="SciRank",
    layout="wide"
)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/arxiv_papers.csv")

@st.cache_data
def load_embeddings():
    return np.load("data/processed/sbert_embeddings.npy")

st.header("Sci-Rank")
st.subheader("Semantic Search for Scientific Literature")

st.markdown("""
- SBERT for retrieval
- Cross-Encoder (MS MARCO) for ranking
""")

with st.spinner("Loading..."):
    df = load_data()
    model = load_model()
    cross_encoder = load_encoder()
    embeddings = load_embeddings()

category_mapping = {
    "All": "None",
    "Machine Learning": "cs.LG",
    "Artificial Intelligence": "cs.AI",
    "Computer Vision": "cs.CV",
    "Natural Language Processing": "cs.CL",
    "Image and Video Processing": "eess.IV",
    "Cryptography and Security": "cs.CR"
}

query = st.text_input("Enter Query")
category_display = st.selectbox("Preferred Category", list(category_mapping.keys()))
year = st.selectbox("Prefered Year", ["All", "2020", "2019"])
k = st.slider("Number of results", 5, 20, 10)

category_code = category_mapping[category_display]

reverse_mapping = {v: k for k, v in category_mapping.items()}

if query:
    
    with st.spinner("Searching and ranking papers..."):
        candidates = retrieve_top_k(query=query, preferred_category=category_code, df=df, embeddings=embeddings, model=model, k=max(20, k))
        results = rerank(query, candidates, cross_encoder).sort_values("cross_score", ascending=False)
        if year != "All":
            results = results[results['published'] == int(year)]

        results = results.head(k)
        
        if results.empty:
            st.warning("No results found for selected filters.")

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