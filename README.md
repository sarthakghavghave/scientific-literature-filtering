# Scientific Literature Filtering (SciRank)

## Overview
A semantic literature search system for scientific research papers using pretrained sentence embeddings.
The system uses SBERT for fast candidate retrieval and a cross-encoder (MS MARCO) for accurate reranking.

The goal is to assist in literature filtering by using semantic similarity instead of keyword-based approachs.

Streamlit URL: https://semantic-lit-search.streamlit.app/

---

## Methodology

1. **Data Collection**
   - Papers retrieved using the official arXiv API
   - Metadata extracted: id, title, summary(abstract), categories, publication date

2. **Text Embeddings**
   - Embeddings generated using:
      - SBERT `all-MiniLM-L6-v2` (Sentence-BERT)
      - SPECTER `allenai-specter` (Scientific Paper Embeddings using Citation-informed TransformERs)

3. **Candidate Retrieval**
   - Cosine similarity between:
     - User topic embedding
     - Paper abstract embedding

4. **Reranking**
   - Cross Encoder Reranking
     - MS-MARCO `cross-encoder/ms-marco-MiniLM-L-6-v2`
   - Papers sorted by semantic similarity score

---

## Key Features

- Semantic search using SBERT embeddings
- Two-stage retrieval pipeline (bi-encoder + cross-encoder)
- Improved ranking using MS MARCO cross-encoder
- Category-based filtering
- Interactive Streamlit interface

---

## Tech Stack

- Python
- Sentence-Transformers
- scikit-learn
- Pandas
- NumPy

---

## Models Used

- SBERT: `all-MiniLM-L6-v2` (bi-encoder for fast retrieval)
- Cross-Encoder: `ms-marco-MiniLM-L-6-v2` (reranking)
- TF-IDF: baseline comparison
- SPECTER: comparitive study

---

## Project Structure

```
scientific-literature-filtering/

│
├── data/
│   ├── processed/
│   └── raw/
│
├── notebooks/
│   ├── 01_data_collection_and_eda.ipynb
│   ├── 02_semantic_ranking_pipeline.ipynb
│   └── 03_cross_encoder_reranking.ipynb
│
├── src/
│   ├── __init__.py
│   └── retrieval.py
│
├── app.py
├── README.md
└── requirements.txt
```

---

## Notes

- This project focuses on semantic similarity-based ranking, not supervised classification.
- Abstracts are used instead of full papers to reduce noise and computational cost.

---

## Data Source

Paper metadata and abstracts are retrieved from the arXiv API.
This project uses arXiv data solely for academic and educational purposes.
public api source - https://info.arxiv.org/help/api/basics.html