# Scientific Literature Filtering

## Overview
This project develops a semantic literature filtering system for scientific research papers using pretrained sentence embeddings.
Research paper abstracts are collected from the arXiv API (cs.LG category). The system calculates the semantic similarity between the user-defined topic and each paper abstract using SPECTER emmbeddings and cross-encoder similarity score.

The goal is to help with literature filtering by scoring research papers that are semantically similar to a research question.

---

## Methodology

1. **Data Collection**
   - Papers retrieved using the official arXiv API
   - Metadata extracted: id, title, summary(abstract), categories, publication date

2. **Text Embeddings**
   - Embeddings generated using:
      - SBERT `all-MiniLM-L6-v2` (Sentence-BERT)
      - SPECTER `allenai-specter` (Scientific Paper Embeddings using Citation-informed TransformERs)

3. **Relevance Scoring**
   - Cosine similarity between:
     - User topic embedding
     - Paper abstract embedding
   - Cross Encoder Reranking

4. **Ranking**
   - Papers sorted by semantic similarity score
   - Optional filtering based on score threshold

---

## Tech Stack

- Python
- Sentence-Transformers
- scikit-learn
- Pandas
- NumPy

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