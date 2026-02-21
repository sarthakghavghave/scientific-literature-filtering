# Scientific Literature Filtering

## Overview
This project develops a semantic literature filtering system for scientific research papers using pretrained sentence embeddings.
Research paper abstracts are collected from the arXiv API (cs.LG category). The system calculates the semantic similarity between the user-defined topic and each paper abstract using Sentence-BERT embeddings and the cosine similarity measure.

The goal is to help with literature filtering by scoring research papers that are semantically similar to a research question.

---

## Methodology

1. **Data Collection**
   - Papers retrieved using the official arXiv API
   - Metadata extracted: id, title, summary(abstract), categories, publication date

2. **Text Embeddings**
   - Sentence embeddings generated using `all-MiniLM-L6-v2` (Sentence-BERT)

3. **Relevance Scoring**
   - Cosine similarity between:
     - User topic embedding
     - Paper abstract embedding

4. **Ranking**
   - Papers sorted by semantic similarity score
   - Optional filtering based on score threshold

---

## Tech Stack

- Python
- Sentence-Transformers (SBERT)
- scikit-learn
- Pandas
- NumPy

---

## Notes

- This project focuses on semantic similarity-based ranking, not supervised classification.
- Abstracts are used instead of full papers to reduce noise and computational cost.

---

## Data Source

Paper metadata and abstracts are retrieved from the arXiv API.
This project uses arXiv data solely for academic and educational purposes.