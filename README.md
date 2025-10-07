# 🧠 Mini Search Engine — Machine Learning in Healthcare

A lightweight **Python search engine** built from scratch that ranks short topic-based articles about **Machine Learning in Healthcare** using **TF–IDF relevance** and **ArticleRank** (a PageRank-style centrality score computed on a similarity graph).

This project demonstrates how a simple combination of **vector space modeling** and **graph algorithms** can produce a surprisingly intelligent search system — all in under 200 lines of code.

---

## 🚀 Features

- **TF–IDF based search relevance** using cosine similarity  
- **ArticleRank**: PageRank applied on document similarity graph  
- **Hybrid scoring** combining local relevance and global importance  
- **Interactive CLI** for running queries in real time  
- Simple, interpretable, and fully self-contained — no external API or large models needed  

---

## 🧩 Technologies Used

| Component | Purpose | Library |
|------------|----------|----------|
| TF–IDF Vectorizer | Converts text to numeric feature vectors | `scikit-learn` |
| Cosine Similarity | Measures query–document and document–document similarity | `scikit-learn` |
| Graph Representation | Builds similarity graph | `networkx` |
| PageRank (ArticleRank) | Computes document importance | `networkx` |
| Arrays & Math | Normalization and scoring | `numpy` |

---

## ⚙️ How It Works

1. **Corpus Setup**  
   A small in-memory dataset (`ARTICLES`) contains 10 articles on *Machine Learning in Healthcare*.

2. **TF–IDF Vectorization**  
   Each document is transformed into a high-dimensional vector using the **TF–IDF (Term Frequency–Inverse Document Frequency)** method.

3. **Document Similarity Graph**  
   Documents are compared pairwise using cosine similarity.  
   If similarity ≥ threshold (0.12), an undirected weighted edge is created between the documents.

4. **ArticleRank Calculation**  
   The document similarity graph is fed into the **PageRank** algorithm (via NetworkX).  
   Documents central to the network (i.e., similar to many important documents) get higher ArticleRank.

5. **Hybrid Ranking**  
   When a user enters a query:
   - Compute TF–IDF cosine similarity between query and all documents.  
   - Normalize relevance and ArticleRank to [0,1].  
   - Combine with weight `alpha`:
     ```
     final_score = alpha * relevance + (1 - alpha) * article_rank
     ```
   - Sort and return top results.

6. **Interactive Search**  
   Users type queries in the console and instantly get ranked results with scores and snippets.

---

## 🧮 Equation Summary

\[
\text{FinalScore}_i = \alpha \times \text{Relevance}_i + (1 - \alpha) \times \text{ArticleRank}_i
\]

- **Relevance** — TF–IDF cosine similarity of query and document  
- **ArticleRank** — PageRank centrality score from document similarity graph  
- **α (alpha)** — Weight factor between [0,1]; default = 0.7  

---

## 🧰 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ml-health-search.git
   cd ml-health-search
