# ml_health_search.py
"""
Tiny search engine on the topic: Machine Learning in Healthcare
- TF-IDF for query relevance
- ArticleRank (PageRank over similarity graph) for document centrality
- Final rank = alpha * relevance + (1-alpha) * article_rank_normalized

Run: pip install scikit-learn networkx numpy
     python ml_health_search.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
from pprint import pprint

# --- Small example corpus about "Machine Learning in Healthcare" ---
ARTICLES = {
    "D1": {
        "title": "ML for Medical Imaging",
        "text": "Deep learning improves medical imaging diagnosis. CNNs detect tumors, segment organs, and assist radiologists with image interpretation."
    },
    "D2": {
        "title": "Predictive Models for Patient Risk",
        "text": "Supervised models estimate hospital readmission and predict patient deterioration. Features include labs, vitals, and prior admissions."
    },
    "D3": {
        "title": "Electronic Health Records (EHR) & NLP",
        "text": "NLP on EHR notes extracts symptoms and medications. Named entity recognition helps structure unstructured clinical texts for downstream tasks."
    },
    "D4": {
        "title": "Federated Learning in Healthcare",
        "text": "Federated learning enables training across hospitals without sharing raw patient data. Privacy-preserving methods reduce data leakage risks."
    },
    "D5": {
        "title": "Explainability and Model Trust",
        "text": "Explainable AI helps clinicians understand model predictions. Techniques like SHAP and LIME identify important features for decisions."
    },
    "D6": {
        "title": "Drug Discovery with ML",
        "text": "ML accelerates drug discovery by predicting molecular properties, suggesting candidate compounds, and modeling protein-ligand interactions."
    },
    "D7": {
        "title": "Time Series Analysis for ICU Monitoring",
        "text": "Recurrent and transformer models analyze ICU time-series vitals for early warning systems and sepsis prediction."
    },
    "D8": {
        "title": "Bias and Fairness in Clinical Models",
        "text": "Clinical models must be audited for bias across demographics. Fairness-aware training mitigates disparate impact on vulnerable groups."
    },
    "D9": {
        "title": "Reinforcement Learning for Treatment Policies",
        "text": "Reinforcement learning can personalize treatment policies, optimize dosing schedules, and adapt interventions over time."
    },
    "D10": {
        "title": "Regulatory & Deployment Challenges",
        "text": "Deploying ML in hospitals faces regulatory hurdles, validation requirements, and the need for robust monitoring post-deployment."
    }
}

# --- Prepare corpus lists ---
ids = list(ARTICLES.keys())
docs = [ARTICLES[_]["text"] for _ in ids]
titles = {k: ARTICLES[k]["title"] for k in ids}

# --- Build TF-IDF index ---
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
tfidf = vectorizer.fit_transform(docs)  # shape: (n_docs, n_terms)

# --- Document-document similarity matrix (cosine) ---
sim_matrix = cosine_similarity(tfidf)
np.fill_diagonal(sim_matrix, 0.0)  # no self-links

# --- Build similarity graph and run PageRank (ArticleRank) ---
G = nx.Graph()
for idx, doc_id in enumerate(ids):
    G.add_node(doc_id)

# Threshold edges (keeps graph sparse). If no edges survive, we'll fallback to k-NN connections.
THRESHOLD = 0.12
for i in range(len(ids)):
    for j in range(i+1, len(ids)):
        w = float(sim_matrix[i, j])
        if w >= THRESHOLD:
            G.add_edge(ids[i], ids[j], weight=w)

# fallback: if graph has no/few edges, connect each doc to its top-k nearest neighbors
if G.number_of_edges() < len(ids) // 2:
    k = 2
    for i in range(len(ids)):
        sims = [(j, sim_matrix[i, j]) for j in range(len(ids)) if j != i]
        sims_sorted = sorted(sims, key=lambda x: -x[1])
        for j, w in sims_sorted[:k]:
            if w > 0:
                G.add_edge(ids[i], ids[j], weight=float(w))

# Use PageRank on this weighted graph
pagerank_scores = nx.pagerank(G, weight='weight')  # dict {doc_id: score}

# --- Normalize helper ---
def normalize_arr(arr):
    arr = np.array(arr, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

# --- Search function ---
def search(query, top_k=5, alpha=0.7):
    """
    query: str
    top_k: number of results
    alpha: weight for relevance (TF-IDF cosine) vs ArticleRank
           final = alpha * relevance_norm + (1-alpha) * art_rank_norm
    """
    q_vec = vectorizer.transform([query])
    relevances = cosine_similarity(q_vec, tfidf).flatten()
    rel_norm = normalize_arr(relevances)

    pr_values = np.array([pagerank_scores.get(i, 0.0) for i in ids])
    pr_norm = normalize_arr(pr_values)

    final = alpha * rel_norm + (1.0 - alpha) * pr_norm
    ranked_idx = np.argsort(-final)[:top_k]

    results = []
    for idx in ranked_idx:
        doc_id = ids[idx]
        results.append({
            "id": doc_id,
            "title": titles[doc_id],
            "final_score": float(final[idx]),
            "relevance_score": float(rel_norm[idx]),
            "article_rank_score": float(pr_norm[idx]),
            "snippet": ARTICLES[doc_id]["text"][:240]
        })
    return results

# --- Simple CLI demo ---
def demo():
    print("Tiny search engine: topic = Machine Learning in Healthcare")
    print("Type a query and press Enter. Type 'exit' to quit.")
    print("You can adjust 'alpha' to prefer relevance (1.0) or centrality (0.0).")
    while True:
        q = input("\nQuery> ").strip()
        if q.lower() in ("exit", "quit"):
            print("Bye — happy searching!")
            break
        # optionally allow alpha adjustment inline like: "alpha=0.5: sepsis prediction"
        alpha = 0.7
        if q.startswith("alpha="):
            try:
                left, rest = q.split(":", 1)
                alpha = float(left.split("=", 1)[1])
                q = rest.strip()
            except Exception:
                print("Bad alpha syntax. Use e.g. alpha=0.5: sepsis prediction")
                continue
        results = search(q, top_k=5, alpha=alpha)
        print(f"\nResults (alpha={alpha}):")
        for r in results:
            print(f"- {r['id']} | {r['title']} | score={r['final_score']:.3f} (rel={r['relevance_score']:.3f}, AR={r['article_rank_score']:.3f})")
            print(f"  {r['snippet']}")
        print()

if __name__ == "__main__":
    # show a few example queries automatically, then drop to interactive demo
    example_queries = [
        "tumor detection in medical images",
        "privacy across hospitals without sharing data",
        "predict patient readmission using vitals",
        "explainable AI for clinicians",
        "early warning systems in ICU"
    ]
    print("Example queries and top result:\n")
    for q in example_queries:
        top = search(q, top_k=1, alpha=0.75)[0]
        print(f"Query: {q}")
        print(f" -> {top['id']} | {top['title']} | score={top['final_score']:.3f}")
        print(f"    {top['snippet']}\n")
    print("Entering interactive mode.")
    demo()
