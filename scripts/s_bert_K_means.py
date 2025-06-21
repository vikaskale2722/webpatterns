import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

#SETUP
project_root = Path(__file__).resolve().parents[1]
DATA_PATH = project_root / "data/cleaned/company_clean_docs.csv"
OUT_DIR = project_root / "experiments/sbert_kmeans_grid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "clean_text"
K_RANGE = list(range(4, 21))
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

#LOAD
df = pd.read_csv(DATA_PATH)
texts = df[TEXT_COL].astype(str).tolist()

#SBERT
print(f"Loading SBERT model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

print(f"Generating SBERT embeddings for {len(texts)} documents...")
embeddings = model.encode(texts, show_progress_bar=True)

#GRID SEARCH
results = []

print("Running KMeans for each k in range...")
for k in tqdm(K_RANGE):
    try:
        model_kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = model_kmeans.fit_predict(embeddings)

        sil = silhouette_score(embeddings, labels)
        dbi = davies_bouldin_score(embeddings, labels)
        inertia = model_kmeans.inertia_

        results.append({
            "k": k,
            "silhouette": sil,
            "davies_bouldin": dbi,
            "inertia": inertia
        })

    except Exception as e:
        print(f"[ERROR] k={k}: {e}")
        continue

#SAVE RESULTS
res_df = pd.DataFrame(results)
res_path = OUT_DIR / "sbert_kmeans_results.csv"
res_df.to_csv(res_path, index=False)

#VISUALIZATIONS

# Silhouette vs K
plt.figure(figsize=(10, 6))
sns.lineplot(data=res_df, x="k", y="silhouette", marker="o")
plt.title("SBERT: Silhouette Score vs Number of Clusters")
plt.savefig(OUT_DIR / "sbert_silhouette_vs_k.png")
plt.close()

# Davies-Bouldin vs K
plt.figure(figsize=(10, 6))
sns.lineplot(data=res_df, x="k", y="davies_bouldin", marker="o")
plt.title("SBERT: Davies-Bouldin Index vs Number of Clusters")
plt.savefig(OUT_DIR / "sbert_dbi_vs_k.png")
plt.close()

# Elbow Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=res_df, x="k", y="inertia", marker="o")
plt.title("SBERT: Elbow Plot (Inertia vs K)")
plt.savefig(OUT_DIR / "sbert_elbow_inertia_vs_k.png")
plt.close()

print(f"DONE: Results saved to {OUT_DIR}/")
