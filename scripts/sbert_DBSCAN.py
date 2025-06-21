import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sentence_transformers import SentenceTransformer
import umap

#CONFIG
project_root = Path(__file__).resolve().parents[1]
DATA_PATH = project_root / "data/cleaned/company_clean_docs.csv"
OUT_DIR = project_root / "experiments/sbert_dbscan_grid"
TEXT_COL = "clean_text"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

eps_list = [0.2, 0.4, 0.6, 0.8, 1.0]
min_samples_list = [5, 10, 15]

#LOAD TEXT
df = pd.read_csv(DATA_PATH)
texts = df[TEXT_COL].astype(str).tolist()

#LOAD SBERT MODEL
print("Loading SBERT model...")
model = SentenceTransformer(MODEL_NAME)

#EMBEDDING
print("Generating SBERT embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

#UMAP REDUCTION
print("🔽 Reducing dimensions with UMAP...")
X_umap = umap.UMAP(n_components=5, random_state=42).fit_transform(embeddings)

#DBSCAN GRID SEARCH
results = []
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Running DBSCAN grid search...")
for eps in eps_list:
    for min_samp in min_samples_list:
        try:
            db = DBSCAN(eps=eps, min_samples=min_samp)
            labels = db.fit_predict(X_umap)

            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = list(labels).count(-1) / len(labels)

            sil = silhouette_score(X_umap, labels) if num_clusters >= 2 else -1
            dbi = davies_bouldin_score(X_umap, labels) if num_clusters >= 2 else -1

            results.append({
                "eps": eps,
                "min_samples": min_samp,
                "num_clusters": num_clusters,
                "noise_ratio": noise_ratio,
                "silhouette_score": sil,
                "davies_bouldin_score": dbi
            })

        except Exception as e:
            print(f"[ERROR] eps={eps}, min_samples={min_samp} → {e}")

#SAVE RESULTS
results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "sbert_dbscan_results.csv", index=False)

#VISUALIZATIONS
# Heatmaps
pivot_sil = results_df.pivot(index="eps", columns="min_samples", values="silhouette_score")
pivot_dbi = results_df.pivot(index="eps", columns="min_samples", values="davies_bouldin_score")

sns.heatmap(pivot_sil, annot=True, fmt=".2f", cmap="Blues")
plt.title("Silhouette Score (SBERT + DBSCAN)")
plt.savefig(OUT_DIR / "heatmap_silhouette.png")
plt.close()

sns.heatmap(pivot_dbi, annot=True, fmt=".2f", cmap="Oranges")
plt.title("Davies-Bouldin Index (SBERT + DBSCAN)")
plt.savefig(OUT_DIR / "heatmap_dbi.png")
plt.close()

# Line plots
plt.figure(figsize=(8, 6))
sns.lineplot(data=results_df, x="eps", y="num_clusters", hue="min_samples", marker="o")
plt.title("Number of Clusters vs eps (SBERT + DBSCAN)")
plt.savefig(OUT_DIR / "clusters_vs_eps.png")
plt.close()

plt.figure(figsize=(8, 6))
sns.lineplot(data=results_df, x="eps", y="noise_ratio", hue="min_samples", marker="o")
plt.title("Noise Ratio vs eps (SBERT + DBSCAN)")
plt.savefig(OUT_DIR / "noise_vs_eps.png")
plt.close()

print(f"DONE: All results saved to {OUT_DIR}/")
