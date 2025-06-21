import os
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

#SETUP
project_root = Path(__file__).resolve().parents[1]
DATA_PATH = project_root / "data/cleaned/company_clean_docs.csv"
OUT_DIR = project_root / "experiments/tfidf_kmeans_grid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "clean_text"
TFIDF_CONFIGS = [
    {"ngram_range": (1, 1), "max_features": 1000, "min_df": 2},
    {"ngram_range": (1, 2), "max_features": 3000, "min_df": 3},
    {"ngram_range": (1, 3), "max_features": 5000, "min_df": 3},
]
K_RANGE = list(range(4, 21))  # KMeans cluster counts

#LOAD DATA
df = pd.read_csv(DATA_PATH)
texts = df[TEXT_COL].astype(str).tolist()

#GRID SEARCH
results = []

for config in TFIDF_CONFIGS:
    vectorizer = TfidfVectorizer(ngram_range=config["ngram_range"],
                                 max_features=config["max_features"],
                                 min_df=config["min_df"])
    X = vectorizer.fit_transform(texts)

    for k in K_RANGE:
        try:
            model = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = model.fit_predict(X)
            sil = silhouette_score(X, labels)
            dbi = davies_bouldin_score(X.toarray(), labels)
            inertia = model.inertia_

            results.append({
                "ngram_range": str(config["ngram_range"]),
                "max_features": config["max_features"],
                "min_df": config["min_df"],
                "num_features": X.shape[1],
                "k": k,
                "silhouette": sil,
                "davies_bouldin": dbi,
                "inertia": inertia
            })

        except Exception as e:
            print(f"[ERROR] TF-IDF={config}, k={k}: {e}")
            continue

#SAVE RESULTS
res_df = pd.DataFrame(results)
res_path = OUT_DIR / "tfidf_kmeans_results.csv"
res_df.to_csv(res_path, index=False)

#VISUALIZATION

# Silhouette vs K
plt.figure(figsize=(10, 6))
sns.lineplot(data=res_df, x="k", y="silhouette", hue="ngram_range", style="min_df", markers=True)
plt.title("Silhouette Score vs Number of Clusters (K)")
plt.tight_layout()
plt.savefig(OUT_DIR / "silhouette_vs_k.png")
plt.close()

# Davies-Bouldin vs K
plt.figure(figsize=(10, 6))
sns.lineplot(data=res_df, x="k", y="davies_bouldin", hue="ngram_range", style="min_df", markers=True)
plt.title("Davies-Bouldin Index vs Number of Clusters (K)")
plt.tight_layout()
plt.savefig(OUT_DIR / "dbi_vs_k.png")
plt.close()

# Elbow plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=res_df, x="k", y="inertia", hue="ngram_range", style="min_df", markers=True)
plt.title("Elbow Plot (Inertia vs K)")
plt.tight_layout()
plt.savefig(OUT_DIR / "elbow_inertia_vs_k.png")
plt.close()

# Heatmap: Silhouette
pivot_sil = res_df.pivot_table(index=["ngram_range", "max_features"], columns="k", values="silhouette", aggfunc="mean")
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_sil, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("Silhouette Heatmap (TF-IDF × k)")
plt.tight_layout()
plt.savefig(OUT_DIR / "heatmap_silhouette.png")
plt.close()

# Heatmap: Davies-Bouldin
pivot_dbi = res_df.pivot_table(index=["ngram_range", "max_features"], columns="k", values="davies_bouldin", aggfunc="mean")
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_dbi, annot=True, fmt=".3f", cmap="YlOrRd_r")
plt.title("Davies-Bouldin Heatmap (TF-IDF × k)")
plt.tight_layout()
plt.savefig(OUT_DIR / "heatmap_dbi.png")
plt.close()

print(f"DONE: Results and plots saved to {OUT_DIR}/")
