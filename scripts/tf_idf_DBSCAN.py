import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import DBSCAN
import umap

#CONFIG
project_root = Path(__file__).resolve().parents[1]
DATA_PATH = project_root / "data/cleaned/company_clean_docs.csv"
OUT_DIR = project_root / "experiments/tfidf_dbscan_grid"
TEXT_COL = "clean_text"

ngram_ranges = [(1, 1), (1, 2), (1, 3)]
max_features_list = [1000, 3000, 5000]
min_df = 2

eps_list = [0.2, 0.4, 0.6, 0.8, 1.0]
min_samples_list = [5, 10, 15]

#LOAD TEXT
df = pd.read_csv(DATA_PATH)
texts = df[TEXT_COL].astype(str).tolist()

#RESULTS STORAGE
results = []
OUT_DIR.mkdir(parents=True, exist_ok=True)

#GRID SEARCH
print("🔁 Starting TF-IDF + DBSCAN grid search...")
for ngram in ngram_ranges:
    for max_feat in max_features_list:
        print(f"→ TF-IDF: ngram={ngram}, max_features={max_feat}")

        vectorizer = TfidfVectorizer(ngram_range=ngram, max_features=max_feat, min_df=min_df)
        X_tfidf = vectorizer.fit_transform(texts)

        # PCA → UMAP for dimensionality reduction
        X_pca = PCA(n_components=min(100, X_tfidf.shape[1])).fit_transform(X_tfidf.toarray())
        X_umap = umap.UMAP(n_components=5, random_state=42).fit_transform(X_pca)

        for eps in eps_list:
            for min_samp in min_samples_list:
                try:
                    model = DBSCAN(eps=eps, min_samples=min_samp)
                    labels = model.fit_predict(X_umap)

                    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    noise_ratio = list(labels).count(-1) / len(labels)

                    sil_score = silhouette_score(X_umap, labels) if num_clusters >= 2 else -1
                    db_score = davies_bouldin_score(X_umap, labels) if num_clusters >= 2 else -1

                    results.append({
                        "ngram_range": str(ngram),
                        "max_features": max_feat,
                        "min_df": min_df,
                        "eps": eps,
                        "min_samples": min_samp,
                        "num_clusters": num_clusters,
                        "noise_ratio": noise_ratio,
                        "silhouette_score": sil_score,
                        "davies_bouldin_score": db_score
                    })

                except Exception as e:
                    print(f"[ERROR] {ngram} {max_feat} eps={eps}, min_samp={min_samp} → {e}")
                    continue

#SAVE RESULTS
results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "tfidf_dbscan_results.csv", index=False)

#PLOTTING
pivot_sil = results_df.pivot_table(index="eps", columns="max_features", values="silhouette_score", aggfunc="max")
pivot_dbi = results_df.pivot_table(index="eps", columns="max_features", values="davies_bouldin_score", aggfunc="min")

sns.heatmap(pivot_sil, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Silhouette Score (Max) vs eps/max_features")
plt.savefig(OUT_DIR / "heatmap_silhouette.png")
plt.close()

sns.heatmap(pivot_dbi, annot=True, fmt=".2f", cmap="YlOrBr")
plt.title("Davies-Bouldin Index (Min) vs eps/max_features")
plt.savefig(OUT_DIR / "heatmap_dbi.png")
plt.close()

print(f"DONE: Results saved to {OUT_DIR}/")
