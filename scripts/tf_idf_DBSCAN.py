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

umap_dims_list = [2, 5, 10, 20, 50, 100]

eps_list = [0.2, 0.4, 0.6, 0.8, 1.0]
min_samples_list = [5, 10, 15]

#LOAD TEXt
df = pd.read_csv(DATA_PATH)
texts = df[TEXT_COL].astype(str).tolist()

#RESULTS STORAGE
results = []
OUT_DIR.mkdir(parents=True, exist_ok=True)

#GRID SEARCH
print("Starting TF-IDF + DBSCAN grid search (with UMAP dims)...")
for ngram in ngram_ranges:
    for max_feat in max_features_list:
        print(f"→ TF-IDF: ngram={ngram}, max_features={max_feat}")
        vectorizer = TfidfVectorizer(ngram_range=ngram, max_features=max_feat, min_df=min_df)
        X_tfidf = vectorizer.fit_transform(texts)
        X_pca = PCA(n_components=min(100, X_tfidf.shape[1])).fit_transform(X_tfidf.toarray())

        for umap_dims in umap_dims_list:
            print(f"  UMAP dims: {umap_dims}")
            X_umap = umap.UMAP(n_components=umap_dims, random_state=42).fit_transform(X_pca)

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
                            "umap_dims": umap_dims,
                            "eps": eps,
                            "min_samples": min_samp,
                            "num_clusters": num_clusters,
                            "noise_ratio": noise_ratio,
                            "silhouette_score": sil_score,
                            "davies_bouldin_score": db_score
                        })

                    except Exception as e:
                        print(f"[ERROR] {ngram} {max_feat} UMAP={umap_dims} eps={eps}, min_samp={min_samp} → {e}")
                        continue

#SAVE RESULTS
results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "tfidf_dbscan_results_with_umap_dims.csv", index=False)

#PLOTTING: One uncluttered heatmap per UMAP dims (for each max_features)
sns.set(font_scale=1.0)
for max_feat in max_features_list:
    for umap_dims in umap_dims_list:
        df_sub = results_df[(results_df['max_features'] == max_feat) & (results_df['umap_dims'] == umap_dims)]
        pivot_sil = df_sub.pivot_table(
            index="eps", columns="min_samples", values="silhouette_score", aggfunc="max"
        )
        plt.figure(figsize=(6, 4))
        sns.heatmap(pivot_sil, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=0.6)
        plt.title(f"Silhouette (max_feat={max_feat}, UMAP dims={umap_dims})")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"heatmap_silhouette_maxfeat{max_feat}_umap{umap_dims}.png")
        plt.close()

print(f"DONE: Results and individual heatmaps saved to {OUT_DIR}/")
