import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from gensim.models.fasttext import load_facebook_model
from gensim.utils import simple_preprocess
import umap
import urllib.request
import gzip
import shutil

#CONFIG
project_root = Path(__file__).resolve().parents[1]
DATA_PATH = project_root / "data/cleaned/company_clean_docs.csv"
MODEL_PATH = project_root / "data/models/cc.de.300.bin"
OUT_DIR = project_root / "experiments/word2vec_dbscan_grid"
TEXT_COL = "clean_text"

umap_dims_list = [2, 5, 10, 20, 50, 100]
eps_list = [0.2, 0.4, 0.6, 0.8, 1.0]
min_samples_list = [5, 10, 15]

#LOAD DATA
df = pd.read_csv(DATA_PATH)
texts = df[TEXT_COL].astype(str).tolist()

#DOWNLOAD FASTTEXT IF MISSING
if not MODEL_PATH.exists():
    print("Downloading FastText German model...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz"
    gz_path = MODEL_PATH.with_suffix(".bin.gz")
    urllib.request.urlretrieve(url, gz_path)
    with gzip.open(gz_path, 'rb') as f_in, open(MODEL_PATH, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)

#LOAD MODEL
print("Loading Word2Vec model...")
model = load_facebook_model(str(MODEL_PATH))

#EMBEDDINGS
def get_doc_embedding(text):
    words = simple_preprocess(text, deacc=True)
    valid = [w for w in words if w in model.wv]
    return np.mean([model.wv[w] for w in valid], axis=0) if valid else np.zeros(model.vector_size)

print("Generating document embeddings...")
doc_embeddings = np.vstack([get_doc_embedding(t) for t in tqdm(texts)])

#DBSCAN GRID WITH UMAP DIMS
results = []
OUT_DIR.mkdir(parents=True, exist_ok=True)

for umap_dims in umap_dims_list:
    print(f"\nApplying UMAP (dims={umap_dims})...")
    X_umap = umap.UMAP(n_components=umap_dims, random_state=42).fit_transform(doc_embeddings)

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
                    "umap_dims": umap_dims,
                    "eps": eps,
                    "min_samples": min_samp,
                    "num_clusters": num_clusters,
                    "noise_ratio": noise_ratio,
                    "silhouette_score": sil,
                    "davies_bouldin_score": dbi
                })

            except Exception as e:
                print(f"[ERROR] umap_dims={umap_dims}, eps={eps}, min_samples={min_samp} → {e}")

#SAVE
results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "word2vec_dbscan_results_with_umap_dims.csv", index=False)

#PLOTS
sns.set(font_scale=1.0)
for umap_dims in umap_dims_list:
    df_sub = results_df[results_df['umap_dims'] == umap_dims]
    # Heatmap: silhouette
    pivot_sil = df_sub.pivot(index="eps", columns="min_samples", values="silhouette_score")
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot_sil, annot=True, fmt=".2f", cmap="BuPu", vmin=0, vmax=0.6)
    plt.title(f"Silhouette (Word2Vec + DBSCAN, UMAP dims={umap_dims})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"heatmap_silhouette_umap{umap_dims}.png")
    plt.close()
    # Heatmap: DBI
    pivot_dbi = df_sub.pivot(index="eps", columns="min_samples", values="davies_bouldin_score")
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot_dbi, annot=True, fmt=".2f", cmap="OrRd")
    plt.title(f"Davies-Bouldin (Word2Vec + DBSCAN, UMAP dims={umap_dims})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"heatmap_dbi_umap{umap_dims}.png")
    plt.close()
    # Lineplot: clusters vs eps
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df_sub, x="eps", y="num_clusters", hue="min_samples", marker="o")
    plt.title(f"Num. Clusters vs eps (Word2Vec + DBSCAN, UMAP dims={umap_dims})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"clusters_vs_eps_umap{umap_dims}.png")
    plt.close()
    # Lineplot: noise vs eps
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df_sub, x="eps", y="noise_ratio", hue="min_samples", marker="o")
    plt.title(f"Noise Ratio vs eps (Word2Vec + DBSCAN, UMAP dims={umap_dims})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"noise_vs_eps_umap{umap_dims}.png")
    plt.close()

print(f"DONE: All results saved to {OUT_DIR}/")
