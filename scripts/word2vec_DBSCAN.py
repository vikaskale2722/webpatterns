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

eps_list = [0.2, 0.4, 0.6, 0.8, 1.0]
min_samples_list = [5, 10, 15]

#LOAD DATA
df = pd.read_csv(DATA_PATH)
texts = df[TEXT_COL].astype(str).tolist()

#DOWNLOAD FASTTEXT IF MISSING
if not MODEL_PATH.exists():
    print("🔽 Downloading FastText German model...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz"
    gz_path = MODEL_PATH.with_suffix(".bin.gz")
    urllib.request.urlretrieve(url, gz_path)
    with gzip.open(gz_path, 'rb') as f_in, open(MODEL_PATH, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)

#LOAD MODEL
print("📦 Loading Word2Vec model...")
model = load_facebook_model(str(MODEL_PATH))

#EMBEDDINGS
def get_doc_embedding(text):
    words = simple_preprocess(text, deacc=True)
    valid = [w for w in words if w in model.wv]
    return np.mean([model.wv[w] for w in valid], axis=0) if valid else np.zeros(model.vector_size)

print("Generating document embeddings...")
doc_embeddings = np.vstack([get_doc_embedding(t) for t in tqdm(texts)])

#UMAP REDUCTION
print("Applying UMAP...")
X_umap = umap.UMAP(n_components=5, random_state=42).fit_transform(doc_embeddings)

#DBSCAN GRID
results = []
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

#SAVE
results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "word2vec_dbscan_results.csv", index=False)

#PLOTS
pivot_sil = results_df.pivot(index="eps", columns="min_samples", values="silhouette_score")
pivot_dbi = results_df.pivot(index="eps", columns="min_samples", values="davies_bouldin_score")

sns.heatmap(pivot_sil, annot=True, fmt=".2f", cmap="BuPu")
plt.title("Silhouette Score (Word2Vec + DBSCAN)")
plt.savefig(OUT_DIR / "heatmap_silhouette.png")
plt.close()

sns.heatmap(pivot_dbi, annot=True, fmt=".2f", cmap="OrRd")
plt.title("Davies-Bouldin Index (Word2Vec + DBSCAN)")
plt.savefig(OUT_DIR / "heatmap_dbi.png")
plt.close()

plt.figure(figsize=(8, 6))
sns.lineplot(data=results_df, x="eps", y="num_clusters", hue="min_samples", marker="o")
plt.title("Number of Clusters vs eps (Word2Vec + DBSCAN)")
plt.savefig(OUT_DIR / "clusters_vs_eps.png")
plt.close()

plt.figure(figsize=(8, 6))
sns.lineplot(data=results_df, x="eps", y="noise_ratio", hue="min_samples", marker="o")
plt.title("Noise Ratio vs eps (Word2Vec + DBSCAN)")
plt.savefig(OUT_DIR / "noise_vs_eps.png")
plt.close()

print(f"DONE: All results saved to {OUT_DIR}/")
