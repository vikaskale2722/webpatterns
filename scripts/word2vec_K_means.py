import os
import gzip
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from gensim.models.fasttext import load_facebook_model
from gensim.utils import simple_preprocess
import urllib.request

#CONFIG
project_root = Path(__file__).resolve().parents[1]
DATA_PATH = project_root / "data/cleaned/company_clean_docs.csv"
MODEL_DIR = project_root / "data/models"
MODEL_PATH = MODEL_DIR / "cc.de.300.bin"
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz"
OUT_DIR = project_root / "experiments/word2vec_kmeans_grid"
TEXT_COL = "clean_text"
K_RANGE = list(range(4, 21))

#DOWNLOAD FASTTEXT IF MISSING
if not MODEL_PATH.exists():
    print("FastText model not found. Downloading...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = MODEL_PATH.with_suffix(".bin.gz")
    urllib.request.urlretrieve(MODEL_URL, gz_path)
    print("Extracting model...")
    with gzip.open(gz_path, 'rb') as f_in, open(MODEL_PATH, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)
    print(f"Model saved to {MODEL_PATH}")

#LOAD DATA
df = pd.read_csv(DATA_PATH)
texts = df[TEXT_COL].astype(str).tolist()

#LOAD MODEL
print("Loading FastText model...")
model = load_facebook_model(str(MODEL_PATH))

#EMBED DOCUMENTS
def get_embedding(text):
    words = simple_preprocess(text, deacc=True)
    valid_words = [w for w in words if w in model.wv]
    if not valid_words:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[w] for w in valid_words], axis=0)

print("Generating document embeddings...")
doc_embeddings = np.vstack([get_embedding(text) for text in tqdm(texts)])

#KMEANS EXPERIMENT
results = []
OUT_DIR.mkdir(parents=True, exist_ok=True)

for k in tqdm(K_RANGE):
    try:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(doc_embeddings)

        sil = silhouette_score(doc_embeddings, labels)
        dbi = davies_bouldin_score(doc_embeddings, labels)
        inertia = kmeans.inertia_

        results.append({
            "k": k,
            "silhouette": sil,
            "davies_bouldin": dbi,
            "inertia": inertia
        })

    except Exception as e:
        print(f"[ERROR] k={k}: {e}")

#SAVE & PLOT
res_df = pd.DataFrame(results)
res_df.to_csv(OUT_DIR / "word2vec_kmeans_results.csv", index=False)

# Silhouette vs K
plt.figure(figsize=(10, 6))
sns.lineplot(data=res_df, x="k", y="silhouette", marker="o")
plt.title("Word2Vec: Silhouette Score vs K")
plt.savefig(OUT_DIR / "w2v_silhouette_vs_k.png")
plt.close()

# DBI vs K
plt.figure(figsize=(10, 6))
sns.lineplot(data=res_df, x="k", y="davies_bouldin", marker="o")
plt.title("Word2Vec: Davies-Bouldin Index vs K")
plt.savefig(OUT_DIR / "w2v_dbi_vs_k.png")
plt.close()

# Elbow Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=res_df, x="k", y="inertia", marker="o")
plt.title("Word2Vec: Inertia vs K")
plt.savefig(OUT_DIR / "w2v_elbow_inertia_vs_k.png")
plt.close()

print(f"DONE: Results saved to {OUT_DIR}/")
