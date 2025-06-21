import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from models.feature_schema import TfidfRequest, SbertRequest, Word2VecRequest
from gensim.models.fasttext import load_facebook_model
from gensim.utils import simple_preprocess
import urllib.request
import gzip
import shutil

project_root = Path(__file__).resolve().parents[2]
features_dir = project_root / "data/features"
features_dir.mkdir(parents=True, exist_ok=True)

def save_features_and_metadata(features_df, metadata_df, base_filename):
    features_df.columns = [f"f_{i}" for i in range(features_df.shape[1])]
    features_df.to_csv(features_dir / f"{base_filename}_features.csv", index=False)
    metadata_df.to_csv(features_dir / f"{base_filename}_meta.csv", index=False)

def extract_metadata(df, label_column):
    meta_cols = ["web", "wz_code", "zweck"]
    if label_column not in meta_cols:
        meta_cols.append(label_column)
    return df[meta_cols].copy()

#TF-IDF
def generate_tfidf_features(req: TfidfRequest):
    df = pd.read_csv(project_root / req.input_csv_path)
    texts = df[req.text_column].astype(str).tolist()

    vectorizer = TfidfVectorizer(
        ngram_range=req.ngram_range,
        max_features=req.max_features,
        min_df=req.min_df
    )
    X = vectorizer.fit_transform(texts)
    features_df = pd.DataFrame(X.toarray())
    metadata_df = extract_metadata(df, req.label_column)

    base = f"tfidf_ng{req.ngram_range[0]}{req.ngram_range[1]}_max{req.max_features}_df{req.min_df}"
    save_features_and_metadata(features_df, metadata_df, base)
    return features_df

#SBERT
def generate_sbert_features(req: SbertRequest):
    from sentence_transformers import SentenceTransformer

    df = pd.read_csv(project_root / req.input_csv_path)
    texts = df[req.text_column].astype(str).tolist()

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(texts, show_progress_bar=True)
    features_df = pd.DataFrame(embeddings)
    metadata_df = extract_metadata(df, req.label_column)

    base = f"sbert_umap{req.dimensions}"
    save_features_and_metadata(features_df, metadata_df, base)
    return features_df

#Word2Vec (FastText)
def generate_word2vec_features(req: Word2VecRequest):
    df = pd.read_csv(project_root / req.input_csv_path)
    texts = df[req.text_column].astype(str).tolist()

    model_path = project_root / "data/models/cc.de.300.bin"
    if not model_path.exists():
        print("🔽 Downloading FastText model...")
        gz_path = model_path.with_suffix(".bin.gz")
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz"
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, 'rb') as f_in, open(model_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()

    print("Loading the model")
    model = load_facebook_model(str(model_path))
    print("Finished loading the model")

    def doc_vector(text):
        tokens = simple_preprocess(text)
        valid = [t for t in tokens if t in model.wv]
        return np.mean([model.wv[t] for t in valid], axis=0) if valid else np.zeros(model.vector_size)

    embeddings = np.vstack([doc_vector(t) for t in texts])
    features_df = pd.DataFrame(embeddings)
    metadata_df = extract_metadata(df, req.label_column)

    base = "word2vec_fasttext"
    save_features_and_metadata(features_df, metadata_df, base)
    return features_df
