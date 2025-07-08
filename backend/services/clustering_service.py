from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from pathlib import Path
import pandas as pd
import json
import umap
import math
from models.clustering_schema import ClusteringRequest

project_root = Path(__file__).resolve().parents[2]
features_dir = project_root / "data/features"
cluster_dir = project_root / "data/clusters"
cluster_dir.mkdir(parents=True, exist_ok=True)

def make_json_safe(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    return obj

def run_clustering(req: ClusteringRequest):
    # --- Determine base filename for features ---
    if req.method == "tfidf":
        base = f"tfidf_ng{req.ngram_range[0]}{req.ngram_range[1]}_max{req.max_features}_df{req.min_df}"
    elif req.method == "sbert":
        base = "sbert"
    elif req.method == "word2vec":
        base = "word2vec_fasttext"
    else:
        return {"status": "error", "message": f"Unsupported method: {req.method}"}

    # --- Use UMAP-reduced features for DBSCAN if dimensions param is set ---
    feat_file = f"{base}_features.csv"
    if req.algorithm == "dbscan" and getattr(req, "dimensions", None) is not None:
        feat_file = f"{base}_umap{req.dimensions}_features.csv"
    feat_path = features_dir / feat_file
    meta_path = features_dir / f"{base}_meta.csv"

    if not feat_path.exists() or not meta_path.exists():
        return {"status": "error", "message": f"Required input files missing for {feat_file}"}

    X = pd.read_csv(feat_path)
    metadata = pd.read_csv(meta_path)

    if req.label_column not in metadata.columns:
        return {"status": "error", "message": f"Label column '{req.label_column}' missing in metadata."}

    if req.algorithm == "kmeans":
        model = KMeans(n_clusters=req.n_clusters, random_state=42)
        labels = model.fit_predict(X)
        algo_suffix = f"kmeans_k{req.n_clusters}"
    elif req.algorithm == "dbscan":
        model = DBSCAN(eps=req.eps, min_samples=req.min_samples)
        labels = model.fit_predict(X)
        algo_suffix = f"dbscan_eps{req.eps}_min{req.min_samples}_dims{getattr(req, 'dimensions', 'full')}"
    else:
        return {"status": "error", "message": f"Unsupported clustering algorithm: {req.algorithm}"}

    metadata["cluster"] = labels
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = list(labels).count(-1) / len(labels)
    silhouette = silhouette_score(X, labels) if num_clusters >= 2 else -1
    dbi = davies_bouldin_score(X, labels) if num_clusters >= 2 else -1

    X_array = X.to_numpy()
    reducer = PCA(n_components=2) if X_array.shape[0] < 5 else umap.UMAP(
        n_components=2,
        n_neighbors=max(2, min(10, X_array.shape[0] - 1)),
        random_state=42
    )

    try:
        umap_coords = reducer.fit_transform(X_array)
    except Exception as e:
        return {"status": "error", "message": f"UMAP or PCA failed: {str(e)}"}

    output_name = f"{req.method}_{algo_suffix}.csv"
    metadata[[req.label_column, "cluster"]].to_csv(cluster_dir / output_name, index=False)

    metrics_path = cluster_dir / "metrics.csv"
    metrics_row = {
        "method": req.method,
        "algorithm": req.algorithm,
        "file": output_name,
        "num_clusters": num_clusters,
        "noise_ratio": noise_ratio,
        "silhouette_score": silhouette,
        "davies_bouldin_score": dbi,
        "params": json.dumps(req.dict())
    }

    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics_row])], ignore_index=True)
    else:
        metrics_df = pd.DataFrame([metrics_row])
    metrics_df.to_csv(metrics_path, index=False)

    cluster_summary = []
    if "wz_code" in metadata.columns:
        for cluster_id in sorted(metadata["cluster"].unique()):
            if cluster_id == -1:
                top_code = "N/A"
                size = len(metadata[metadata["cluster"] == -1])
            else:
                cluster_df = metadata[metadata["cluster"] == cluster_id]
                wz_counts = cluster_df["wz_code"].value_counts()
                top_code = wz_counts.idxmax()
                size = len(cluster_df)

            cluster_summary.append({
                "cluster": int(cluster_id),
                "label": f"Cluster {cluster_id}: WZ {top_code}",
                "size": size,
                "top_wz_code": top_code
            })

    response = {
        "status": "success",
        "saved_as": str(cluster_dir / output_name),
        "clusters_found": num_clusters,
        "noise_ratio": noise_ratio,
        "silhouette_score": silhouette,
        "davies_bouldin_score": dbi,
        "umap_coords": umap_coords.tolist(),
        "labels": labels.tolist(),
        "metadata": metadata.to_dict(orient="records"),
        "cluster_summary": cluster_summary
    }

    return make_json_safe(response)
