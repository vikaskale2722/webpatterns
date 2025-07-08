from pydantic import BaseModel
from typing import Tuple, Optional

class ClusteringRequest(BaseModel):
    method: str  # 'tfidf', 'sbert', 'word2vec'
    algorithm: str  # 'kmeans', 'dbscan'
    
    # KMeans param
    n_clusters: Optional[int] = 5

    # DBSCAN params
    eps: Optional[float] = 0.5
    min_samples: Optional[int] = 5

    # Common
    label_column: Optional[str] = "web"

    # TF-IDF config
    ngram_range: Optional[Tuple[int, int]] = (1, 2)
    max_features: Optional[int] = 3000
    min_df: Optional[int] = 2

    # SBERT config
    dimensions: Optional[int] = 5
