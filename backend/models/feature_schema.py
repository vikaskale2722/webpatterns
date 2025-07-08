from pydantic import BaseModel
from typing import Tuple, Optional

#TF-IDF
class TfidfRequest(BaseModel):
    ngram_range: Tuple[int, int] = (1, 1)
    max_features: int = 3000
    min_df: int = 2
    input_csv_path: str = "data/cleaned/company_clean_docs.csv"
    text_column: str = "clean_text"
    label_column: str
    dimensions: Optional[int] = None

#SBERT
class SbertRequest(BaseModel):
    input_csv_path: str = "data/cleaned/company_clean_docs.csv"
    text_column: str = "clean_text"
    label_column: str
    dimensions: Optional[int] = None

#Word2Vec
class Word2VecRequest(BaseModel):
    input_csv_path: str = "data/cleaned/company_clean_docs.csv"
    text_column: str = "clean_text"
    label_column: str
    dimensions: Optional[int] = None
