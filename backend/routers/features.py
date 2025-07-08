from fastapi import APIRouter
from models.feature_schema import TfidfRequest, SbertRequest, Word2VecRequest
from services.feature_service import (
    generate_tfidf_features,
    generate_sbert_features,
    generate_word2vec_features
)

router = APIRouter()

@router.post("/tfidf")
def tfidf(request: TfidfRequest):
    df = generate_tfidf_features(request)
    return {"status": "success", "n_rows": len(df), "preview": df.head(5).to_dict()}

@router.post("/sbert")
def sbert(request: SbertRequest):
    df = generate_sbert_features(request)
    return {"status": "success", "n_rows": len(df), "preview": df.head(5).to_dict()}

@router.post("/word2vec")
def word2vec(request: Word2VecRequest):
    df = generate_word2vec_features(request)
    return {"status": "success", "n_rows": len(df), "preview": df.head(5).to_dict()}
