from fastapi import APIRouter
from models.clustering_schema import ClusteringRequest
from services.clustering_service import run_clustering

router = APIRouter()

@router.post("/run")
def clustering(request: ClusteringRequest):
    return run_clustering(request)
