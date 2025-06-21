from fastapi import FastAPI
from routers import features, clustering
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Company Clustering API")

# Include routers
app.include_router(features.router, prefix="/features", tags=["Feature Generation"])
app.include_router(clustering.router, prefix="/clustering", tags=["Clustering"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
