# WebPatterns: An Interactive Dashboard for Website Text Clustering

**WebPatterns** is an interactive web application for clustering and visualizing German company websites based on their textual content. It enables exploratory analysis of business sectors using modern NLP techniques and clustering algorithms.

---

## Features

- **Preprocessed Website Content**: Texts extracted from company websites.
- **Feature Generation**: Supports TF-IDF, SBERT, and Word2Vec (FastText pretrained).
- **Clustering**: Choose between K-Means and DBSCAN.
- **Visualization**: UMAP-based 2D cluster plots with tooltips and cluster labeling.
- **Interactive Exploration**:
  - Hover to view company details.
  - Click to inspect full metadata.
  - Scrollable cluster summary.
- **Help Section**:
  - Project overview and usage guide.
  - Searchable WZ code descriptions (`kodes.json`).
  - Searchable company metadata (`extracted.json`).
  - Parameter descriptions and usage tips.


---

## Technologies Used                              

 Frontend -   React, Plotly.js, Tailwind CSS         
 Backend -    FastAPI (Python)                       
 NLP Models - TF-IDF, SBERT, Word2Vec (FastText)     
 Clustering - K-Means, DBSCAN                        
 Visualization - UMAP                            

---

## How to Use


### Backend (FastAPI)

1. bash
2. cd backend
3. python -m venv .venv
4. source .venv/bin/activate  # or .venv\Scripts\activate on Windows
5. pip install -r requirements.txt
6. uvicorn main:app --reload

### Frontend (ReactJS)
1. cd frontend
2. npm install
3. npm start

