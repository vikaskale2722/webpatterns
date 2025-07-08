import { useState } from "react";

export default function Sidebar({ onSubmit }) {
  const [featureMethod, setFeatureMethod] = useState("tfidf");
  const [clusteringMethod, setClusteringMethod] = useState("kmeans");

  // Shared
  const [labelColumn, setLabelColumn] = useState("web");

  // TF-IDF
  const [ngramRange, setNgramRange] = useState([1, 1]);
  const [maxFeatures, setMaxFeatures] = useState(3000);
  const [minDf, setMinDf] = useState(2);

  // UMAP/PCA Dimensions for reduction
  const [dimensions, setDimensions] = useState(5);

  // Clustering
  const [nClusters, setNClusters] = useState(8);
  const [eps, setEps] = useState(0.5);
  const [minSamples, setMinSamples] = useState(5);

  const handleSubmit = () => {
    const baseFeatureParams = {
      input_csv_path: "data/cleaned/company_clean_docs.csv",
      text_column: "clean_text",
    };

    const featureParams =
      featureMethod === "tfidf"
        ? {
            ...baseFeatureParams,
            ngram_range: ngramRange,
            max_features: maxFeatures,
            min_df: minDf,
            ...(clusteringMethod === "dbscan" && { dimensions }), // <--- always send for DBSCAN
          }
        : featureMethod === "sbert"
        ? {
            ...baseFeatureParams,
            ...(clusteringMethod === "dbscan" && { dimensions }), // <--- always send for DBSCAN
          }
        : {
            ...baseFeatureParams,
            ...(clusteringMethod === "dbscan" && { dimensions }), // <--- always send for DBSCAN
          };

    const clusteringParams =
      clusteringMethod === "kmeans"
        ? { n_clusters: nClusters }
        : { eps, min_samples: minSamples, ...(dimensions && { dimensions }) };

    onSubmit({
      featureMethod,
      featureParams: { ...featureParams, label_column: labelColumn },
      clusteringMethod,
      clusteringParams,
    });
  };

  // Show dimensions input only for DBSCAN, no matter which feature method
  const showDimensions = clusteringMethod === "dbscan";

  return (
    <div className="p-4 space-y-6">
      <div>
        <label className="block font-semibold mb-1">Feature Method</label>
        <select
          value={featureMethod}
          onChange={(e) => setFeatureMethod(e.target.value)}
          className="w-full border px-2 py-1 rounded"
        >
          <option value="tfidf">TF-IDF</option>
          <option value="sbert">SBERT</option>
          <option value="word2vec">Word2Vec</option>
        </select>
      </div>

      {featureMethod === "tfidf" && (
        <>
          <div>
            <label className="block font-medium text-sm">n-gram range</label>
            <input
              type="text"
              value={ngramRange.join(",")}
              onChange={(e) =>
                setNgramRange(e.target.value.split(",").map(Number))
              }
              className="w-full border px-2 py-1 rounded"
            />
          </div>
          <div>
            <label className="block font-medium text-sm">Max Features</label>
            <input
              type="number"
              value={maxFeatures}
              onChange={(e) => setMaxFeatures(Number(e.target.value))}
              className="w-full border px-2 py-1 rounded"
            />
          </div>
          <div>
            <label className="block font-medium text-sm">Min DF</label>
            <input
              type="number"
              value={minDf}
              onChange={(e) => setMinDf(Number(e.target.value))}
              className="w-full border px-2 py-1 rounded"
            />
          </div>
        </>
      )}

      {showDimensions && (
        <div>
          <label className="block font-medium text-sm">UMAP Dimensions</label>
          <input
            type="number"
            value={dimensions}
            onChange={(e) => setDimensions(Number(e.target.value))}
            className="w-full border px-2 py-1 rounded"
            min={2}
          />
        </div>
      )}

      <div>
        <label className="block font-semibold mb-1">Clustering Method</label>
        <select
          value={clusteringMethod}
          onChange={(e) => setClusteringMethod(e.target.value)}
          className="w-full border px-2 py-1 rounded"
        >
          <option value="kmeans">KMeans</option>
          <option value="dbscan">DBSCAN</option>
        </select>
      </div>

      {clusteringMethod === "kmeans" ? (
        <div>
          <label className="block font-medium text-sm">No. of Clusters</label>
          <input
            type="number"
            value={nClusters}
            onChange={(e) => setNClusters(Number(e.target.value))}
            className="w-full border px-2 py-1 rounded"
          />
        </div>
      ) : (
        <>
          <div>
            <label className="block font-medium text-sm">Epsilon</label>
            <input
              type="number"
              value={eps}
              onChange={(e) => setEps(Number(e.target.value))}
              className="w-full border px-2 py-1 rounded"
            />
          </div>
          <div>
            <label className="block font-medium text-sm">Min Samples</label>
            <input
              type="number"
              value={minSamples}
              onChange={(e) => setMinSamples(Number(e.target.value))}
              className="w-full border px-2 py-1 rounded"
            />
          </div>
        </>
      )}

      <div>
        <label className="block font-medium text-sm">Label Column</label>
        <input
          type="text"
          value={labelColumn}
          onChange={(e) => setLabelColumn(e.target.value)}
          className="w-full border px-2 py-1 rounded"
        />
      </div>

      <button
        onClick={handleSubmit}
        className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700"
      >
        Generate & Cluster
      </button>
    </div>
  );
}
