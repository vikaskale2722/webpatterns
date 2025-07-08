import { useState } from "react";
import Sidebar from "./components/Sidebar";
import ClusterPlot from "./components/ClusterPlot";
import LoadingSpinner from "./components/LoadingSpinner";
import ClusterSummaryTable from "./components/ClusterSummaryTable";
import HelpPanel from "./components/HelpPanel"; // to be created next
import api from "./api";

export default function App() {
  const [clusterData, setClusterData] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [clustersFound, setClustersFound] = useState(null);
  const [noiseRatio, setNoiseRatio] = useState(null);
  const [summary, setSummary] = useState(null);
  const [selectedCompany, setSelectedCompany] = useState(null);
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState("clustering"); // NEW

  const handleSubmit = async ({
    featureMethod,
    featureParams,
    clusteringMethod,
    clusteringParams,
  }) => {
    setLoading(true);
    setSelectedCompany(null);
    try {
      const featureRes = await api.post(`/features/${featureMethod}`, featureParams);
      const featureData = featureRes.data;

      if (featureData.status && featureData.status !== "success") {
        alert("Feature generation failed.");
        setLoading(false);
        return;
      }

      const clusteringRes = await api.post("/clustering/run", {
        method: featureMethod,
        algorithm: clusteringMethod,
        label_column: featureParams.label_column,
        ngram_range: featureParams.ngram_range,
        max_features: featureParams.max_features,
        min_df: featureParams.min_df,
        dimensions: featureParams.dimensions,
        n_clusters: clusteringParams.n_clusters,
        eps: clusteringParams.eps,
        min_samples: clusteringParams.min_samples,
      });

      const data = clusteringRes.data;
      if (data.status !== "success") {
        alert("Clustering failed.\n" + (data.message || ""));
        setLoading(false);
        return;
      }

      setClusterData({
        umap_coords: data.umap_coords,
        labels: data.labels,
        metadata: data.metadata,
      });

      setMetrics({
        silhouette: data.silhouette_score,
        db: data.davies_bouldin_score,
      });
      setClustersFound(data.clusters_found);
      setNoiseRatio(data.noise_ratio);
      setSummary(data.cluster_summary);
    } catch (error) {
      console.error("Error in clustering pipeline:", error);
      alert("Something went wrong. See console for details.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative h-screen bg-gray-50">
      {/* Navbar in top-right corner */}
      <div className="absolute top-4 right-6 z-50">
        <button
          className={`mr-4 text-sm font-medium ${
            view === "clustering" ? "text-blue-700 underline" : "text-gray-600 hover:underline"
          }`}
          onClick={() => setView("clustering")}
        >
          Clustering
        </button>
        <button
          className={`text-sm font-medium ${
            view === "help" ? "text-blue-700 underline" : "text-gray-600 hover:underline"
          }`}
          onClick={() => setView("help")}
        >
          Help
        </button>
      </div>

      {view === "clustering" ? (
        <div className="flex h-full">
          {/* Sidebar */}
          <aside className="w-80 bg-white border-r shadow-md">
            <Sidebar onSubmit={handleSubmit} />
          </aside>

          {/* Main Clustering View */}
          <main className="flex-1 p-8 overflow-y-auto">
            <h1 className="text-2xl font-bold mb-6">Cluster Visualization</h1>

            {loading ? (
              <LoadingSpinner message="Generating features and clustering..." />
            ) : clusterData?.umap_coords ? (
              <ClusterPlot
                umapCoords={clusterData.umap_coords}
                labels={clusterData.labels}
                metadata={clusterData.metadata}
                clusterSummary={summary}
              />
            ) : (
              <p className="text-sm text-gray-500">No results yet. Run a clustering task.</p>
            )}

            {/* Evaluation Metrics */}
            <h2 className="text-xl font-semibold mt-10 text-gray-800">Evaluation Metrics</h2>
            {metrics ? (
              <ul className="list-disc ml-6 text-gray-700 space-y-1 mt-2">
                <li>Silhouette Score: {metrics.silhouette}</li>
                <li>Davies-Bouldin Index: {metrics.db}</li>
                <li>Clusters Found: {clustersFound}</li>
                <li>Noise Ratio: {noiseRatio}</li>
              </ul>
            ) : (
              <p className="text-sm text-gray-500 mt-2">No metrics to show yet.</p>
            )}

            {/* Cluster Summary Table */}
            <ClusterSummaryTable summary={summary} />
          </main>
        </div>
      ) : (
        <main className="p-8 overflow-y-auto">
          <HelpPanel />
        </main>
      )}
    </div>
  );
}
