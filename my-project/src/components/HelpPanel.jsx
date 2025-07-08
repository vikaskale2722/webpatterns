import { useState, useEffect } from "react";
import kodes from "../data/kodes.json";
import extracted from "../data/extracted.json";

export default function HelpPanel() {
  const [tab, setTab] = useState("overview");
  const [search, setSearch] = useState("");
  const [filteredKodes, setFilteredKodes] = useState([]);
  const [filteredCompanies, setFilteredCompanies] = useState([]);

  useEffect(() => {
    const lower = search.toLowerCase();
    if (tab === "kodes") {
      const matches = Object.entries(kodes)
        .filter(
          ([key, val]) =>
            key.includes(lower) ||
            val.title?.toLowerCase().includes(lower) ||
            val.description?.toLowerCase().includes(lower)
        )
        .map(([key, val]) => ({ code: key, ...val }));
      setFilteredKodes(matches.slice(0, 30));
    } else if (tab === "companies") {
      const matches = extracted.filter(
        (comp) =>
          comp.name?.toLowerCase().includes(lower) ||
          comp.zweck?.toLowerCase().includes(lower) ||
          comp.code?.includes(lower) ||
          comp.addr?.toLowerCase().includes(lower)
      );
      setFilteredCompanies(matches.slice(0, 30));
    }
  }, [search, tab]);

  return (
    <div className="p-10 text-sm text-gray-800 bg-white shadow-lg rounded-md max-w-7xl mx-auto mt-10">
      {/* Top Tab Bar */}
      <div className="flex justify-between items-center border-b mb-4">
        <div className="space-x-6 font-semibold text-blue-700 text-base">
          <button onClick={() => setTab("overview")} className={tab === "overview" ? "underline" : ""}>Overview</button>
          <button onClick={() => setTab("kodes")} className={tab === "kodes" ? "underline" : ""}>WZ Codes</button>
          <button onClick={() => setTab("companies")} className={tab === "companies" ? "underline" : ""}>Companies</button>
          <button onClick={() => setTab("parameters")} className={tab === "parameters" ? "underline" : ""}>Parameters</button>
        </div>
        {tab !== "overview" && tab !== "parameters" && (
          <input
            type="text"
            className="border px-2 py-1 rounded w-72"
            placeholder={`Search ${tab}...`}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        )}
      </div>

      {/* Overview Tab */}
      {tab === "overview" && (
        <div className="space-y-5 leading-relaxed text-[15px]">
          <p>
            This interactive dashboard allows you to explore clusters of German company websites. The clustering is done using advanced NLP-based vectorization techniques like <strong>TF-IDF</strong>, <strong>SBERT</strong>, and <strong>Word2Vec</strong>, followed by clustering algorithms like <strong>K-Means</strong> or <strong>DBSCAN</strong>.
          </p>

          <p>
            <strong>Clustering</strong> enables unsupervised discovery of structure within data. Here, businesses with similar website descriptions naturally fall into groups, revealing patterns and semantic proximity in industry sectors.
          </p>

          <p>
            Each cluster is evaluated using the <em>Silhouette Score</em> and <em>Davies-Bouldin Index</em> to help you assess clustering quality.
          </p>

          <h3 className="font-semibold text-base mt-6">🛠 How to Use the Dashboard</h3>
          <ul className="list-disc list-inside space-y-2">
            <li><strong>Step 1:</strong> Select a feature method (TF-IDF, SBERT, or Word2Vec).</li>
            <li><strong>Step 2:</strong> Tune feature generation parameters like n-gram range or max features.</li>
            <li><strong>Step 3:</strong> Choose a clustering algorithm (K-Means or DBSCAN).</li>
            <li><strong>Step 4:</strong> Configure clustering parameters.</li>
            <li><strong>Step 5:</strong> Click <em>Generate & Cluster</em> to view visualization, scores, and summaries.</li>
          </ul>
        </div>
      )}

      {/* Parameters Tab */}
      {tab === "parameters" && (
        <div className="space-y-6 text-sm leading-relaxed">
          <h3 className="text-lg font-semibold">TF-IDF Parameters</h3>
          <ul className="list-disc list-inside space-y-1">
            <li><strong>max_features</strong>: Limits the number of features (i.e., most frequent terms) used in the TF-IDF vectorization.</li>
            <li><strong>min_df</strong>: Minimum number of documents a term must appear in to be included.</li>
            <li><strong>n_gram_range</strong>: Controls the size of n-grams (e.g., [1,1] for unigrams; [1,2] includes bigrams).</li>
          </ul>

          <h3 className="text-lg font-semibold">SBERT Parameters</h3>
          <ul className="list-disc list-inside space-y-1">
            <li><strong>dimensions</strong>: Number of dimensions to reduce embeddings to using UMAP. Smaller values simplify visualization but may reduce clustering quality.</li>
          </ul>

          <h3 className="text-lg font-semibold">Word2Vec Parameters</h3>
          <ul className="list-disc list-inside space-y-1">
            <li>Word2Vec uses a pretrained FastText model (German) and currently does not expose additional tunable parameters.</li>
          </ul>

          <h3 className="text-lg font-semibold">Clustering: K-Means</h3>
          <ul className="list-disc list-inside space-y-1">
            <li><strong>n_clusters</strong>: Number of clusters to divide data into.</li>
          </ul>

          <h3 className="text-lg font-semibold">Clustering: DBSCAN</h3>
          <ul className="list-disc list-inside space-y-1">
            <li><strong>eps</strong>: Maximum distance between two samples to be considered as in the same neighborhood.</li>
            <li><strong>min_samples</strong>: Minimum number of samples required to form a dense region (cluster).</li>
          </ul>
        </div>
      )}

      {/* WZ Code Search */}
      {tab === "kodes" && (
        <div className="space-y-2 max-h-[500px] overflow-y-auto">
          {filteredKodes.map((k) => (
            <div key={k.code} className="border rounded p-3 bg-gray-50">
              <h3 className="font-semibold text-sm mb-1">{k.code}: {k.title}</h3>
              <p className="text-gray-600 whitespace-pre-line text-xs">{k.description}</p>
            </div>
          ))}
          {filteredKodes.length === 0 && <p className="text-gray-500">No WZ codes matched.</p>}
        </div>
      )}

      {/* Company Search */}
      {tab === "companies" && (
        <div className="space-y-2 max-h-[500px] overflow-y-auto">
          {filteredCompanies.map((c) => (
            <div key={c.crefonummer} className="border rounded p-3 bg-gray-50">
              <h3 className="font-semibold text-sm">{c.name || "Unnamed Company"}</h3>
              <p className="text-xs"><strong>Website:</strong> {c.web || "N/A"}</p>
              <p className="text-xs"><strong>WZ Code:</strong> {c.code}</p>
              <p className="text-xs"><strong>Purpose:</strong> {c.zweck}</p>
              <p className="text-xs"><strong>Address:</strong> {c.addr}</p>
            </div>
          ))}
          {filteredCompanies.length === 0 && <p className="text-gray-500">No companies matched.</p>}
        </div>
      )}
    </div>
  );
}
