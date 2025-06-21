import Plot from "react-plotly.js";

export default function ClusterPlot({ umapCoords, labels, metadata, clusterSummary }) {
  const valid =
    Array.isArray(umapCoords) &&
    Array.isArray(labels) &&
    Array.isArray(metadata) &&
    umapCoords.length === labels.length &&
    umapCoords.length === metadata.length;

  if (!valid) {
    return (
      <div className="h-80 flex items-center justify-center text-gray-400 border rounded bg-gray-100">
        UMAP data not available or mismatched.
      </div>
    );
  }

  const labelMap = {};
  clusterSummary?.forEach((entry) => {
    labelMap[String(entry.cluster)] = entry.label;
  });

  const trace = {
    x: umapCoords.map((p) => p[0]),
    y: umapCoords.map((p) => p[1]),
    mode: "markers",
    type: "scatter",
    marker: {
      size: 8,
      color: labels,
      colorscale: "Viridis",
      showscale: true,
    },
    text: metadata.map((m, i) => {
      const clusterLabel = labelMap[String(labels[i])] || `Cluster ${labels[i]}`;
      const zweck = m.zweck?.length > 200 ? m.zweck.slice(0, 200) + "..." : m.zweck || "N/A";
      return `
        <b>Website:</b> ${m.web || "?"}<br>
        <b>WZ Code:</b> ${m.wz_code || "?"}<br>
        <b>Cluster:</b> ${clusterLabel}<br>
        <b>Zweck:</b> ${zweck}
      `;
    }),
    hoverinfo: "text",
    hovertemplate: '%{text}<extra></extra>',
  };

  return (
    <div className="relative w-full h-[500px] overflow-visible">
      <Plot
        data={[trace]}
        layout={{
          title: "UMAP Cluster Visualization",
          autosize: true,
          height: 500,
          margin: { l: 40, r: 20, t: 40, b: 40 },
          xaxis: { title: "UMAP-1" },
          yaxis: { title: "UMAP-2" },
          hoverlabel: {
            bgcolor: "#ffffff",
            bordercolor: "#dddddd",
            font: { size: 12, color: "#333333" },
          },
        }}
        style={{ width: "100%", height: "100%", pointerEvents: "auto" }}
        useResizeHandler={true}
      />
    </div>
  );
}
