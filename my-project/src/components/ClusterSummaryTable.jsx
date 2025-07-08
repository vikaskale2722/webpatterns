export default function ClusterSummaryTable({ summary }) {
  if (!Array.isArray(summary) || summary.length === 0) {
    return <p className="text-sm text-gray-500">No cluster summary available.</p>;
  }

  return (
    <div className="mt-10">
      <h2 className="text-xl font-semibold text-gray-800 mb-3">Cluster Summary</h2>
      
      <div className="overflow-x-auto border rounded bg-white shadow-sm max-h-[300px] overflow-y-auto">
        <table className="w-full text-sm text-left text-gray-700 border-collapse">
          <thead className="bg-gray-100 text-gray-800 font-semibold sticky top-0 z-10">
            <tr>
              <th className="px-4 py-2">Cluster</th>
              <th className="px-4 py-2">Size</th>
              <th className="px-4 py-2">Top WZ Code</th>
            </tr>
          </thead>
          <tbody>
            {summary.map((row) => (
              <tr key={row.cluster} className="border-t hover:bg-gray-50">
                <td className="px-4 py-2">{row.cluster}</td>
                <td className="px-4 py-2">{row.size}</td>
                <td className="px-4 py-2">{row.top_wz_code}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
