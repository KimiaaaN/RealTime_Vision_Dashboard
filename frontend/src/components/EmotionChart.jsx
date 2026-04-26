import React, { useMemo } from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import { Doughnut } from "react-chartjs-2";
ChartJS.register(ArcElement, Tooltip, Legend);

export default function EmotionChart({ events }) {
  const counts = useMemo(() => {
    const map = {};
    events?.slice(0, 200).forEach((evt) => {
      // Handle both old format (evt.faces array) and new format (direct face object)
      if (evt.faces && Array.isArray(evt.faces)) {
        evt.faces.forEach((f) => {
          const e = f.emotion || "unknown";
          map[e] = (map[e] || 0) + 1;
        });
      } else if (evt.emotion) {
        // New format: event is a face object directly
        const e = evt.emotion || "unknown";
        map[e] = (map[e] || 0) + 1;
      }
    });
    return map;
  }, [events]);

  const labels = Object.keys(counts);
  const data = {
    labels,
    datasets: [
      {
        data: labels.map((l) => counts[l]),
        backgroundColor: [
          "#FF6384",
          "#36A2EB",
          "#FFCE56",
          "#4BC0C0",
          "#9966FF",
          "#FF9F40",
        ],
      },
    ],
  };

  return (
    <div className="chart">
      <h3>Emotion Distribution</h3>
      {labels.length ? <Doughnut data={data} /> : <p>No data yet</p>}
    </div>
  );
}
