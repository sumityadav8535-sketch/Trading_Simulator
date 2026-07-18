(function () {
  const p = window.__v2ChartPayload;
  if (!p || typeof Chart === "undefined") return;

  function buildPriceChart(canvasId, data, ma, maLabel, title) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !data || !data.ohlc) return;

    const candleData = data.ohlc.map((b) => ({ x: b.x, o: b.o, h: b.h, l: b.l, c: b.c }));
    const maData = data.ohlc.map((b, i) => ({ x: b.x, y: ma[i] })).filter((d) => d.y != null);

    new Chart(canvas.getContext("2d"), {
      data: {
        datasets: [
          {
            type: "candlestick",
            label: title,
            data: candleData,
            yAxisID: "y",
            color: { up: "#059669", down: "#dc2626", unchanged: "#94a3b8" },
            borderColor: { up: "#059669", down: "#dc2626", unchanged: "#94a3b8" },
          },
          {
            type: "line",
            label: maLabel,
            data: maData,
            yAxisID: "y",
            borderColor: "#2563eb",
            pointRadius: 0,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#475569", boxWidth: 12 } } },
        scales: {
          x: { type: "time", ticks: { color: "#64748b", maxTicksLimit: 6 }, grid: { color: "#e2e8f0" } },
          y: { position: "right", ticks: { color: "#64748b" }, grid: { color: "#e2e8f0" } },
        },
      },
    });
  }

  if (p.weekly) {
    buildPriceChart("v2-chart-weekly", p.weekly, p.weekly.ma, "30W MA", "Weekly");
  } else if (p.ohlc) {
    buildPriceChart("v2-chart-weekly", { ohlc: p.ohlc }, p.ma_30w, "30W MA", "Weekly");
  }

  if (p.daily) {
    buildPriceChart("v2-chart-daily", p.daily, p.daily.ma, "150D MA", "Daily");
  }

  const rsCanvas = document.getElementById("v2-chart-rs");
  if (rsCanvas && p.rs_line && p.rs_line.length) {
    new Chart(rsCanvas.getContext("2d"), {
      type: "line",
      data: {
        datasets: [{
          label: "RS vs Nifty 50",
          data: p.rs_line.map((d) => ({ x: d.x, y: d.y })),
          borderColor: "#7c3aed",
          backgroundColor: "rgba(124,58,237,0.08)",
          fill: true,
          pointRadius: 0,
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#475569" } } },
        scales: {
          x: { type: "time", ticks: { color: "#64748b", maxTicksLimit: 6 }, grid: { color: "#e2e8f0" } },
          y: { ticks: { color: "#7c3aed" }, grid: { color: "#e2e8f0" } },
        },
      },
    });
  }
})();