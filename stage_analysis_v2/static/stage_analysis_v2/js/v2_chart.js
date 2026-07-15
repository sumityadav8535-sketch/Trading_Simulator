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
            color: { up: "#34d399", down: "#f87171", unchanged: "#94a3b8" },
            borderColor: { up: "#34d399", down: "#f87171", unchanged: "#94a3b8" },
          },
          {
            type: "line",
            label: maLabel,
            data: maData,
            yAxisID: "y",
            borderColor: "#60a5fa",
            pointRadius: 0,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#cbd5e1", boxWidth: 12 } } },
        scales: {
          x: { type: "time", ticks: { color: "#94a3b8", maxTicksLimit: 6 }, grid: { color: "#334155" } },
          y: { position: "right", ticks: { color: "#94a3b8" }, grid: { color: "#334155" } },
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
          borderColor: "#a78bfa",
          backgroundColor: "rgba(167,139,250,0.1)",
          fill: true,
          pointRadius: 0,
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#cbd5e1" } } },
        scales: {
          x: { type: "time", ticks: { color: "#94a3b8", maxTicksLimit: 6 }, grid: { color: "#334155" } },
          y: { ticks: { color: "#a78bfa" }, grid: { color: "#334155" } },
        },
      },
    });
  }
})();