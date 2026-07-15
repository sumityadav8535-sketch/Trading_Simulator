(function () {
  const payload = window.__stageChartPayload;
  const canvas = document.getElementById("stage-chart");
  if (!canvas || !payload || typeof Chart === "undefined") return;

  const stageLabels = {
    1: "Stage 1 — Accumulation",
    2: "Stage 2 — Advancing",
    3: "Stage 3 — Distribution",
    4: "Stage 4 — Declining",
  };

  const candleData = payload.ohlc.map((bar) => ({
    x: bar.x,
    o: bar.o,
    h: bar.h,
    l: bar.l,
    c: bar.c,
  }));

  const maData = payload.labels
    .map((label, i) => ({ x: label, y: payload.ma_30w[i] }))
    .filter((p) => p.y !== null);

  const volumeData = payload.labels.map((label, i) => ({
    x: label,
    y: payload.volume[i],
  }));

  const stageColor = payload.stage_color || "#34d399";

  new Chart(canvas.getContext("2d"), {
    data: {
      datasets: [
        {
          type: "candlestick",
          label: "Weekly OHLC",
          data: candleData,
          yAxisID: "y",
          color: { up: "#34d399", down: "#f87171", unchanged: "#94a3b8" },
          borderColor: { up: "#34d399", down: "#f87171", unchanged: "#94a3b8" },
        },
        {
          type: "line",
          label: "30-Week MA",
          data: maData,
          yAxisID: "y",
          borderColor: "#60a5fa",
          backgroundColor: "transparent",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
        },
        {
          type: "bar",
          label: "Volume",
          data: volumeData,
          yAxisID: "yVolume",
          backgroundColor: "rgba(148, 163, 184, 0.25)",
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#cbd5e1" } },
        title: {
          display: true,
          text: stageLabels[payload.stage] || "Stage Analysis",
          color: stageColor,
          font: { size: 14, weight: "bold" },
        },
        tooltip: {
          callbacks: {
            label(ctx) {
              if (ctx.raw && ctx.raw.o !== undefined) {
                const r = ctx.raw;
                return `O:${r.o} H:${r.h} L:${r.l} C:${r.c}`;
              }
              return `${ctx.dataset.label}: ${ctx.parsed.y}`;
            },
          },
        },
      },
      scales: {
        x: {
          type: "time",
          time: { unit: "month", tooltipFormat: "MMM yyyy" },
          ticks: { color: "#94a3b8", maxTicksLimit: 10 },
          grid: { color: "rgba(51, 65, 85, 0.5)" },
        },
        y: {
          position: "right",
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(51, 65, 85, 0.5)" },
        },
        yVolume: {
          position: "left",
          display: false,
          grid: { display: false },
        },
      },
    },
  });
})();