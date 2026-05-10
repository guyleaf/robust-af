function normalizeToArray(val, n, defaultVal) {
  if (Array.isArray(val)) {
    if (val.length > 1) {
      return val.slice();
    }
    val = val[0];
  }
  return new Array(n).fill(val !== undefined ? val : defaultVal);
}

function adjustAlpha(color, alpha) {
  let r = parseInt(color.slice(1, 3), 16);
  let g = parseInt(color.slice(3, 5), 16);
  let b = parseInt(color.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function animate(plot, data, traces) {
  Plotly.animate(
    plot,
    {
      data: data,
      traces: traces,
    },
    {
      transition: { duration: 0 }, // 不要動畫,立即生效
      frame: { duration: 0, redraw: false },
    },
  );
}

let plot = document.getElementById("{plot_id}");
console.log(plot.data);

// constant arrays
let HOVERED_OPACITIES = plot.data.map((t) =>
  t.customdata ? new Array(t.x.length).fill(0.05) : [1.0],
);
let TRACE_INDICES = plot.data.map((_, i) => i);

// create index for image_id to accelrate hovering
let idIndex = new Map(); // image_id -> [{traceIdx, ptIdx}, ...]
plot.data.forEach((trace, traceIdx) => {
  if (!trace.customdata) return;
  trace.customdata.forEach((item, ptIdx) => {
    const id = item[0];
    if (!idIndex.has(id)) idIndex.set(id, []);
    idIndex.get(id).push({ traceIdx, ptIdx });
  });
});

let originalState = {
  opacity: plot.data.map((t) =>
    normalizeToArray(t.marker.opacity, t.x.length, 1.0),
  ),
  // color: plot.data.map((t) =>
  //   normalizeToArray(t.marker.color, t.x.length, "#FFFFFF"),
  // ),
};
console.log(originalState);

plot.on("plotly_hover", function (data) {
  let hoveredId = data.points[0].customdata[0];
  let newOpacities = structuredClone(HOVERED_OPACITIES);

  let matches = idIndex.get(hoveredId) || [];
  matches.forEach(({ traceIdx, ptIdx }) => {
    newOpacities[traceIdx][ptIdx] = originalState.opacity[traceIdx][ptIdx];
  });
  // Plotly.restyle(plot, { "marker.opacity": newOpacities });

  let newData = plot.data.map((_, i) => ({
    marker: { opacity: newOpacities[i] },
  }));
  animate(plot, newData, TRACE_INDICES);
});

plot.on("plotly_unhover", function (data) {
  // 還原
  // Plotly.restyle(plot, { "marker.opacity": originalState.opacity });

  let newData = plot.data.map((_, i) => ({
    marker: { opacity: originalState.opacity[i] },
  }));
  animate(plot, newData, TRACE_INDICES);
});
