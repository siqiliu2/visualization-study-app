// Thickness exaggeration: p in [0,1] -> ring thickness
const thicknessScale = d3.scaleLinear()
  .domain([0, 1])
  .range([10, 70]);

// Feature palette (match your tree project)
const featureColors = {
  "FAC": "#A1C4FD",
  "CSC": "#C2F0C2",
  "HCS": "#FDDC9B",
  "ECS": "#E6CCFF",
  "NPE": "#FFB6C1",
  "EDD": "#FFFACD",
  "LEAF": "#9aa4b2"
};

const width = 820;
const radius = width / 2;
const HOVER_EXPAND = 18;

fetch("/sunburst_data")
  .then(r => r.json())
  .then(data => drawSunburst(data))
  .catch(err => console.error("Failed to load sunburst data:", err));

function computeRadii(root) {
  root.each(d => {
    if (!d.parent) d.inner = 0;
    else d.inner = d.parent.outer;
    d.outer = d.inner + thicknessScale(d.data.p || 0);
  });
}

function drawSunburst(data) {
  const tooltip = d3.select("#tooltip");

  // --- NEW: feature definitions for sidebar ---
  const featureLabels = {
    "FAC": "Food access reflects food insecurity and the ability to obtain nutritious meals.",
    "CSC": "Social connectedness reflects social support, isolation, and ties to community networks.",
    "HCS": "Healthcare services reflects access to medical care and continuity of care.",
    "ECS": "Economic stability reflects income, employment, and financial security.",
    "NPE": "Neighborhood physical environment reflects housing, safety, and local conditions.",
    "EDD": "Education reflects educational attainment, literacy, and access to learning resources."
  };
  const sidebarFeatures = ["FAC","CSC","HCS","ECS","NPE","EDD"];

  // --- NEW: selection state ---
  let selectedFeature = null;

  // --- NEW: richer/dimmer color function (keeps your palette) ---
  function colorForNode(d){
    const feat = d.data?.feature;
    const base = featureColors[feat] || "#6b7280";

    // If nothing selected, normal colors
    if (!selectedFeature) return base;

    // You can decide how LEAF behaves. Here: dim leaf when not selected.
    if (feat === selectedFeature) {
      const c = d3.hsl(base);
      c.s = Math.min(1, c.s * 1.8);   // richer saturation
      c.l = Math.max(0, c.l * 0.95);  // slightly darker
      return c.formatHex();
    } else {
      const c = d3.hsl(base);
      c.s = c.s * 0.25;              // desaturate
      c.l = c.l * 0.85;              // dim
      return c.formatHex();
    }
  }

  // --- NEW: sidebar rendering + click to toggle ---
  function renderSidebar() {
    const list = d3.select("#featureList");

    const items = list
      .selectAll(".feature-item")
      .data(sidebarFeatures, d => d);

    const enter = items
      .enter()
      .append("div")
      .attr("class", "feature-item")
      .on("click", (event, feat) => {
        selectedFeature = selectedFeature === feat ? null : feat;
        renderSidebar();
        updateHighlight();
      });

    const header = enter
      .append("div")
      .attr("class", "feature-header-row");

    header
      .append("div")
      .attr("class", "swatch")
      .style("background", feat => featureColors[feat] || "#999");

    header
      .append("div")
      .attr("class", "feature-code")
      .text(feat => feat);

    enter
      .append("div")
      .attr("class", "feature-desc")
      .text(feat => featureLabels[feat] || "");

    items
      .merge(enter)
      .classed("active", feat => feat === selectedFeature);

    items.exit().remove();

    d3.select("#clearFeature").on("click", () => {
      selectedFeature = null;
      renderSidebar();
      updateHighlight();
    });
  }

  // --- your existing hierarchy + custom radii logic ---
  const root = d3.hierarchy(data).sum(d => 1);

  d3.partition().size([2 * Math.PI, radius])(root);
  computeRadii(root);
  
  const maxOuterRadius =
    d3.max(root.descendants(), d => d.outer) + HOVER_EXPAND;

  const VIEW_PADDING = 10;
  const VIEW_SIZE = (maxOuterRadius + VIEW_PADDING) * 2;
  const VERTICAL_SHIFT = 35;

  
  const VERTICAL_OFFSET = 35;

  const svgElement = d3.select("#chart")
    .append("svg")
    .attr(
      "viewBox",
      `
        ${radius - VIEW_SIZE / 2}
        ${radius - VIEW_SIZE / 2 + VERTICAL_SHIFT}
        ${VIEW_SIZE}
        ${VIEW_SIZE}
      `
    )
    .attr("preserveAspectRatio", "xMidYMid meet")
    .style("width", "100%")
    .style("height", "100%")
    .style("display", "block")
    .style("overflow", "visible");

  const svg = svgElement
    .append("g")
    .attr("transform", `translate(${radius},${radius})`);


  const arc = d3.arc()
    .startAngle(d => d.x0)
    .endAngle(d => d.x1)
    .innerRadius(d => d.inner)
    .outerRadius(d => d.outer + (d.expand ? HOVER_EXPAND : 0));

  // --- IMPORTANT: keep a handle to all paths so we can update on click ---
  const paths = svg.selectAll("path")
    .data(root.descendants())
    .enter()
    .append("path")
      .attr("display", d => d.depth ? null : "none")
      .attr("d", arc)
      .style("stroke", "none")
      .style("fill", d => colorForNode(d))   // ✅ changed
      .style("opacity", d => {
        // If selected, dim non-selected further (optional)
        if (selectedFeature && d.data?.feature !== selectedFeature) return 0.45;
        return 0.95;
      })
      .on("mouseover", function(event, d) {
        d.expand = true;
        d3.select(this).transition().duration(160).attr("d", arc);

        tooltip
          .style("opacity", 1)
          .html(`
            <b>${escapeHtml(d.data.name || "")}</b><br/>
            node_id: ${d.data.node_id}<br/>
            feature: ${escapeHtml(d.data.feature || "")}<br/>
            p(risk=1): ${(d.data.p ?? 0).toFixed(3)}
          `);
      })
      .on("mousemove", function(event) {
        tooltip
          .style("left", (event.pageX + 12) + "px")
          .style("top", (event.pageY + 12) + "px");
      })
      .on("mouseout", function(event, d) {
        d.expand = false;
        d3.select(this).transition().duration(160).attr("d", arc);
        tooltip.style("opacity", 0);
      });

  // --- NEW: update sunburst styling when feature toggles ---
  function updateHighlight(){
    paths.transition()
      .duration(220)
      .style("fill", d => colorForNode(d))
      .style("opacity", d => {
        if (!selectedFeature) return 0.95;
        return (d.data?.feature === selectedFeature) ? 0.98 : 0.45;
      });
  }

  // initial render
  renderSidebar();
}

// Tiny safety helper for tooltip text
function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}