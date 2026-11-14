from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from graphviz import Digraph


app = Flask(__name__)

# =========================================================
# Shared paths
# =========================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Visualization 1: Decision tree image
PARTIAL_TREE_JSONL = DATA_DIR / "partial_tree.jsonl"

# Visualization 2: Sunburst multi-tree data
TREES_DIR = DATA_DIR / "sunburst_trees"

# Visualization 3: Dashboard prediction data
TREE_TXT = DATA_DIR / "best_tree.txt"
LEAF_META_PATH = DATA_DIR / "leaf_meta.jsonl"
TRAIN_PATH = DATA_DIR / "training_data.csv"


# =========================================================
# Shared feature metadata
# =========================================================

feature_colors = {
    "FAC": "#A1C4FD",
    "CSC": "#C2F0C2",
    "HCS": "#FDDC9B",
    "ECS": "#E6CCFF",
    "NPE": "#FFB6C1",
    "EDD": "#FFFACD",
    "LEAF": "#9aa4b2",
    "ROOT": "#9aa4b2"
}

feature_explanations = {
    "FAC": "Food access reflects food insecurity and the ability to obtain nutritious meals.",
    "CSC": "Social connectedness reflects social support, isolation, and ties to community networks.",
    "HCS": "Healthcare services reflects access to medical care and continuity of care.",
    "ECS": "Economic stability reflects income, employment, and financial security.",
    "NPE": "Neighborhood physical environment reflects housing, safety, and local conditions.",
    "EDD": "Education reflects educational attainment, literacy, and access to learning resources."
}


# =========================================================
# Home route
# =========================================================

@app.route("/")
def home():
    return render_template("home.html")


# =========================================================
# Visualization 1: Single Decision Tree
# =========================================================

def load_nodes(filepath: Path) -> Dict[int, Dict[str, Any]]:
    nodes: Dict[int, Dict[str, Any]] = {}

    if not filepath.exists():
        raise FileNotFoundError(f"Missing node file: {filepath}")

    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                nodes[int(data["node_id"])] = data
            except Exception:
                print("Skipping malformed line:", line)

    return nodes


def probability_to_color(p: float) -> str:
    p = max(0.0, min(1.0, p))

    if p <= 0.5:
        ratio = p / 0.5
        r = int(255 - (255 - 128) * ratio)
        g = 0
        b = int(0 + 128 * ratio)
    else:
        ratio = (p - 0.5) / 0.5
        r = int(128 * (1 - ratio))
        g = 0
        b = int(128 + (127 * ratio))

    return f"#{r:02x}{g:02x}{b:02x}"


def build_tree_graph(file_path: Path, highlight: Optional[str] = None) -> Digraph:
    """Render both legacy nodes and text-reconstructed nodes without inventing statistics."""
    import math

    nodes = load_nodes(file_path)
    if not nodes:
        raise ValueError(f"No tree nodes found in {file_path}")

    def probability(node):
        value = node.get("p_positive")
        if value is None or isinstance(value, bool):
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and 0 <= value <= 1 else None

    dot = Digraph()
    # The reconstructed tree has hundreds of nodes. Avoid shrinking it to 4x4 inches.
    dot.attr(dpi="120", rankdir="TB", splines="polyline",
             fontname="Inter", bgcolor="transparent")
    dot.graph_attr.update({"margin": "0", "pad": "0.1"})
    dot.node_attr.update({"margin": "0.15,0.07"})
    dot.edge_attr.update({"color": "black", "fontcolor": "black", "penwidth": "1.2"})

    for node_id, node in nodes.items():
        truncated = node.get("node_type") == "truncated" or node.get("is_truncated") is True
        leaf = not truncated and node.get("is_leaf") is True
        feature = node.get("feature")
        p = probability(node)
        p_text = f"p={p:.3f}" if p is not None else "Probability unavailable"
        fontcolor = "black"
        if truncated:
            label = "Subtree omitted\nClassification unavailable"
            depth = node.get("reported_subtree_depth")
            if depth is not None:
                label += f"\nReported subtree depth: {depth}"
            fillcolor, shape = "#FFF2CC", "ellipse"
        elif leaf:
            label = "Leaf"
            if node.get("predicted_class") is not None:
                label += f"\nClass: {node['predicted_class']}"
            label += "\n" + p_text
            fillcolor = probability_to_color(p) if p is not None else "#E6E6E6"
            fontcolor = "white" if p is not None else "black"
            shape = "ellipse"
        else:
            threshold = node.get("threshold")
            if threshold is None:
                raise ValueError(f"Split node {node_id} has no threshold")
            if node.get("threshold_precision") == "printed_only":
                threshold_text = node.get("threshold_text", str(threshold))
                label = f"{feature} ≤ {threshold_text} (rounded)"
            else:
                label = f"{feature} ≤ {float(threshold):.3f}"
            label += "\n" + p_text
            fillcolor = (feature_colors.get(feature, "#D0D0D0")
                         if not highlight or feature == highlight else "#E6E6E6")
            shape = "box"
        dot.node(str(node_id), label, style="filled,rounded", fillcolor=fillcolor,
                 color="black", penwidth="1", shape=shape,
                 fontname="Inter", fontcolor=fontcolor)

    omitted = set()
    for node_id, node in nodes.items():
        if (node.get("is_leaf") is True or node.get("node_type") == "truncated"
                or node.get("is_truncated") is True):
            continue
        for key, branch in (("left_child", "≤"), ("right_child", ">")):
            child = node.get(key)
            if child is None:
                continue
            # Legacy partial files may reference nodes absent from the file.
            if child not in nodes and child not in omitted:
                dot.node(str(child), f"Node {child}\nSubtree not included",
                         shape="ellipse", style="filled", fillcolor="#FFF2CC")
                omitted.add(child)
            dot.edge(str(node_id), str(child), label=branch)
    return dot


def visualize_tree(file_path: Path, highlight: Optional[str] = None) -> io.BytesIO:
    return io.BytesIO(build_tree_graph(file_path, highlight).pipe(format="png"))


@app.route("/decision-tree")
def decision_tree_page():
    feature = request.args.get("feature")

    return render_template(
        "decision_tree.html",
        selected=feature,
        feature_colors=feature_colors,
        feature_explanations=feature_explanations
    )


@app.route("/api/decision-tree")
def decision_tree_data():
    nodes = load_nodes(PARTIAL_TREE_JSONL)
    if not nodes:
        return jsonify({"error": "The tree file contains no nodes."}), 422
    child_ids = {n.get(k) for n in nodes.values()
                 for k in ("left_child", "right_child") if n.get(k) is not None}
    roots = [i for i in nodes if i not in child_ids]
    if len(roots) != 1:
        return jsonify({"error": "Expected one tree root."}), 422
    return jsonify({"root_id": roots[0], "nodes": list(nodes.values())})


@app.route("/tree_image")
def tree_image():
    feature = request.args.get("feature")
    stream = visualize_tree(PARTIAL_TREE_JSONL, highlight=feature)
    return send_file(stream, mimetype="image/png")


# =========================================================
# Visualization 2: Sunburst Multi-Tree Visualization
# =========================================================

def bin_from_p(p: float) -> str:
    if p < 0.33:
        return "low"
    elif p < 0.66:
        return "medium"
    return "high"


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def load_all_trees() -> List[Dict[str, Any]]:
    if not TREES_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {TREES_DIR}")

    trees: List[Dict[str, Any]] = []

    for fp in sorted(TREES_DIR.glob("*.json")):
        with fp.open("r", encoding="utf-8") as f:
            trees.append(json.load(f))

    if not trees:
        raise FileNotFoundError(f"No *.json trees found in {TREES_DIR}")

    return trees


def node_key(n: Dict[str, Any]) -> Tuple[str, str]:
    feature = str(n.get("feature", "UNKNOWN"))
    name = str(n.get("name", feature))
    return feature, name


def aggregate_trees(trees: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Preserve display paths; compute tooltip statistics across feature occurrences."""
    import math

    feature_stats: Dict[Tuple[str, Any], Dict[str, Any]] = {}

    def new_agg_node(feature: str, name: str) -> Dict[str, Any]:
        return {"feature": feature, "name": name, "children_map": {}}

    root = new_agg_node("ROOT", "Aggregate Sunburst")

    def merge(parent_agg: Dict[str, Any], node: Dict[str, Any]) -> None:
        feature, name = node_key(node)
        raw_p = node.get("p", node.get("p_positive"))
        p = None if isinstance(raw_p, bool) else safe_float(raw_p)
        if p is not None and (not math.isfinite(p) or not 0 <= p <= 1):
            p = None
        is_leaf = feature.upper() == "LEAF"
        outcome = None
        if is_leaf:
            outcome = "unknown" if p is None else ("high" if p >= 0.5 else "low")
            name = {"high": "High ED Utilization", "low": "Low ED Utilization",
                    "unknown": "ED Utilization Unknown"}[outcome]
        # Keep leaf outcomes separate even when they share a display parent.
        metric_key = ("leaf", outcome) if is_leaf else ("feature", feature)
        cmap = parent_agg["children_map"]
        agg = cmap.setdefault((feature, name), new_agg_node(feature, name))
        agg["metric_key"] = metric_key
        agg["outcome"] = outcome
        if is_leaf or feature.upper() not in {"ROOT", "UNKNOWN", "NONE", ""}:
            stats = feature_stats.setdefault(metric_key, {"freq": 0, "p_sum": 0.0, "p_count": 0})
            stats["freq"] += 1
            if p is not None:
                stats["p_sum"] += p
                stats["p_count"] += 1

        children = node.get("children", [])
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    merge(agg, child)

    for tree in trees:
        children = tree.get("children")
        if isinstance(children, list) and tree.get("feature") in (None, "ROOT", "root", "Root"):
            for child in children:
                if isinstance(child, dict):
                    merge(root, child)
        else:
            merge(root, tree)

    def finalize(node: Dict[str, Any]) -> Dict[str, Any]:
        stats = feature_stats.get(node.get("metric_key"))
        p_count = stats["p_count"] if stats else 0
        p_avg = stats["p_sum"] / p_count if p_count else None
        out: Dict[str, Any] = {
            "feature": node["feature"],
            "name": node["name"],
            "freq": stats["freq"] if stats else None,
            "p": p_avg,
            "p_count": p_count,
            "metric_scope": ("leaf_outcome_global" if node.get("outcome") else "feature_global") if stats else "structural",
            "outcome": node.get("outcome"),
            "bin": bin_from_p(p_avg) if p_avg is not None else "medium"
        }
        children = [finalize(v) for v in node["children_map"].values()]
        children.sort(key=lambda x: x["freq"] or 0, reverse=True)
        if children:
            out["children"] = children
        return out

    return finalize(root)


@app.route("/sunburst")
def sunburst_page():
    return render_template(
        "sunburst.html",
        feature_colors=feature_colors,
        feature_explanations=feature_explanations
    )


@app.route("/sunburst_data")
def sunburst_data():
    trees = load_all_trees()
    agg = aggregate_trees(trees)
    return jsonify(agg)


# =========================================================
# Visualization 3: Interactive Dashboard
# =========================================================

@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    threshold: float

    def is_satisfied(self, value: float) -> bool:
        if self.op == "<=":
            return value <= self.threshold
        return value > self.threshold


@dataclass
class LeafRule:
    leaf_id: int
    label: int
    conditions: List[Condition]


COND_RE = re.compile(r"---\s+([A-Za-z_]\w*)\s+(<=|>)\s+(-?\d+(?:\.\d+)?)")
CLASS_RE = re.compile(r"---\s+class:\s+([01])(?:\.0)?")


def indent_depth(line: str) -> int:
    return line.count("|   ")


def parse_best_tree_txt(path: Path) -> List[LeafRule]:
    if not path.exists():
        raise FileNotFoundError(f"Missing tree file: {path}")

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    stack: List[Tuple[int, Condition]] = []
    leaves: List[LeafRule] = []
    leaf_id = 0

    for raw in lines:
        line = raw.rstrip()

        if not line.strip():
            continue

        if "truncated branch" in line:
            continue

        depth = indent_depth(line)

        m_cond = COND_RE.search(line)
        if m_cond:
            cond = Condition(
                feature=m_cond.group(1),
                op=m_cond.group(2),
                threshold=float(m_cond.group(3))
            )

            while stack and stack[-1][0] >= depth:
                stack.pop()

            stack.append((depth, cond))
            continue

        m_class = CLASS_RE.search(line)
        if m_class:
            label = int(m_class.group(1))
            conditions = [c for _, c in stack]
            leaves.append(
                LeafRule(
                    leaf_id=leaf_id,
                    label=label,
                    conditions=conditions
                )
            )
            leaf_id += 1

    leaves.sort(key=lambda r: len(r.conditions), reverse=True)
    return leaves


def load_leaf_meta_jsonl(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Please generate or copy leaf_meta.jsonl into the data folder."
        )

    meta: Dict[int, Dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)
            leaf_id = int(obj["leaf_id"])

            meta[leaf_id] = {
                "leaf_id": leaf_id,
                "class_label": int(obj.get("class_label", obj.get("label", 0))),
                "n_samples": int(obj.get("n_samples", 0)),
                "p_positive": float(obj.get("p_positive", 0.5)),
                "conditions": obj.get("conditions", [])
            }

    return meta


LEAF_RULES: List[LeafRule] = parse_best_tree_txt(TREE_TXT)
# Dashboard inference reads the same node file as Visualization 1.
# Legacy leaf_meta.jsonl is no longer used for predictions.


def infer_feature_stats() -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}

    if TRAIN_PATH.exists():
        df = pd.read_csv(TRAIN_PATH)
        numeric_df = df.select_dtypes(include=["number"])

        for col in numeric_df.columns:
            series = numeric_df[col].dropna()

            if series.empty:
                continue

            stats[col] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean())
            }

        return stats

    thresh: Dict[str, List[float]] = {}

    for rule in LEAF_RULES:
        for cond in rule.conditions:
            thresh.setdefault(cond.feature, []).append(float(cond.threshold))

    for feat, vals in thresh.items():
        vals = sorted(vals)

        if not vals:
            continue

        stats[feat] = {
            "min": vals[0],
            "max": vals[-1],
            "mean": sum(vals) / len(vals)
        }

    return stats


FEATURE_STATS = infer_feature_stats()


def required_features() -> List[str]:
    feats = set()

    for rule in LEAF_RULES:
        for cond in rule.conditions:
            feats.add(cond.feature)

    return sorted(feats)


def match_rule(
    rule: LeafRule,
    features: Dict[str, float]
) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
    path: List[Dict[str, Any]] = []
    missing: List[str] = []

    for cond in rule.conditions:
        if cond.feature not in features:
            missing.append(cond.feature)
            path.append({
                "feature": cond.feature,
                "op": cond.op,
                "threshold": cond.threshold,
                "value": None,
                "satisfied": None,
                "decision": f"{cond.feature} {cond.op} {cond.threshold}"
            })
            continue

        value = float(features[cond.feature])
        ok = cond.is_satisfied(value)

        path.append({
            "feature": cond.feature,
            "op": cond.op,
            "threshold": cond.threshold,
            "value": value,
            "satisfied": ok,
            "decision": f"{cond.feature} {cond.op} {cond.threshold} (value={value})"
        })

        if not ok:
            return False, path, None

    if missing:
        return False, path, f"Missing feature(s) for this rule: {', '.join(sorted(set(missing)))}"

    return True, path, None


def predict_from_best_tree(features: Dict[str, float]) -> Dict[str, Any]:
    """Route the shared tree without substituting probabilities or partial matches."""
    import math
    nodes = load_nodes(PARTIAL_TREE_JSONL)
    child_ids = {n.get(k) for n in nodes.values() for k in ("left_child", "right_child")
                 if n.get(k) is not None}
    roots = [i for i in nodes if i not in child_ids]
    if len(roots) != 1:
        return {"ok": False, "error": "Expected exactly one tree root", "path": []}
    node_id, path, visited = roots[0], [], set()

    def probability(n):
        raw = n.get("p_positive")
        if raw is None or isinstance(raw, bool):
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and 0 <= value <= 1 else None

    while True:
        if node_id in visited or node_id not in nodes:
            return {"ok": False, "error": "Tree branch is missing or circular", "path": path}
        visited.add(node_id)
        n = nodes[node_id]
        source = n.get("probability_source", n.get("statistics_source", "unavailable"))
        if n.get("node_type") == "truncated" or n.get("is_truncated") is True:
            return {"ok": False, "error": "This path reaches an omitted subtree; its final classification and probability are unknown.",
                    "path": path, "stopped_node_id": node_id}
        p = probability(n)
        if n.get("is_leaf") is True:
            label = int(p >= 0.5) if p is not None else n.get("predicted_class")
            if label not in (0, 1):
                return {"ok": False, "error": "Leaf classification unavailable", "path": path}
            utilization = "High ED Utilization" if label == 1 else "Low ED Utilization"
            demo = source == "synthetic_demo"
            leaf = {"node_id": node_id, "n_samples": n.get("n_samples"),
                    "p_positive": p, "label": label, "utilization_label": utilization,
                    "probability_source": source}
            return {"ok": True, "p_positive": p, "label": label,
                    "utilization_label": utilization, "leaf_id": node_id,
                    "leaf_samples": n.get("n_samples"), "leaf": leaf,
                    "path": path, "probability_source": source, "classification_cutoff": 0.5,
                    "warning": "Demo probabilities; not fitted model estimates." if demo else None}
        feature = n.get("feature")
        raw = features.get(feature)
        if raw is None or isinstance(raw, bool):
            return {"ok": False, "error": f"Missing input for {feature}", "missing": [feature], "path": path}
        try:
            value, threshold = float(raw), float(n["threshold"])
        except (TypeError, ValueError, KeyError):
            return {"ok": False, "error": f"Invalid value or split for {feature}", "path": path}
        if not math.isfinite(value) or not math.isfinite(threshold):
            return {"ok": False, "error": f"Nonfinite value for {feature}", "path": path}
        left = value <= threshold
        op = "<=" if left else ">"
        path.append({"node_id": node_id, "feature": feature, "op": op,
                     "threshold": threshold, "value": value, "satisfied": True,
                     "branch": "left" if left else "right",
                     "decision": f"{feature} {op} {threshold} (value={value})",
                     "p_positive": p, "n_samples": n.get("n_samples"),
                     "impurity": n.get("impurity"), "probability_source": source})
        node_id = n.get("left_child" if left else "right_child")


@app.route("/dashboard")
def dashboard_page():
    return render_template(
        "dashboard.html",
        features=required_features(),
        feature_stats=FEATURE_STATS,
        feature_colors=feature_colors,
        feature_explanations=feature_explanations
    )


@app.route("/api/features")
def api_features():
    return jsonify({
        "features": required_features(),
        "stats": FEATURE_STATS
    })

@app.route("/focus-questions")
def focus_questions_page():
    return render_template("focus_questions.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True, silent=True) or {}
    features_in = payload.get("features", {})

    cleaned: Dict[str, float] = {}

    for k, v in features_in.items():
        try:
            cleaned[str(k)] = float(v)
        except Exception:
            continue

    result = predict_from_best_tree(cleaned)
    return jsonify(result)


# =========================================================
# Run app
# =========================================================

if __name__ == "__main__":
    app.run(debug=True, port=5000)