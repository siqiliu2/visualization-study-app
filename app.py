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


def visualize_tree(file_path: Path, highlight: Optional[str] = None) -> io.BytesIO:
    nodes = load_nodes(file_path)

    dot = Digraph()
    dot.attr(
        size="4,4",
        dpi="260",
        rankdir="TB",
        splines="ortho",
        fontname="Inter",
        bgcolor="transparent"
    )

    dot.graph_attr.update({"margin": "0", "pad": "0"})
    dot.node_attr.update({"margin": "0.15,0.07"})
    dot.edge_attr.update({
        "color": "black",
        "fontcolor": "black",
        "penwidth": "1.2"
    })

    for node_id, node in nodes.items():
        if node["is_leaf"]:
            label = f"Leaf\np={node['p_positive']:.3f}"
        else:
            try:
                threshold = float(node["threshold"])
                label = f"{node['feature']} ≤ {threshold:.3f}\np={node['p_positive']:.3f}"
            except Exception:
                label = f"{node['feature']} ≤ {node['threshold']}\np={node['p_positive']:.3f}"

        feature = node["feature"]

        if node["is_leaf"]:
            fillcolor = probability_to_color(float(node["p_positive"]))
            fontcolor = "white"
        else:
            if highlight:
                fillcolor = feature_colors.get(feature, "#D0D0D0") if feature == highlight else "#E6E6E6"
            else:
                fillcolor = feature_colors.get(feature, "#D0D0D0")
            fontcolor = "black"

        shape = "ellipse" if node["is_leaf"] else "box"

        dot.node(
            str(node_id),
            label,
            style="filled,rounded",
            fillcolor=fillcolor,
            color="black",
            penwidth="1",
            shape=shape,
            fontname="Inter",
            fontcolor=fontcolor
        )

    for node_id, node in nodes.items():
        if not node["is_leaf"]:
            left_child = node.get("left_child")
            right_child = node.get("right_child")

            if left_child is not None:
                dot.edge(str(node_id), str(left_child))

            if right_child is not None:
                dot.edge(str(node_id), str(right_child))

    return io.BytesIO(dot.pipe(format="png"))


@app.route("/decision-tree")
def decision_tree_page():
    feature = request.args.get("feature")

    return render_template(
        "decision_tree.html",
        selected=feature,
        feature_colors=feature_colors,
        feature_explanations=feature_explanations
    )


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
    total = len(trees)

    def new_agg_node(feature: str, name: str) -> Dict[str, Any]:
        return {
            "feature": feature,
            "name": name,
            "freq": 0,
            "p_sum": 0.0,
            "p_count": 0,
            "children_map": {}
        }

    root = new_agg_node("ROOT", "Aggregate Sunburst")

    def merge(parent_agg: Dict[str, Any], node: Dict[str, Any]) -> None:
        feature, name = node_key(node)
        cmap: Dict[Tuple[str, str], Dict[str, Any]] = parent_agg["children_map"]

        if (feature, name) not in cmap:
            cmap[(feature, name)] = new_agg_node(feature, name)

        agg = cmap[(feature, name)]
        agg["freq"] += 1

        p = safe_float(node.get("p"), None)
        if p is not None:
            agg["p_sum"] += p
            agg["p_count"] += 1

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
        freq = int(node.get("freq", 0))
        p_count = int(node.get("p_count", 0))
        p_avg = (node["p_sum"] / p_count) if p_count > 0 else None

        out: Dict[str, Any] = {
            "feature": node["feature"],
            "name": node["name"],
            "freq": freq,
            "pct": (freq / total) if total > 0 else 0.0,
            "p": p_avg,
            "bin": bin_from_p(p_avg) if isinstance(p_avg, (int, float)) else "medium"
        }

        children_map = node.get("children_map", {})
        children_list = [finalize(v) for v in children_map.values()]
        children_list.sort(key=lambda x: x.get("freq", 0), reverse=True)

        if children_list:
            out["children"] = children_list

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
LEAF_META: Dict[int, Dict[str, Any]] = load_leaf_meta_jsonl(LEAF_META_PATH)


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


def leaf_outputs(leaf_id: int, fallback_label: int) -> Tuple[float, int, int]:
    meta = LEAF_META.get(int(leaf_id))

    if not meta:
        p = float(fallback_label)
        return p, int(1 if p >= 0.5 else 0), 0

    p = float(meta.get("p_positive", 0.5))
    n = int(meta.get("n_samples", 0))
    label = int(1 if p >= 0.5 else 0)

    return p, label, n


def predict_from_best_tree(features: Dict[str, float]) -> Dict[str, Any]:
    for rule in LEAF_RULES:
        ok, path, warn = match_rule(rule, features)

        if ok:
            p, label, n_samples = leaf_outputs(rule.leaf_id, rule.label)

            return {
                "ok": True,
                "p_positive": float(p),
                "label": int(label),
                "leaf_id": int(rule.leaf_id),
                "leaf_samples": int(n_samples),
                "leaf": {
                    "node_id": int(rule.leaf_id),
                    "n_samples": int(n_samples)
                },
                "path": path,
                "warning": None
            }

    best: Optional[LeafRule] = None
    best_score = -1
    best_path: Optional[List[Dict[str, Any]]] = None

    for rule in LEAF_RULES:
        score = 0
        path: List[Dict[str, Any]] = []
        violated = False

        for cond in rule.conditions:
            if cond.feature not in features:
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

            if ok:
                score += 1
            else:
                violated = True
                break

        if violated:
            continue

        if score > best_score or (
            score == best_score and best and len(rule.conditions) > len(best.conditions)
        ):
            best = rule
            best_score = score
            best_path = path

    if best is None:
        return {
            "ok": False,
            "error": "Could not match any rule from best_tree.txt"
        }

    p, label, n_samples = leaf_outputs(best.leaf_id, best.label)

    return {
        "ok": True,
        "p_positive": float(p),
        "label": int(label),
        "leaf_id": int(best.leaf_id),
        "leaf_samples": int(n_samples),
        "leaf": {
            "node_id": int(best.leaf_id),
            "n_samples": int(n_samples)
        },
        "path": best_path,
        "warning": "No exact leaf matched with provided values; showing best partial match based on available information."
    }


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