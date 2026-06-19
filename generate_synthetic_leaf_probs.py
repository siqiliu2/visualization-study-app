from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


DATA_DIR = Path(__file__).parent / "data"
TREE_TXT = DATA_DIR / "best_tree.txt"
OUT_JSONL = DATA_DIR / "synthetic_leaf_probs.jsonl"


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str          # "<=" or ">"
    threshold: float


@dataclass
class LeafRule:
    leaf_id: int
    label: int
    conditions: List[Condition]


COND_RE = re.compile(r"---\s+([A-Za-z_]\w*)\s+(<=|>)\s+(-?\d+(?:\.\d+)?)")
CLASS_RE = re.compile(r"---\s+class:\s+([01])(?:\.0)?")


def indent_depth(line: str) -> int:
    # sklearn export_text uses "|   " blocks. Count how many appear.
    return line.count("|   ")


def parse_best_tree_txt(path: Path) -> List[LeafRule]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    stack: List[Tuple[int, Condition]] = []   # (depth, condition)
    leaves: List[LeafRule] = []
    leaf_id = 0

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue

        # Skip "truncated branch..." lines (they are not leaves)
        if "truncated branch" in line:
            continue

        d = indent_depth(line)

        m_cond = COND_RE.search(line)
        if m_cond:
            cond = Condition(
                feature=m_cond.group(1),
                op=m_cond.group(2),
                threshold=float(m_cond.group(3)),
            )
            # Pop to parent depth
            while stack and stack[-1][0] >= d:
                stack.pop()
            stack.append((d, cond))
            continue

        m_class = CLASS_RE.search(line)
        if m_class:
            label = int(m_class.group(1))
            conditions = [c for _, c in stack]
            leaves.append(LeafRule(leaf_id=leaf_id, label=label, conditions=conditions))
            leaf_id += 1

    return leaves


def stable_jitter(key: str, lo: float = -0.05, hi: float = 0.05) -> float:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    u = int(h[:8], 16) / 0xFFFFFFFF
    return lo + (hi - lo) * u


def synth_prob(label: int, n: int, key: str) -> float:
    # Base separation
    base = 0.70 if label == 1 else 0.30
    # More conditions => a bit more confident (cap the effect)
    bump = min(n, 10) * 0.02
    p = base + (bump if label == 1 else -bump)
    p += stable_jitter(key)

    # Clamp
    p = max(0.01, min(0.99, p))
    return float(p)


def main():
    if not TREE_TXT.exists():
        raise FileNotFoundError(f"Missing {TREE_TXT}")

    leaves = parse_best_tree_txt(TREE_TXT)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for leaf in leaves:
            key = " & ".join([f"{c.feature} {c.op} {c.threshold}" for c in leaf.conditions]) + f" -> {leaf.label}"
            p = synth_prob(leaf.label, len(leaf.conditions), key)

            obj = {
                "leaf_id": leaf.leaf_id,
                "label": leaf.label,
                "p_positive": p,
                "n_conditions": len(leaf.conditions),
                "conditions": [
                    {"feature": c.feature, "op": c.op, "threshold": c.threshold}
                    for c in leaf.conditions
                ],
            }
            f.write(json.dumps(obj) + "\n")

    print(f"✅ Wrote {OUT_JSONL} with {len(leaves)} leaf rows")


if __name__ == "__main__":
    main()