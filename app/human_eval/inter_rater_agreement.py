from __future__ import annotations

from collections import defaultdict
from itertools import combinations


def cohens_kappa(r1: list, r2: list, weighted: bool = False) -> float:
    """Cohen's kappa for two raters. weighted=True uses quadratic weights (ordinal, numeric ratings)."""
    if len(r1) != len(r2) or not r1:
        raise ValueError("Rating lists must be non-empty and the same length")

    cats = sorted(set(r1) | set(r2))
    k, n = len(cats), len(r1)
    idx = {c: i for i, c in enumerate(cats)}
    obs = [[0] * k for _ in range(k)]
    for a, b in zip(r1, r2):
        obs[idx[a]][idx[b]] += 1
    rows = [sum(r) for r in obs]
    cols = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    span = (cats[-1] - cats[0]) if weighted and k > 1 else 0
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            if weighted:
                w = ((cats[i] - cats[j]) / span) ** 2 if span else 0.0
            else:
                w = 0.0 if i == j else 1.0
            num += w * obs[i][j] / n
            den += w * rows[i] * cols[j] / (n * n)
    return 1.0 if den == 0 else 1.0 - num / den


def rater_agreement(ratings: list[dict]) -> dict[tuple[str, str], dict]:
    """ratings: [{'rater':..., 'case_id':..., 'rating': int}, ...] -> pairwise kappa on shared cases."""
    by_rater: dict[str, dict[str, int]] = defaultdict(dict)
    for r in ratings:
        by_rater[r["rater"]][r["case_id"]] = r["rating"]

    out = {}
    for a, b in combinations(sorted(by_rater), 2):
        common = sorted(set(by_rater[a]) & set(by_rater[b]))
        if len(common) < 2:
            continue
        ra, rb = [by_rater[a][c] for c in common], [by_rater[b][c] for c in common]
        out[(a, b)] = {
            "n": len(common),
            "kappa": cohens_kappa(ra, rb),
            "weighted_kappa": cohens_kappa(ra, rb, weighted=True),
        }
    return out