"""Ranking metrics for the feed recommender: Recall@K and NDCG@K, plus a baseline.

The held-out set per user is their genuinely-enjoyed videos (high watch_ratio) that
the model never saw during training. A good recommender should surface those in the
top-K of its ranking over *unseen* videos.
"""

from __future__ import annotations

import numpy as np

from .model import ImplicitMF


def recall_at_k(ranked: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Fraction of relevant items that appear in the top-k of ``ranked``."""
    if len(relevant) == 0:
        return np.nan
    topk = set(ranked[:k].tolist())
    hits = sum(1 for r in relevant if r in topk)
    return hits / len(relevant)


def ndcg_at_k(ranked: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Normalised DCG@k with binary relevance."""
    if len(relevant) == 0:
        return np.nan
    rel_set = set(relevant.tolist())
    dcg = 0.0
    for i, vid in enumerate(ranked[:k]):
        if vid in rel_set:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


class PopularityBaseline:
    """Non-personalised baseline: rank every user by global video popularity.

    Popularity = number of (training) interactions a video received. This is the
    standard "is your model actually learning anything personal?" yardstick.
    """

    def __init__(self) -> None:
        self.order: np.ndarray | None = None
        self._seen: dict[int, set[int]] = {}

    def fit(self, video_idx: np.ndarray, user_idx: np.ndarray, n_videos: int) -> "PopularityBaseline":
        counts = np.bincount(video_idx, minlength=n_videos)
        self.order = np.argsort(-counts)
        self._seen = {}
        for u, v in zip(user_idx.tolist(), video_idx.tolist()):
            self._seen.setdefault(u, set()).add(v)
        return self

    def next_videos(self, user: int, k: int = 10) -> np.ndarray:
        assert self.order is not None
        seen = self._seen.get(user, set())
        out = [v for v in self.order if v not in seen]
        return np.array(out[:k])


def evaluate_model(
    model: ImplicitMF, heldout: dict[int, np.ndarray], k: int = 10
) -> dict[str, float]:
    """Mean Recall@k and NDCG@k for a fitted ImplicitMF over the held-out users."""
    recalls, ndcgs = [], []
    for user, relevant in heldout.items():
        ranked = model.next_videos(user, k=k)
        recalls.append(recall_at_k(ranked, relevant, k))
        ndcgs.append(ndcg_at_k(ranked, relevant, k))
    return {
        f"recall@{k}": float(np.nanmean(recalls)),
        f"ndcg@{k}": float(np.nanmean(ndcgs)),
        "n_users": len(heldout),
    }


def evaluate_baseline(
    baseline: PopularityBaseline, heldout: dict[int, np.ndarray], k: int = 10
) -> dict[str, float]:
    """Mean Recall@k and NDCG@k for the popularity baseline over held-out users."""
    recalls, ndcgs = [], []
    for user, relevant in heldout.items():
        ranked = baseline.next_videos(user, k=k)
        recalls.append(recall_at_k(ranked, relevant, k))
        ndcgs.append(ndcg_at_k(ranked, relevant, k))
    return {
        f"recall@{k}": float(np.nanmean(recalls)),
        f"ndcg@{k}": float(np.nanmean(ndcgs)),
        "n_users": len(heldout),
    }
