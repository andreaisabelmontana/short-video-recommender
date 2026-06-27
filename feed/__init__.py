"""feed -- an implicit-feedback recommender for an endless short-video feed.

Public API:
    generate, train_test_split, Interactions   (synthetic watch-signal data)
    ImplicitMF                                  (confidence-weighted ALS model)
    PopularityBaseline                          (non-personalised baseline)
    recall_at_k, ndcg_at_k, evaluate_model, evaluate_baseline
"""

from .data import Interactions, generate, train_test_split
from .evaluate import (
    PopularityBaseline,
    evaluate_baseline,
    evaluate_model,
    ndcg_at_k,
    recall_at_k,
)
from .model import ImplicitMF

__all__ = [
    "Interactions",
    "generate",
    "train_test_split",
    "ImplicitMF",
    "PopularityBaseline",
    "recall_at_k",
    "ndcg_at_k",
    "evaluate_model",
    "evaluate_baseline",
]
