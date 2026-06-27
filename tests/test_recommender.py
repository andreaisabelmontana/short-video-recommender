"""Tests for the implicit-feedback feed recommender.

These check the four properties that actually matter:
  1. the MF recovers a known low-rank preference matrix within tolerance;
  2. confidence weighting ranks high-watch-ratio items above low-watch-ratio ones;
  3. Recall@K and NDCG@K beat the popularity baseline on planted-structure data;
  4. an online watch update moves the matching topic's items up the ranking.
"""

import numpy as np
import pytest

from feed import (
    ImplicitMF,
    PopularityBaseline,
    evaluate_baseline,
    evaluate_model,
    generate,
    ndcg_at_k,
    recall_at_k,
    train_test_split,
)


# ---------------------------------------------------------------- 1. recovery
def test_mf_reconstructs_low_rank_preference():
    """ALS should reconstruct a known rank-2 preference matrix within tolerance.

    Build a genuine rank-2 binary preference matrix, expose every positive with a
    high watch_ratio, then check the model's predicted scores separate positives
    from negatives cleanly (high AUC + a margin between the two populations).
    """
    rng = np.random.default_rng(1)
    n_users, n_videos, rank = 40, 60, 2
    U = rng.random((n_users, rank))
    V = rng.random((n_videos, rank))
    P = (U @ V.T > 0.55).astype(float)  # known low-rank-ish binary preference

    u_idx, v_idx, wr = [], [], []
    for u in range(n_users):
        for v in range(n_videos):
            if P[u, v] > 0:
                u_idx.append(u)
                v_idx.append(v)
                wr.append(0.95)  # watched almost fully
    u_idx = np.array(u_idx)
    v_idx = np.array(v_idx)
    wr = np.array(wr)

    model = ImplicitMF(factors=8, alpha=40, regularization=0.05, iterations=20, seed=0)
    model.fit(u_idx, v_idx, wr, n_users, n_videos)

    scores = np.array([model.predict(u) for u in range(n_users)])
    pos = scores[P > 0]
    neg = scores[P == 0]
    # positives should score clearly higher than negatives on average
    assert pos.mean() > neg.mean() + 0.3
    # and the ranking should be near-perfect: AUC well above chance
    auc = _auc(pos, neg)
    assert auc > 0.95


def _auc(pos, neg):
    """Probability a random positive outranks a random negative (subsampled)."""
    rng = np.random.default_rng(0)
    a = rng.choice(pos, size=min(2000, len(pos)))
    b = rng.choice(neg, size=min(2000, len(neg)))
    return float((a[:, None] > b[None, :]).mean())


# ------------------------------------------------- 2. confidence weighting
def test_confidence_weighting_ranks_high_watch_above_low():
    """A video watched to completion should outrank one barely watched.

    One user, two otherwise-identical videos. The user watches video A almost fully
    (watch_ratio 0.95) and video B barely (0.05). After fit, A must score above B,
    and the confidence entry for A must exceed that for B.
    """
    n_users, n_videos = 1, 2
    u_idx = np.array([0, 0])
    v_idx = np.array([0, 1])
    wr = np.array([0.95, 0.05])

    model = ImplicitMF(factors=4, alpha=40, regularization=0.01, iterations=30, seed=0)
    model.fit(u_idx, v_idx, wr, n_users, n_videos)

    score_a = model.predict(0, np.array([0]))[0]
    score_b = model.predict(0, np.array([1]))[0]
    assert score_a > score_b

    # confidence excess (alpha * watch_ratio) must reflect the watch signal
    C = model.build_confidence(u_idx, v_idx, wr, n_users, n_videos, alpha=40)
    assert C[0, 0] > C[0, 1]


# ------------------------------------------------ 3. beats popularity baseline
def test_personalized_beats_popularity_on_planted_structure():
    """On planted-interest data, MF must beat the popularity baseline at Recall/NDCG."""
    inter = generate(n_users=300, n_videos=400, n_topics=6, density=0.15, seed=7)
    train, heldout = train_test_split(inter, test_frac=0.2, seed=7)
    assert len(heldout) > 100  # enough users to be meaningful

    model = ImplicitMF(factors=32, alpha=40, regularization=0.1, iterations=15, seed=0)
    model.fit(
        train.user_idx, train.video_idx, train.watch_ratio,
        train.n_users, train.n_videos,
    )
    base = PopularityBaseline().fit(train.video_idx, train.user_idx, train.n_videos)

    m = evaluate_model(model, heldout, k=10)
    b = evaluate_baseline(base, heldout, k=10)

    assert m["recall@10"] > b["recall@10"]
    assert m["ndcg@10"] > b["ndcg@10"]
    # personalised recall should be a clear win, not a coin-flip margin
    assert m["recall@10"] > b["recall@10"] * 1.3


# ----------------------------------------------------- 4. online update moves up
def test_online_update_moves_matching_topic_up():
    """Watching a topic-T video should raise other topic-T videos in the next ranking.

    Construct a clean two-topic world: videos 0..9 are topic-A, 10..19 are topic-B.
    A user who has only watched a couple of topic-A videos then watches a topic-B
    video to completion. The mean rank of the remaining topic-B videos must improve.
    """
    n_users, n_videos = 1, 20
    topic_a = np.arange(0, 10)
    topic_b = np.arange(10, 20)

    # seed: user watched two topic-A videos fully, nothing of topic-B
    u_idx = np.array([0, 0])
    v_idx = np.array([topic_a[0], topic_a[1]])
    wr = np.array([0.95, 0.9])

    # To give the items meaningful factors we co-train with a couple of helper users
    # who establish the two topic clusters (videos within a topic co-occur).
    helper_u, helper_v, helper_wr = [], [], []
    uid = 1
    for cluster in (topic_a, topic_b):
        for _ in range(8):
            vids = cluster
            helper_u.extend([uid] * len(vids))
            helper_v.extend(vids.tolist())
            helper_wr.extend([0.9] * len(vids))
            uid += 1
    n_users = uid
    u_all = np.concatenate([u_idx, np.array(helper_u)])
    v_all = np.concatenate([v_idx, np.array(helper_v)])
    wr_all = np.concatenate([wr, np.array(helper_wr)])

    model = ImplicitMF(factors=8, alpha=40, regularization=0.05, iterations=25, seed=0)
    model.fit(u_all, v_all, wr_all, n_users, n_videos)

    def mean_rank_of_topic_b():
        scores = model.predict(0).copy()
        scores[model.seen_videos(0)] = -np.inf
        order = np.argsort(-scores)
        rank = {v: i for i, v in enumerate(order)}
        unseen_b = [v for v in topic_b if v not in set(model.seen_videos(0))]
        return np.mean([rank[v] for v in unseen_b])

    before = mean_rank_of_topic_b()
    # the user now watches a topic-B video to completion
    model.online_update(user=0, video=int(topic_b[0]), watch_ratio=0.97)
    after = mean_rank_of_topic_b()

    # lower mean rank == higher up the list
    assert after < before


# ----------------------------------------------------------- metric sanity
def test_metric_helpers():
    ranked = np.array([3, 1, 4, 0, 2])
    relevant = np.array([4, 0])
    assert recall_at_k(ranked, relevant, k=4) == 1.0
    assert recall_at_k(ranked, relevant, k=2) == 0.0
    assert 0.0 < ndcg_at_k(ranked, relevant, k=5) <= 1.0
    # perfect ranking -> ndcg 1.0
    assert ndcg_at_k(np.array([4, 0, 1, 2, 3]), relevant, k=5) == pytest.approx(1.0)
