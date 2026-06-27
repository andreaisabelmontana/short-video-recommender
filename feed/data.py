"""Synthetic short-video interaction data with planted latent structure.

The data here is SYNTHETIC. It is generated so that watch behaviour actually
carries signal: every user has a hidden interest vector over a handful of topics,
every video has a hidden topic mix, and how long a user watches a video is driven
by how well those two line up (plus noise). That gives a recommender something
real to recover, while keeping the whole pipeline reproducible and dependency-light.

Nothing here is scraped from a real platform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Interactions:
    """A bundle of generated interactions plus the ground-truth that made them.

    Attributes
    ----------
    users, videos : index ranges implied by the matrices' shapes.
    watch_ratio : (n_obs,) float array in [0, 1], watch_time / duration.
    user_idx, video_idx : (n_obs,) int arrays, the (u, v) pair per observation.
    durations : (n_videos,) float array, seconds.
    user_topics, video_topics : ground-truth latent factors (for evaluation only).
    """

    user_idx: np.ndarray
    video_idx: np.ndarray
    watch_ratio: np.ndarray
    durations: np.ndarray
    user_topics: np.ndarray
    video_topics: np.ndarray
    n_users: int
    n_videos: int

    def watch_ratio_matrix(self) -> np.ndarray:
        """Dense (n_users, n_videos) watch-ratio matrix; unobserved entries are NaN."""
        m = np.full((self.n_users, self.n_videos), np.nan, dtype=np.float64)
        m[self.user_idx, self.video_idx] = self.watch_ratio
        return m


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def generate(
    n_users: int = 300,
    n_videos: int = 800,
    n_topics: int = 8,
    density: float = 0.06,
    noise: float = 0.08,
    seed: int = 0,
) -> Interactions:
    """Generate a synthetic short-video interaction log with planted structure.

    Each user gets a sparse interest distribution over ``n_topics`` topics and each
    video a topic mix. The *affinity* of a user for a video is the dot product of
    those two distributions. We then sample which videos each user is exposed to
    (popular videos more likely), and the watch_ratio for an exposed pair is a
    noisy, saturating function of affinity -- a good match is watched longer.

    Returns an :class:`Interactions` bundle.
    """
    rng = np.random.default_rng(seed)

    # --- latent ground truth -------------------------------------------------
    # Users: each cares about ~2 topics strongly.
    user_logits = rng.normal(0.0, 1.0, size=(n_users, n_topics))
    top2 = np.argsort(user_logits, axis=1)[:, :-2]  # all but top-2 columns
    for u in range(n_users):
        user_logits[u, top2[u]] -= 6.0
    user_topics = _softmax(user_logits * 1.5, axis=1)

    # Videos: each belongs mostly to ~1-2 topics.
    video_logits = rng.normal(0.0, 1.0, size=(n_videos, n_topics))
    video_topics = _softmax(video_logits * 2.0, axis=1)

    durations = rng.uniform(8.0, 60.0, size=n_videos)  # seconds

    # video popularity (a few hits, long tail) for exposure sampling
    popularity = rng.pareto(2.0, size=n_videos) + 0.2
    popularity = popularity / popularity.sum()

    # --- affinity ------------------------------------------------------------
    affinity = user_topics @ video_topics.T  # (n_users, n_videos), in [0,1]
    # spread it out so watch-ratio uses the full range
    affinity = (affinity - affinity.min()) / (affinity.max() - affinity.min() + 1e-12)

    # --- sample exposures ----------------------------------------------------
    n_per_user = max(1, int(round(density * n_videos)))
    user_list, video_list, ratio_list = [], [], []
    for u in range(n_users):
        # Exposure is a realistic blend: the feed mostly serves popular videos but
        # is already tilted toward the user's interests. We mix global popularity
        # with this user's affinity so a user *sees* a mix of crowd-pleasers and
        # on-interest videos -- exactly the setting where pure popularity is a weak
        # personal predictor and a learned model should win.
        expose = 0.5 * popularity + 0.5 * (affinity[u] / affinity[u].sum())
        expose = expose / expose.sum()
        seen = rng.choice(n_videos, size=n_per_user, replace=False, p=expose)
        a = affinity[u, seen]
        # watch_ratio: a saturating (logistic) function of affinity plus noise,
        # clipped to [0,1]. A well-matched video is watched to near-completion;
        # a poorly-matched one is flicked away after a moment.
        base = 1.0 / (1.0 + np.exp(-7.0 * (a - 0.45)))
        wr = base + rng.normal(0.0, noise, size=a.shape)
        wr = np.clip(wr, 0.0, 1.0)
        user_list.append(np.full(n_per_user, u))
        video_list.append(seen)
        ratio_list.append(wr)

    return Interactions(
        user_idx=np.concatenate(user_list),
        video_idx=np.concatenate(video_list),
        watch_ratio=np.concatenate(ratio_list),
        durations=durations,
        user_topics=user_topics,
        video_topics=video_topics,
        n_users=n_users,
        n_videos=n_videos,
    )


def train_test_split(
    inter: Interactions, test_frac: float = 0.2, seed: int = 0
) -> tuple[Interactions, dict[int, np.ndarray]]:
    """Hold out a fraction of each user's interactions for evaluation.

    Returns ``(train_interactions, heldout)`` where ``heldout`` maps a user index
    to the array of video indices held out for that user (their "relevant" set,
    restricted to genuinely-enjoyed videos, watch_ratio >= 0.6).
    """
    rng = np.random.default_rng(seed)
    keep_mask = np.ones(inter.user_idx.shape[0], dtype=bool)
    heldout: dict[int, np.ndarray] = {}

    for u in range(inter.n_users):
        rows = np.where(inter.user_idx == u)[0]
        if len(rows) < 3:
            continue
        n_test = max(1, int(round(test_frac * len(rows))))
        test_rows = rng.choice(rows, size=n_test, replace=False)
        keep_mask[test_rows] = False
        # only "liked" held-out videos count as relevant for recall/ndcg
        liked = test_rows[inter.watch_ratio[test_rows] >= 0.6]
        if len(liked):
            heldout[u] = inter.video_idx[liked]

    train = Interactions(
        user_idx=inter.user_idx[keep_mask],
        video_idx=inter.video_idx[keep_mask],
        watch_ratio=inter.watch_ratio[keep_mask],
        durations=inter.durations,
        user_topics=inter.user_topics,
        video_topics=inter.video_topics,
        n_users=inter.n_users,
        n_videos=inter.n_videos,
    )
    return train, heldout
