"""Implicit-feedback matrix factorization for a short-video feed, from scratch.

The model is implicit-ALS in the style of Hu, Koren & Volinsky (2008), specialised
to watch signals. There are no explicit ratings on a short-video feed -- only how
long you watched. We turn that into two quantities per (user, video) pair:

    preference  p_uv = 1 if the video was watched at all, else 0
    confidence  c_uv = 1 + alpha * watch_ratio

A pair you barely watched still counts as a (weak) positive, but a pair you watched
to completion gets far more weight when fitting the factors. Everything you were
never shown is treated as a zero-preference observation with the baseline
confidence of 1 -- the classic implicit-feedback trick that lets a *sparse* set of
positives shape a *dense* prediction over the whole catalogue.

We learn user factors ``X`` (n_users x f) and item factors ``Y`` (n_videos x f) by
alternating ridge-regression least squares. The predicted preference for a pair is
``x_u . y_v``; ranking unseen videos by that score is what ``next_videos`` does.

Pure NumPy/SciPy, no ML framework.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class ImplicitMF:
    """Implicit-feedback matrix factorization trained with confidence-weighted ALS.

    Parameters
    ----------
    factors : latent dimension f.
    alpha : confidence scaling -- confidence = 1 + alpha * watch_ratio.
    regularization : ridge penalty on the factor rows.
    iterations : number of ALS sweeps (one sweep updates X then Y).
    seed : RNG seed for factor initialisation.
    """

    def __init__(
        self,
        factors: int = 32,
        alpha: float = 40.0,
        regularization: float = 0.1,
        iterations: int = 15,
        seed: int = 0,
    ) -> None:
        self.factors = factors
        self.alpha = alpha
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed
        self.X: np.ndarray | None = None  # user factors
        self.Y: np.ndarray | None = None  # item factors
        self._Cui: sp.csr_matrix | None = None  # confidence matrix (rows=users)
        self.n_users = 0
        self.n_videos = 0

    # ------------------------------------------------------------------ build
    @staticmethod
    def build_confidence(
        user_idx: np.ndarray,
        video_idx: np.ndarray,
        watch_ratio: np.ndarray,
        n_users: int,
        n_videos: int,
        alpha: float,
    ) -> sp.csr_matrix:
        """Sparse confidence matrix C with C_uv = 1 + alpha * watch_ratio on observed pairs.

        Stored sparsely: only observed pairs carry a stored value, and that value is
        ``alpha * watch_ratio`` (the *excess* confidence over the baseline of 1).
        The baseline 1 for every entry is folded into the ALS math analytically, so we
        never have to materialise the dense matrix.
        """
        data = alpha * watch_ratio.astype(np.float64)
        c = sp.csr_matrix(
            (data, (user_idx, video_idx)), shape=(n_users, n_videos)
        )
        c.sum_duplicates()
        return c

    def fit(
        self,
        user_idx: np.ndarray,
        video_idx: np.ndarray,
        watch_ratio: np.ndarray,
        n_users: int,
        n_videos: int,
    ) -> "ImplicitMF":
        """Fit factors to the observed watch signals via confidence-weighted ALS."""
        self.n_users, self.n_videos = n_users, n_videos
        self._Cui = self.build_confidence(
            user_idx, video_idx, watch_ratio, n_users, n_videos, self.alpha
        )
        Ciu = self._Cui.T.tocsr()

        rng = np.random.default_rng(self.seed)
        f = self.factors
        self.X = 0.01 * rng.standard_normal((n_users, f))
        self.Y = 0.01 * rng.standard_normal((n_videos, f))

        for _ in range(self.iterations):
            self._als_step(self.X, self.Y, self._Cui)  # update users
            self._als_step(self.Y, self.X, Ciu)         # update items
        return self

    # --------------------------------------------------------------- ALS core
    def _als_step(
        self, solve_for: np.ndarray, fixed: np.ndarray, C: sp.csr_matrix
    ) -> None:
        """One ALS half-step: update ``solve_for`` rows given ``fixed`` factors.

        Implements the Hu/Koren/Volinsky closed form. For user u with confidence
        row c (excess over baseline 1, sparse):

            A = Y^T Y + Y^T diag(c) Y + reg * I
            b = Y^T (c + 1) restricted to observed items   (because preference=1 there)
            x_u = A^{-1} b

        ``Y^T Y`` is precomputed once; the per-row work only touches that row's
        nonzeros, so the cost scales with the number of observations, not the
        dense matrix size.
        """
        f = self.factors
        YtY = fixed.T @ fixed  # (f, f)
        reg_eye = self.regularization * np.eye(f)
        base = YtY + reg_eye

        indptr, indices, cdata = C.indptr, C.indices, C.data
        for row in range(solve_for.shape[0]):
            start, end = indptr[row], indptr[row + 1]
            cols = indices[start:end]
            if len(cols) == 0:
                solve_for[row] = 0.0
                continue
            c_excess = cdata[start:end]          # = alpha * watch_ratio
            Yi = fixed[cols]                      # (k, f)
            # A = base + Yi^T diag(c_excess) Yi
            A = base + (Yi.T * c_excess) @ Yi
            # b = Yi^T (c_excess + 1)  [preference is 1 on observed items]
            b = Yi.T @ (c_excess + 1.0)
            solve_for[row] = np.linalg.solve(A, b)

    # ------------------------------------------------------------ prediction
    def predict(self, user: int, videos: np.ndarray | None = None) -> np.ndarray:
        """Predicted preference scores x_u . y_v for the given videos (default: all)."""
        assert self.X is not None and self.Y is not None, "model not fitted"
        if videos is None:
            return self.X[user] @ self.Y.T
        return self.Y[videos] @ self.X[user]

    def seen_videos(self, user: int) -> np.ndarray:
        """Videos this user has already watched (the observed nonzeros for the row)."""
        assert self._Cui is not None
        return self._Cui.indices[self._Cui.indptr[user] : self._Cui.indptr[user + 1]]

    def next_videos(self, user: int, k: int = 10, exclude_seen: bool = True) -> np.ndarray:
        """Rank unseen videos for ``user`` and return the top-k video indices."""
        scores = self.predict(user)
        if exclude_seen:
            scores = scores.copy()
            scores[self.seen_videos(user)] = -np.inf
        if k >= len(scores):
            return np.argsort(-scores)
        top = np.argpartition(-scores, k)[:k]
        return top[np.argsort(-scores[top])]

    # ----------------------------------------------------------- online update
    def online_update(
        self, user: int, video: int, watch_ratio: float, lr_passes: int = 1
    ) -> None:
        """Fold a fresh watch into the model and nudge the next recommendation.

        Rather than refitting everything, we (1) record the new confidence in the
        sparse matrix and (2) re-solve just this user's factor row against the fixed
        item factors -- the same ALS half-step the batch fit uses, but for one user.
        Cheap, online, and consistent with the batch objective.
        """
        assert self.X is not None and self.Y is not None and self._Cui is not None
        # update the confidence entry for (user, video)
        self._Cui = self._Cui.tolil()
        self._Cui[user, video] = self.alpha * float(watch_ratio)
        self._Cui = self._Cui.tocsr()

        f = self.factors
        YtY = self.Y.T @ self.Y + self.regularization * np.eye(f)
        for _ in range(lr_passes):
            cols = self.seen_videos(user)
            start = self._Cui.indptr[user]
            end = self._Cui.indptr[user + 1]
            c_excess = self._Cui.data[start:end]
            Yi = self.Y[cols]
            A = YtY + (Yi.T * c_excess) @ Yi
            b = Yi.T @ (c_excess + 1.0)
            self.X[user] = np.linalg.solve(A, b)
