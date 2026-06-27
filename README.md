# short-video-recommender

A recommendation system for an endless short-video feed: learn from split-second
watch signals to decide what plays next.

There are no stars or reviews on a short-video feed — only how long you watched
before you swiped. This is an **implicit-feedback** recommender that turns those
watch signals into the next pick, with a confidence-weighted matrix factorization
trained **from scratch** in NumPy/SciPy.

An original from-scratch build to learn the tools behind it — my own code.

- **Live page:** https://andreaisabelmontana.github.io/short-video-recommender/
- **Index of all my builds:** https://andreaisabelmontana.github.io/coursework-rebuilds/

## The model

**Watch signals as implicit feedback.** For each `(user, video)` interaction we
compute `watch_ratio = watch_time / duration ∈ [0, 1]` and split it, à la Hu/Koren/
Volinsky (2008), into:

- **preference** `p = 1` if the video was watched at all, else `0`;
- **confidence** `c = 1 + alpha * watch_ratio`.

A video you barely watched is a *weak* positive; a video you watched to completion
gets far more weight when fitting the factors. Everything you were never shown is a
zero-preference observation with baseline confidence `1` — the trick that lets a
sparse set of positives shape a dense prediction over the whole catalogue.

**Matrix factorization, from scratch.** We learn user factors `X` and item factors
`Y` by **confidence-weighted alternating least squares** (`feed/model.py`). Each ALS
half-step solves, per user `u`:

```
A = YᵀY + Yᵀ diag(c_excess) Y + reg·I
b = Yᵀ (c_excess + 1)        # preference = 1 on observed items
x_u = A⁻¹ b
```

`YᵀY` is precomputed once per sweep; the per-user work only touches that user's
observed items, so cost scales with the number of observations, not the dense
matrix. Predicted preference for a pair is `x_u · y_v`; `next_videos(user, k)` ranks
unseen videos by that score.

**Online update.** `online_update(user, video, watch_ratio)` folds a fresh watch in
by re-solving just that user's factor row against the fixed item factors — cheap,
online, and consistent with the batch objective. A completed watch nudges the next
recommendation toward that topic.

## The data (synthetic — honest about it)

There is no real platform behind this. `feed/data.py` generates synthetic
interactions with **planted structure**: every user has a sparse interest vector
over latent topics, every video a topic mix, and `watch_ratio` is a noisy, saturating
function of how well the two align. Exposure is a realistic blend of global
popularity and personal affinity, so the feed shows a mix of crowd-pleasers and
on-interest videos. Committed under `data/` (`python data/generate_dataset.py`
regenerates it).

## Real results

Held-out **Recall@10 / NDCG@10** on the planted-structure data (300 users, 400
videos, 6 topics, 18,000 interactions; relevant = held-out videos watched ≥ 60%),
from `python demo.py`:

| model               | Recall@10 | NDCG@10 |
|---------------------|-----------|---------|
| **implicit-MF**     | **0.1410** | **0.0927** |
| popularity baseline | 0.0518    | 0.0261  |

Personalised Recall@10 is **2.72×** the popularity baseline over 230 held-out users.

The live-session part of the demo shows the online update at work: after a user
watches three videos from their least-watched topic to completion, that topic's
share of the top-5 next-video recommendations rises.

## Tests

```
pip install -r requirements.txt
python -m pytest -q
```

```
.....                                                                    [100%]
5 passed
```

The suite checks the properties that matter:

1. the MF reconstructs a known low-rank preference matrix (AUC > 0.95, clear margin);
2. confidence weighting ranks a watched-to-completion video above a barely-watched one;
3. Recall@10 / NDCG@10 beat the popularity baseline on planted-structure data;
4. an online watch update moves the matching topic's videos up the next ranking.

## Layout

```
feed/
  data.py       synthetic watch-signal generator + train/test split
  model.py      confidence-weighted implicit-ALS matrix factorization
  evaluate.py   Recall@K, NDCG@K, popularity baseline
data/
  generate_dataset.py   writes interactions.csv + videos.csv
  interactions.csv      committed synthetic data (18k rows)
  videos.csv
demo.py         end-to-end: train, evaluate, simulate a live session
tests/          pytest suite
```

## Built with

Python 3.12, NumPy, SciPy. No ML framework — the matrix factorization is hand-rolled.
Page is plain HTML/CSS on GitHub Pages.
