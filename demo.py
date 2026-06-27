"""End-to-end demo of the short-video feed recommender.

Run:  python demo.py

It (1) generates synthetic watch-signal data with planted interests, (2) trains the
confidence-weighted implicit-MF, (3) reports held-out Recall@K / NDCG@K against a
popularity baseline, and (4) simulates a live session -- showing how the next-video
recommendations shift as a user watches a particular topic. All numbers printed are
from the actual run.
"""

from __future__ import annotations

import numpy as np

from feed import (
    ImplicitMF,
    PopularityBaseline,
    evaluate_baseline,
    evaluate_model,
    generate,
    train_test_split,
)

SEED = 7
K = 10


def dominant_topic(video_topics: np.ndarray, video: int) -> int:
    return int(np.argmax(video_topics[video]))


def main() -> None:
    print("=" * 68)
    print("Short-video feed recommender -- implicit watch-signal MF")
    print("=" * 68)

    # 1. data -----------------------------------------------------------------
    inter = generate(n_users=300, n_videos=400, n_topics=6, density=0.15, seed=SEED)
    train, heldout = train_test_split(inter, test_frac=0.2, seed=SEED)
    print(f"\nSynthetic data (planted interests, honest synthetic):")
    print(f"  users={inter.n_users}  videos={inter.n_videos}  topics=6")
    print(f"  observed interactions={len(inter.user_idx)}")
    print(f"  watch_ratio: median={np.median(inter.watch_ratio):.3f} "
          f"p90={np.quantile(inter.watch_ratio, 0.9):.3f}")
    print(f"  held-out users with liked items={len(heldout)}")

    # 2. train ----------------------------------------------------------------
    model = ImplicitMF(factors=32, alpha=40, regularization=0.1, iterations=15, seed=0)
    model.fit(
        train.user_idx, train.video_idx, train.watch_ratio,
        train.n_users, train.n_videos,
    )
    baseline = PopularityBaseline().fit(
        train.video_idx, train.user_idx, train.n_videos
    )

    # 3. evaluate -------------------------------------------------------------
    m = evaluate_model(model, heldout, k=K)
    b = evaluate_baseline(baseline, heldout, k=K)
    print(f"\nHeld-out ranking quality (over {m['n_users']} users):")
    print(f"  {'model':<22}{'Recall@'+str(K):>12}{'NDCG@'+str(K):>12}")
    print(f"  {'implicit-MF':<22}{m['recall@'+str(K)]:>12.4f}{m['ndcg@'+str(K)]:>12.4f}")
    print(f"  {'popularity baseline':<22}{b['recall@'+str(K)]:>12.4f}{b['ndcg@'+str(K)]:>12.4f}")
    lift = m['recall@'+str(K)] / b['recall@'+str(K)]
    print(f"  --> personalised Recall@{K} is {lift:.2f}x the popularity baseline")

    # 4. live session ---------------------------------------------------------
    print("\n" + "-" * 68)
    print("Live session: watch a topic, watch the feed react")
    print("-" * 68)

    # pick a user and a 'fresh' topic they have not engaged with much
    user = 3
    seen = set(model.seen_videos(user).tolist())
    topic_counts = np.zeros(6)
    for v in seen:
        topic_counts[dominant_topic(inter.video_topics, v)] += 1
    target_topic = int(np.argmin(topic_counts))  # their least-watched topic

    def topic_of_top(k: int = 5):
        recs = model.next_videos(user, k=k)
        return [dominant_topic(inter.video_topics, v) for v in recs]

    print(f"\nUser {user}'s least-watched topic so far: topic {target_topic}")
    before_recs = model.next_videos(user, k=5)
    print(f"  top-5 next-video topics BEFORE: {topic_of_top()}")
    share_before = topic_of_top().count(target_topic)

    # simulate the user binge-watching three topic-`target_topic` videos
    unseen_target = [
        v for v in range(inter.n_videos)
        if dominant_topic(inter.video_topics, v) == target_topic and v not in seen
    ]
    watched = unseen_target[:3]
    print(f"\n  user watches 3 topic-{target_topic} videos to completion "
          f"(watch_ratio 0.97): {watched}")
    for v in watched:
        model.online_update(user=user, video=v, watch_ratio=0.97)

    after_topics = topic_of_top()
    share_after = after_topics.count(target_topic)
    print(f"  top-5 next-video topics AFTER:  {after_topics}")
    print(f"\n  topic-{target_topic} share of top-5: {share_before}/5 -> {share_after}/5")
    if share_after > share_before:
        print("  --> the online update steered the feed toward the freshly-watched topic.")
    else:
        print("  --> (no shift this run; try another user/topic)")

    print("\nDone. All numbers above are from this run.")


if __name__ == "__main__":
    main()
