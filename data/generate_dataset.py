"""Generate the committed synthetic dataset (data/interactions.csv, data/videos.csv).

Run from the repo root:  python data/generate_dataset.py

The data is SYNTHETIC -- there is no real platform behind it. It is produced by
feed.data.generate with a fixed seed so the watch signals carry recoverable,
planted interest structure. See feed/data.py for the generative model.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feed import generate

SEED = 7
HERE = Path(__file__).resolve().parent


def main() -> None:
    inter = generate(n_users=300, n_videos=400, n_topics=6, density=0.15, seed=SEED)

    inter_path = HERE / "interactions.csv"
    with inter_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "video_id", "watch_time_s", "duration_s", "watch_ratio"])
        for u, v, wr in zip(inter.user_idx, inter.video_idx, inter.watch_ratio):
            dur = inter.durations[v]
            w.writerow([int(u), int(v), round(float(wr) * dur, 2), round(float(dur), 2),
                        round(float(wr), 4)])

    vids_path = HERE / "videos.csv"
    with vids_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "duration_s", "dominant_topic"])
        for v in range(inter.n_videos):
            dom = int(inter.video_topics[v].argmax())
            w.writerow([v, round(float(inter.durations[v]), 2), dom])

    print(f"wrote {inter_path} ({len(inter.user_idx)} rows)")
    print(f"wrote {vids_path} ({inter.n_videos} rows)")


if __name__ == "__main__":
    main()
