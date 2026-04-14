"""
In-memory mood analytics for the current flight session.

Records periodic snapshots of per-seat emotion data and exposes
aggregation helpers consumed by the analytics API endpoints.
"""
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional
import time
import threading

# Maximum entries to keep (~8 h at 10 s intervals ≈ 2 880)
_MAX_SNAPSHOTS = 3000

# Emotion grouping used by the timeline chart
EMOTION_GROUPS: Dict[str, str] = {
    "happy":    "positive",
    "neutral":  "neutral",
    "angry":    "negative",
    "disgust":  "negative",
    "fear":     "negative",
    "sad":      "negative",
    "surprise": "surprise",
    "sleeping": "sleeping",
}


class MoodAnalytics:
    """Accumulates per-seat emotion snapshots for the current session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshots: List[dict] = []
        self._session_start: Optional[float] = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_snapshot(self, seat_emotions: Dict[str, tuple]) -> None:
        """Append a snapshot.

        *seat_emotions* maps ``seat_id -> (emotion, confidence)``.
        Only occupied seats should be included.
        """
        if not seat_emotions:
            return

        now = time.time()
        entry = {
            "ts": now,
            "iso": datetime.now().isoformat(),
            "seats": {
                sid: {"emotion": emo, "confidence": round(conf, 3)}
                for sid, (emo, conf) in seat_emotions.items()
            },
        }

        with self._lock:
            if self._session_start is None:
                self._session_start = now
            self._snapshots.append(entry)
            # Trim oldest entries when the buffer is full
            if len(self._snapshots) > _MAX_SNAPSHOTS:
                self._snapshots = self._snapshots[-_MAX_SNAPSHOTS:]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_session_start(self) -> Optional[str]:
        with self._lock:
            if self._session_start is None:
                return None
            return datetime.fromtimestamp(self._session_start).isoformat()

    def get_mood_over_time(self, bucket_seconds: int = 60) -> dict:
        """Return time-bucketed mood counts, grouped.

        Returns::

            {
                "buckets": ["12:00", "12:01", ...],
                "series": {
                    "positive": [3, 5, ...],
                    "neutral":  [1, 2, ...],
                    "negative": [0, 1, ...],
                    "surprise": [0, 0, ...],
                    "sleeping": [0, 0, ...],
                }
            }
        """
        with self._lock:
            snapshots = list(self._snapshots)

        if not snapshots:
            return {"buckets": [], "series": {g: [] for g in ["positive", "neutral", "negative", "surprise", "sleeping"]}}

        start_ts = snapshots[0]["ts"]
        end_ts = snapshots[-1]["ts"]

        # Build bucket boundaries
        buckets: List[str] = []
        bucket_counts: Dict[str, List[int]] = {g: [] for g in ["positive", "neutral", "negative", "surprise", "sleeping"]}
        t = start_ts
        while t <= end_ts + bucket_seconds:
            bucket_end = t + bucket_seconds
            label = datetime.fromtimestamp(t).strftime("%H:%M")
            counts: Dict[str, int] = {g: 0 for g in bucket_counts}

            for snap in snapshots:
                if t <= snap["ts"] < bucket_end:
                    for seat_info in snap["seats"].values():
                        group = EMOTION_GROUPS.get(seat_info["emotion"], "neutral")
                        counts[group] += 1

            buckets.append(label)
            for g in bucket_counts:
                bucket_counts[g].append(counts[g])

            t = bucket_end

        return {"buckets": buckets, "series": bucket_counts}

    def get_mood_distribution(self) -> dict:
        """Return total counts per individual emotion across all snapshots.

        Returns::

            {"happy": 120, "neutral": 80, "angry": 5, ...}
        """
        with self._lock:
            snapshots = list(self._snapshots)

        counts: Dict[str, int] = defaultdict(int)
        for snap in snapshots:
            for seat_info in snap["seats"].values():
                counts[seat_info["emotion"]] += 1
        return dict(counts)

    def snapshot_count(self) -> int:
        with self._lock:
            return len(self._snapshots)

    def clear(self) -> None:
        """Reset all data (e.g. when CV stops / restarts)."""
        with self._lock:
            self._snapshots.clear()
            self._session_start = None


# Module-level singleton
mood_analytics = MoodAnalytics()
