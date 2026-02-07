from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import cv2


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@dataclass
class Segment:
    start: float
    end: float

    @property
    def center(self) -> float:
        return 0.5 * (self.start + self.end)


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def get_video_duration_seconds(video_path: Path) -> float:
    """
    Use OpenCV for portability. (FFprobe is faster/more accurate, but OpenCV is fine.)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if total_frames <= 0:
        # fallback: try ffprobe if available
        if shutil.which("ffprobe"):
            cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
            ]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode == 0:
                try:
                    return float(p.stdout.strip())
                except Exception:
                    pass
        raise RuntimeError("Could not determine video duration (frame count unavailable).")
    return float(total_frames / fps)


def load_scores(scores_path: Path) -> pd.DataFrame:
    """
    Tries to load JSON as:
    - JSON array of objects
    - JSON Lines (one object per line)
    """
    try:
        df = pd.read_json(scores_path)
        if isinstance(df, pd.Series):
            df = df.to_frame().T
        return df
    except ValueError:
        # likely jsonl
        rows = []
        with open(scores_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    
def normalize_scores_df(
    df: pd.DataFrame,
    timestamp_col: str,
    score_col: str,
) -> tuple[pd.DataFrame, str, str]:
    """
    Supports both:
      - PC format: timestamp (sec), attention_score
      - Phone format: tsMs (ms), score100_1min

    Returns: (normalized_df, timestamp_col, score_col)
    where timestamp_col is seconds and score_col is numeric.
    """
    cols = set(df.columns)

    # If user didn't override cols, auto-detect phone schema
    if timestamp_col == "timestamp" and score_col == "attention_score":
        if "tsMs" in cols and "score100_1min" in cols:
            timestamp_col = "tsMs"
            score_col = "score100_1min"

    # If using phone timestamp in ms, convert to seconds into a canonical "timestamp" column
    if timestamp_col == "tsMs":
        df = df.copy()
        df["timestamp"] = pd.to_numeric(df["tsMs"], errors="coerce") / 1000.0
        timestamp_col = "timestamp"

    # Canonicalize score column name (optional but convenient)
    if score_col == "score100_1min":
        df = df.copy()
        df["attention_score"] = pd.to_numeric(df["score100_1min"], errors="coerce")
        score_col = "attention_score"
    else:
        df = df.copy()
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    # Drop rows that failed parsing
    df = df.dropna(subset=[timestamp_col, score_col])

    return df, timestamp_col, score_col


def compute_low_attention_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    score_col: str = "attention_score",
    threshold_mode: str = "below_overall_mean",
    threshold_value: Optional[float] = None,
) -> pd.Series:
    if timestamp_col not in df.columns or score_col not in df.columns:
        raise KeyError(f"Expected columns '{timestamp_col}' and '{score_col}' in scores file. Got: {list(df.columns)}")

    # mean per timestamp
    mean_per_t = (
        df.groupby(timestamp_col)[score_col]
        .mean()
        .reset_index()
        .sort_values(timestamp_col)
    )

    overall_mean = float(mean_per_t[score_col].mean())

    if threshold_mode == "below_overall_mean":
        thresh = overall_mean
    elif threshold_mode == "below_value":
        if threshold_value is None:
            raise ValueError("threshold_value is required when threshold_mode='below_value'")
        thresh = float(threshold_value)
    elif threshold_mode == "below_mean_minus_std":
        std = float(mean_per_t[score_col].std(ddof=0)) if len(mean_per_t) else 0.0
        thresh = overall_mean - std
    else:
        raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

    low = mean_per_t[mean_per_t[score_col] < thresh][timestamp_col]
    return low.astype(float)


def merge_timestamps_into_segments(timestamps: List[float], gap_sec: float) -> List[Segment]:
    """
    Merge timestamps that are within gap_sec of each other into continuous segments.
    Example: timestamps [10, 11, 40] with gap=2 => segments [10..11], [40..40]
    """
    if not timestamps:
        return []

    ts = sorted(float(t) for t in timestamps)
    segs: List[Segment] = []
    cur_s = ts[0]
    cur_e = ts[0]

    for t in ts[1:]:
        if t - cur_e <= gap_sec:
            cur_e = t
        else:
            segs.append(Segment(cur_s, cur_e))
            cur_s = cur_e = t

    segs.append(Segment(cur_s, cur_e))
    return segs


def extract_clip_ffmpeg(video_path: Path, out_path: Path, start: float, end: float) -> None:
    ensure_parent(out_path)
    duration = max(0.0, end - start)
    if duration <= 0.01:
        raise ValueError("Clip duration too small.")

    # -ss before -i is fast; -t duration
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c", "copy",
        str(out_path)
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        # fallback to re-encode if stream-copy fails (common with some MP4s)
        cmd2 = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(video_path),
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "veryfast",
            "-c:a", "aac",
            str(out_path)
        ]
        p2 = subprocess.run(cmd2, capture_output=True, text=True)
        if p2.returncode != 0:
            raise RuntimeError(f"FFmpeg failed.\nstream-copy stderr:\n{p.stderr}\nre-encode stderr:\n{p2.stderr}")


def extract_clip_opencv(video_path: Path, out_path: Path, start: float, end: float) -> None:
    ensure_parent(out_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError("Could not read video width/height.")

    start_frame = int(start * fps)
    end_frame = int(end * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for _ in range(max(0, end_frame - start_frame)):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=str, required=True, help="Path to attention_scores.json (or jsonl).")
    ap.add_argument("--video", type=str, required=True, help="Path to the full presentation video.")
    ap.add_argument("--out", type=str, default="clips", help="Output folder for extracted clips.")

    ap.add_argument("--timestamp-col", type=str, default="timestamp")
    ap.add_argument("--score-col", type=str, default="attention_score")

    ap.add_argument("--threshold-mode", type=str, default="below_overall_mean",
                    choices=["below_overall_mean", "below_value", "below_mean_minus_std"])
    ap.add_argument("--threshold-value", type=float, default=None,
                    help="Used only when threshold-mode=below_value.")

    ap.add_argument("--before", type=float, default=30.0, help="Seconds BEFORE the low-attention time/segment to include.")
    ap.add_argument("--after", type=float, default=30.0, help="Seconds AFTER the low-attention time/segment to include.")

    ap.add_argument("--merge-gap", type=float, default=2.0,
                    help="Merge low-attention timestamps if they are within this many seconds.")
    ap.add_argument("--max-clips", type=int, default=0,
                    help="If >0, export at most this many clips (highest priority = earliest).")

    ap.add_argument("--backend", type=str, default="ffmpeg", choices=["ffmpeg", "opencv"],
                    help="Extraction backend. ffmpeg is faster and keeps audio.")
    args = ap.parse_args()

    scores_path = Path(args.scores)
    video_path = Path(args.video)
    out_dir = Path(args.out)

    if not scores_path.exists():
        raise FileNotFoundError(scores_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if video_path.suffix.lower() not in VIDEO_EXTS:
        print(f"[WARN] Video extension {video_path.suffix} not in {sorted(VIDEO_EXTS)}. Proceeding anyway.")

    if args.backend == "ffmpeg" and not ffmpeg_available():
        print("[WARN] ffmpeg not found on PATH; falling back to OpenCV backend.")
        args.backend = "opencv"

    df = load_scores(scores_path)
    df, args.timestamp_col, args.score_col = normalize_scores_df(df, args.timestamp_col, args.score_col)
    low_ts = compute_low_attention_timestamps(
        df,
        timestamp_col=args.timestamp_col,
        score_col=args.score_col,
        threshold_mode=args.threshold_mode,
        threshold_value=args.threshold_value,
    ).tolist()

    if not low_ts:
        print("No low-attention timestamps found (nothing below threshold).")
        return

    duration = get_video_duration_seconds(video_path)
    segs = merge_timestamps_into_segments(low_ts, gap_sec=args.merge_gap)

    # Convert segments into actual clip [start,end] ranges
    clip_ranges: List[Tuple[float, float]] = []
    for s in segs:
        # Use the segment center so you get one clip per "low-attention region"
        center = s.center
        start = max(0.0, center - args.before)
        end = min(duration, center + args.after)
        clip_ranges.append((start, end))

    if args.max_clips and args.max_clips > 0:
        clip_ranges = clip_ranges[: args.max_clips]

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Video duration: {duration:.2f}s")
    print(f"Low-attention timestamps: {len(low_ts)}")
    print(f"Merged segments: {len(segs)}")
    print(f"Exporting clips: {len(clip_ranges)}")
    print(f"Backend: {args.backend}")

    for i, (start, end) in enumerate(clip_ranges):
        out_path = out_dir / f"clip_{i:03d}_{start:.1f}s_{end:.1f}s.mp4"
        print(f"[{i+1}/{len(clip_ranges)}] {out_path.name}  ({start:.2f} -> {end:.2f})")
        if args.backend == "ffmpeg":
            extract_clip_ffmpeg(video_path, out_path, start, end)
        else:
            extract_clip_opencv(video_path, out_path, start, end)

    print("Done.")


if __name__ == "__main__":
    main()
