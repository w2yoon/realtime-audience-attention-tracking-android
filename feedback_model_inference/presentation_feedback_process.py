import base64
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pydub import AudioSegment
from openai import OpenAI

# ----------------- CONFIG -----------------
LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LMSTUDIO_MODEL_ID = "qwen2-vl-2b-instruct"

CLIPS_DIR = Path("clips")
OUT_DIR = Path("out")

MAX_SECONDS = 20          # take at most first 20 seconds worth of frames
FRAME_FPS = 1.0           # 1 frame per second
FRAME_SIZE = 256          # each frame resized to 256x256 before tiling
GRID_COLS = 5             # 5x4 => 20 frames
LOUDNESS_WINDOW_MS = 100  # 0.1s
SILENCE_DBFS = -45.0      # heuristic threshold for "quiet/silence"
TEMPERATURE = 0.4

FFMPEG_BIN = "ffmpeg"

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

# ----------------- UTIL -----------------
def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)

def ensure_ffmpeg_available() -> None:
    try:
        run([FFMPEG_BIN, "-version"])
    except Exception as e:
        raise RuntimeError(
            "ffmpeg executable not found. Install FFmpeg and ensure it's on PATH.\n"
            "Windows: winget install Gyan.FFmpeg"
        ) from e

def iter_videos(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in VIDEO_EXTS])

def image_to_data_url(img_path: Path) -> str:
    # LM Studio accepts data URLs for image input in OpenAI-compatible API.
    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    # We will save JPG; change if you use PNG
    return f"data:image/jpeg;base64,{b64}"

# ----------------- FRAMES -----------------
def extract_frames_1fps(video_path: Path, frames_dir: Path, fps: float = 1.0, max_seconds: int = 20) -> List[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for f in frames_dir.glob("frame_*.jpg"):
        f.unlink()

    # Select first max_seconds seconds, then sample fps
    # -ss before -i is faster; but can be less accurate. Here it's fine.
    # We'll just filter by time with -t.
    out_pattern = str(frames_dir / "frame_%03d.jpg")
    run([FFMPEG_BIN, "-y", "-i", str(video_path), "-t", str(max_seconds), "-vf", f"fps={fps}", out_pattern])

    return sorted(frames_dir.glob("frame_*.jpg"))

def make_grid_image(frame_paths: List[Path], out_path: Path, cols: int = 5, tile_size: int = 256) -> Path:
    if not frame_paths:
        raise RuntimeError("No frames extracted to build a grid image.")

    # Load and resize
    imgs = []
    for p in frame_paths:
        im = Image.open(p).convert("RGB")
        im = im.resize((tile_size, tile_size))
        imgs.append(im)

    rows = math.ceil(len(imgs) / cols)
    grid = Image.new("RGB", (cols * tile_size, rows * tile_size), (0, 0, 0))

    for i, im in enumerate(imgs):
        r = i // cols
        c = i % cols
        grid.paste(im, (c * tile_size, r * tile_size))

    grid.save(out_path, quality=90)
    return out_path

# ----------------- AUDIO + LOUDNESS -----------------
def extract_audio_wav(video_path: Path, wav_path: Path) -> None:
    # Mono 16kHz WAV for easy analysis
    run([FFMPEG_BIN, "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(wav_path)])

def loudness_curve_dbfs(wav_path: Path, window_ms: int = 100) -> Tuple[np.ndarray, float]:
    audio = AudioSegment.from_wav(wav_path).set_channels(1)
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples())

    win = int(sr * window_ms / 1000)
    if win <= 0:
        raise ValueError("Invalid window_ms; resulted in window size <= 0.")

    # Peak depends on sample width
    peak = float(2 ** (8 * audio.sample_width - 1))

    vals = []
    for i in range(0, len(samples), win):
        chunk = samples[i:i + win]
        if len(chunk) == 0:
            continue
        rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
        db = -float("inf") if rms == 0 else 20.0 * np.log10(rms / peak)
        vals.append(db)

    curve = np.array(vals, dtype=np.float64)
    step_sec = window_ms / 1000.0
    return curve, step_sec

def summarize_loudness(curve: np.ndarray, step_sec: float, silence_dbfs: float = -45.0) -> Dict:
    finite = curve[np.isfinite(curve)]
    if len(finite) == 0:
        return {
            "mean_dbfs": None,
            "max_dbfs": None,
            "min_dbfs": None,
            "pct_below_silence": None,
            "longest_quiet_streak_sec": None,
            "duration_sec_est": float(len(curve) * step_sec),
        }

    mean_db = float(np.mean(finite))
    max_db = float(np.max(finite))
    min_db = float(np.min(finite))

    below = (curve < silence_dbfs) | (~np.isfinite(curve))
    pct_below = float(np.mean(below))

    # longest streak below threshold
    longest = 0
    cur = 0
    for b in below:
        if b:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    longest_sec = float(longest * step_sec)

    return {
        "mean_dbfs": mean_db,
        "max_dbfs": max_db,
        "min_dbfs": min_db,
        "pct_below_silence": pct_below,
        "silence_threshold_dbfs": float(silence_dbfs),
        "longest_quiet_streak_sec": longest_sec,
        "duration_sec_est": float(len(curve) * step_sec),
        "step_sec": float(step_sec),
    }

# ----------------- VLM CALL (LM STUDIO) -----------------
def call_qwen_vl(grid_image_path: Path, loudness_summary: Dict) -> str:
    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")

    prompt = (
        "You are a presentation coach.\n"
        "The audience concentration score dropped during this moment.\n\n"
        "You are given:\n"
        "1) A contact-sheet image made from 1 frame per second (ordered left-to-right, top-to-bottom).\n"
        "2) Audio loudness summary in dBFS.\n\n"
        f"Audio summary (dBFS):\n{json.dumps(loudness_summary, indent=2)}\n\n"
        "Task:\n"
        "- From the grid image, check the presenter's posture, head orientation, slide-reading behavior, body movement, or turning away.\n"
        "- Check if the voice is low or loud, regardless of the image data.\n"
        "- Keep it concise and concrete.\n"
    )

    data_url = image_to_data_url(grid_image_path)

    resp = client.chat.completions.create(
        model=LMSTUDIO_MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        temperature=TEMPERATURE,
    )
    return resp.choices[0].message.content or ""

# ----------------- MAIN -----------------
def process_video(video_path: Path) -> Dict:
    out_dir = OUT_DIR / video_path.stem
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Frames -> grid
    frames = extract_frames_1fps(video_path, frames_dir, fps=FRAME_FPS, max_seconds=MAX_SECONDS)
    frames = frames[:MAX_SECONDS]  # ensure <= 20 frames
    grid_path = make_grid_image(frames, out_dir / "grid.jpg", cols=GRID_COLS, tile_size=FRAME_SIZE)

    # 2) Audio -> loudness curve -> summary
    wav_path = out_dir / "audio.wav"
    extract_audio_wav(video_path, wav_path)

    curve, step_sec = loudness_curve_dbfs(wav_path, window_ms=LOUDNESS_WINDOW_MS)
    summary = summarize_loudness(curve, step_sec, silence_dbfs=SILENCE_DBFS)

    # optional: save curve CSV for debugging / demo
    csv_path = out_dir / "loudness_curve.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "dbfs"])
        for i, db in enumerate(curve):
            w.writerow([round(i * step_sec, 3), float(db)])

    # 3) VLM feedback
    feedback = call_qwen_vl(grid_path, summary)

    return {
        "file": str(video_path),
        "artifacts": {
            "grid_image": str(grid_path),
            "loudness_csv": str(csv_path),
        },
        "loudness_summary": summary,
        "feedback": feedback,
    }

def main():
    ensure_ffmpeg_available()

    if not CLIPS_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {CLIPS_DIR.resolve()}")

    videos = iter_videos(CLIPS_DIR)
    if not videos:
        print(f"No videos found in {CLIPS_DIR.resolve()}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for vid in videos:
        print(f"\n=== Processing {vid.name} ===")
        try:
            r = process_video(vid)
            results.append(r)
            print("\n--- Feedback ---\n")
            print(r["feedback"])
            print("\n--------------\n")
        except Exception as e:
            results.append({"file": str(vid), "error": str(e)})
            print(f"[ERROR] {vid.name}: {e}")

    print("\n===== FINAL OUTPUT (JSON) =====\n")
    print(json.dumps(results, indent=2))
    print("\n===== END =====\n")

if __name__ == "__main__":
    main()
