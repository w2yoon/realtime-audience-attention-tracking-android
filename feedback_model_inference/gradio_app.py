from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

from presentation_feedback_process import process_video, ensure_ffmpeg_available

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
CLIPS_DIR = Path("clips")
OUT_DIR = Path("out")

def list_clips():
    if not CLIPS_DIR.exists():
        return []
    return sorted([p.name for p in CLIPS_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS])

def load_existing_result(clip_name: str):
    if not clip_name:
        return None, None, "Select a clip."

    stem = Path(clip_name).stem
    clip_out = OUT_DIR / stem

    grid_path = clip_out / "grid.jpg"
    csv_path = clip_out / "loudness_curve.csv"
    feedback_path = clip_out / "feedback.txt"

    if not grid_path.exists():
        return None, None, "No analysis found yet for this clip."

    fig = None
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            fig = plt.figure()
            plt.plot(df["time_sec"], df["dbfs"])
            plt.xlabel("Time (s)")
            plt.ylabel("dBFS")
            plt.title("Voice Loudness Over Time")
        except Exception:
            fig = None

    feedback = feedback_path.read_text(encoding="utf-8") if feedback_path.exists() else "(No saved feedback.txt)"
    return str(grid_path), fig, feedback

def analyze_selected_clip(selected_name):
    ensure_ffmpeg_available()

    if not selected_name:
        return None, None, "Please select a clip."

    video_path = CLIPS_DIR / selected_name
    if not video_path.exists():
        return None, None, "Clip not found."

    result = process_video(video_path)

    grid_path = result["artifacts"]["grid_image"]
    csv_path = result["artifacts"]["loudness_csv"]
    feedback = result.get("feedback", "No feedback generated.")

    fig = None
    try:
        df = pd.read_csv(csv_path)
        fig = plt.figure()
        plt.plot(df["time_sec"], df["dbfs"])
        plt.xlabel("Time (s)")
        plt.ylabel("dBFS")
        plt.title("Voice Loudness Over Time")
    except Exception:
        fig = None

    return grid_path, fig, feedback

def analyze_all_clips(progress=gr.Progress(track_tqdm=False)):
    ensure_ffmpeg_available()

    clips = sorted([p for p in CLIPS_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS]) if CLIPS_DIR.exists() else []
    if not clips:
        return "No videos found in ./clips", None, None, ""

    ok, fail = 0, 0
    errors = []

    last_grid = None
    last_fig = None
    last_feedback = ""

    n = len(clips)
    for i, vid in enumerate(clips, start=1):
        progress((i - 1) / n, desc=f"Analyzing {vid.name} ({i}/{n})")
        try:
            result = process_video(vid)  # IMPORTANT: use returned result
            ok += 1

            # ---- take the last video's outputs for UI ----
            last_grid = result["artifacts"]["grid_image"]
            csv_path = result["artifacts"]["loudness_csv"]
            last_feedback = result.get("feedback", "")

            # build plot for last clip
            last_fig = None
            try:
                df = pd.read_csv(csv_path)
                last_fig = plt.figure()
                plt.plot(df["time_sec"], df["dbfs"])
                plt.xlabel("Time (s)")
                plt.ylabel("dBFS")
                plt.title("Voice Loudness Over Time")
            except Exception:
                last_fig = None

        except Exception as e:
            fail += 1
            errors.append(f"{vid.name}: {e}")

    progress(1.0, desc="Done")

    status = f"Analyze All complete ✅  Success: {ok}   Failed: {fail}"
    if errors:
        status += "\n\nErrors:\n" + "\n".join(errors[:6])
        if len(errors) > 6:
            status += f"\n... and {len(errors)-6} more"

    shown = clips[-1].name if ok > 0 else "(none)"
    status += f"\n\nShowing last processed clip: {shown}"

    return status, last_grid, last_fig, last_feedback

with gr.Blocks(title="Presentation Attention Feedback") as demo:
    gr.Markdown("# 🎤 Presentation Feedback Assistant")
    gr.Markdown("Workflow: put cropped low-attention clips into `./clips/`, then analyze and review results.")

    with gr.Row():
        clips_dd = gr.Dropdown(choices=list_clips(), label="Select Clip", scale=4)
        refresh_btn = gr.Button("Refresh", scale=1)
        analyze_btn = gr.Button("Analyze Selected", variant="primary", scale=2)
        analyze_all_btn = gr.Button("Analyze All", variant="secondary", scale=2)

    with gr.Row():
        frame_grid = gr.Image(label="Frame Summary (1 FPS)", type="filepath", scale=1)
        loudness_plot = gr.Plot(label="Voice Loudness Curve", scale=1)

    feedback_text = gr.Textbox(label="AI Feedback", lines=10)

    status_box = gr.Textbox(label="Status", lines=6)

    refresh_btn.click(lambda: gr.Dropdown(choices=list_clips()), outputs=[clips_dd])

    analyze_btn.click(
        fn=analyze_selected_clip,
        inputs=[clips_dd],
        outputs=[frame_grid, loudness_plot, feedback_text]
    )

    analyze_all_btn.click(
        fn=analyze_all_clips,
        inputs=[],
        outputs=[status_box, frame_grid, loudness_plot, feedback_text]
    )


demo.launch(server_name="127.0.0.1", server_port=7860)