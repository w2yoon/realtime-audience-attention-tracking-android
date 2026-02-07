# Real-time Audience Attention Tracking & AI Presentation Feedback

A mobile, **on-device** system that helps speakers understand **audience attention in real time**—without sending faces or frames to the cloud—then enables **post-session AI presentation feedback** using a recorded presenter video and time-aligned attention logs.

---

## Why this project

When presenting, it’s hard to tell whether the room is engaged. This project provides:

1. **Real-time audience attention tracking (rear camera, on-device)**
   - Detects faces and estimates attention signals (head pose + eye openness).
   - Aggregates engagement into a **minute-level attention score**.
   - Runs **fully on-device** to reduce privacy risks (no cloud inference, no data upload required).

2. **AI presentation feedback (front camera recording, post-processing)**
   - Optionally records the presenter (front camera) during the session.
   - Saves **time-stamped attention scores** (JSON).
   - After the session, the presenter video + attention timeline can be analyzed to generate actionable feedback
     (e.g., where engagement dropped, pacing/sections to improve).

---

## Key Features

- **On-device face detection** using Google ML Kit
- **Per-person attention estimation** (smoothed over time)
- **Crowd-level attention score** (rolling 1-minute window)
- **Session logging**: attention score + metadata saved as JSON with timestamps
- **Export/sharing**: share JSON (and presenter video when available) to a PC for further analysis
- **Privacy-first design**: audience frames are processed locally and not stored by default

---

## How it works (high level)

### Rear camera (Audience)
1. Rear camera frames → ML Kit face detection  
2. Each detected face → attention score:
   - Head pose deviation (yaw/pitch)
   - Eye openness / eye drop
   - Temporal smoothing (EMA)
3. Per-frame crowd mean score → stored into a rolling 1-minute buffer  
4. Output:
   - `Attention 0–100 (1m)` score
   - number of faces (`nFaces`)
   - confidence estimate (`confidence`)
   - optional face overlays (bbox + per-face score) for demo/debug

### Front camera (Presenter)
- Records the presenter video during the session (device capability dependent).
- At stop, exports:
  - `presenter_<timestamp>.mp4`
  - `attention_<timestamp>.json`

---

## Output Format (JSON)

A session produces an array of minute-level samples:

```json
[
  {
    "tsMs": 1730000000000,
    "score100_1min": 74,
    "faces": 5,
    "confidence": 0.82
  }
]



```markdown
# 🎤 AI Presentation Feedback System

An end-to-end presentation coaching pipeline that analyzes recorded presentation videos and generates automated feedback using:

- 📱 Mobile attention tracking (face/engagement logs)
- 🎬 Low-attention clip extraction
- 🖼 Vision-Language Model (Qwen2-VL via LM Studio)
- 🖥 Gradio demo interface

This system provides post-presentation feedback on:

- Attention trends  
- Low-engagement moments  
- Visual posture / gaze behavior  
- Overall delivery quality  

---

# 📌 System Overview

## Workflow

```

Phone App
↓
presentation_whole.mp4
presentation_self.json
↓
PC Pipeline
↓
Low-attention clip extraction
↓
Frame grid generation
↓
Vision-Language Model (Qwen2-VL)
↓
AI feedback

```

---

# 📁 Project Structure

```

.
├── presentation_whole.mp4
├── presentation_self.json
├── extract_low_attention_clips.py
├── presentation_feedback_process.py
├── gradio_app.py
├── clips/
├── out/
└── README.md

```

---

# 📱 Phone Output Format

After the presentation ends, the phone sends:

## 1️⃣ Full Video

```

presentation_whole.mp4

```

## 2️⃣ Attention Logs

```

presentation_self.json

````

### JSON Structure

```json
[
  {
    "tsMs": 0,
    "score100_1min": 88,
    "faces": 1,
    "confidence": 0.95
  },
  {
    "tsMs": 30000,
    "score100_1min": 85,
    "faces": 1,
    "confidence": 0.93
  }
]
````

### Fields

| Field           | Description               |
| --------------- | ------------------------- |
| `tsMs`          | Timestamp in milliseconds |
| `score100_1min` | Attention score (0–100)   |
| `faces`         | Number of detected faces  |
| `confidence`    | Detection confidence      |

The PC pipeline automatically converts:

* `tsMs → timestamp (seconds)`
* `score100_1min → attention_score`

---

# 🎬 Step 1: Extract Low-Attention Clips

This script:

* Computes overall attention mean
* Finds timestamps below threshold
* Merges nearby timestamps
* Extracts video clips around low-attention moments

### Run:

```bash
python extract_low_attention_clips.py \
  --video presentation_whole.mp4 \
  --scores presentation_self.json
```

### Optional tuning:

```bash
--before 20
--after 20
--threshold-mode below_mean_minus_std
--backend ffmpeg
```

### Output

```
clips/
  clip_000_90.0s_130.0s.mp4
  clip_001_150.0s_190.0s.mp4
```

---

# 🖼 Step 2: Visual Feedback Generation

`presentation_feedback_process.py`:

1. Extracts representative frames
2. Creates a grid image
3. Sends image + summary to Qwen2-VL
4. Receives structured AI feedback

---

# 🤖 Vision Model Setup (LM Studio)

## Requirements

* LM Studio installed
* OpenAI-compatible server enabled
* Vision model loaded (e.g., `qwen2-vl-2b-instruct`)

Server must run at:

```
http://127.0.0.1:1234
```

Test in browser:

```
http://127.0.0.1:1234/v1/models
```

---

# 🖥 Step 3: Run Gradio Demo

Launch UI:

```bash
python gradio_app.py
```

Gradio runs at:

```
http://127.0.0.1:7860
```

Upload video → Get AI feedback.

---

# 🧠 Feedback Logic

The system combines:

## 📊 Attention Analysis

* Overall mean attention
* Low-attention regions
* Recovery trends

## 🖼 Visual Analysis

* Posture
* Gaze direction
* Body movement
* Engagement cues

## 📝 Final Output

Structured coaching feedback:

* Strengths
* Weak moments
* Specific suggestions
* Actionable improvement tips

---

# ⚙ Dependencies

Install:

```bash
pip install pandas opencv-python gradio requests
```

If using ffmpeg backend:

```bash
brew install ffmpeg
```

or

```bash
sudo apt install ffmpeg
```

---

# 🚀 Future Improvements

* Real-time feedback mode
* Confidence-weighted scoring
* Multi-person audience tracking
* Speech-to-text analysis
* Slide-change detection
* Attention visualization graph in UI

---

# 🏗 Architecture Diagram

```
Phone App
   ├── Video Recording
   ├── Face Tracking
   └── Attention Scoring
         ↓
PC Backend
   ├── Low Attention Detection
   ├── Clip Extraction
   ├── Frame Grid Generation
   └── Vision-Language Model
         ↓
AI Feedback
         ↓
Gradio UI
```
