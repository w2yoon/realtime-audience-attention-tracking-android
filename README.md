# Audience Attention Estimation (Android Demo)

An on-device Android demo that estimates **audience attention** in real time using
CameraX + ML Kit Face Detection.

The system is designed for **presentation / demo scenarios**, where a presenter wants
to objectively measure audience engagement without invasive tracking.





Perfect — here is the clean **raw Markdown version**.
You can copy-paste this directly into `README.md`.

---

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

# 🧪 Demo Mode

For testing without phone:

1. Generate synthetic attention JSON
2. Use any presentation video
3. Run pipeline

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

---

# 🎯 Goal

Provide accessible, AI-powered presentation coaching that:

* Identifies weak engagement moments
* Analyzes non-verbal cues
* Offers actionable improvement advice
* Works entirely offline (local VLM)

```

---

If you later want a cleaner GitHub-style version without emojis (more professional), I can rewrite it in a minimal engineering tone as well.
```
