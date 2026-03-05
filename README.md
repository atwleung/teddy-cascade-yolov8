# Teddy Cascade YOLOv8

A two-stage YOLOv8 pipeline for fine-grained teddy bear recognition in video.

The system works as a **cascade detector**:

1. **Stage 1 (YOLOv8 COCO)**  
   Detects `teddy bear` objects in video frames.

2. **Stage 2 (Custom YOLOv8 model)**  
   Classifies each teddy ROI into specific types:
   - `apeach`
   - `formosa_black_bear`
   - `panda`
   - `ryan`

Stage 2 can run either:
- locally on the same machine
- as an **HTTP inference API** (FastAPI)
- on a **remote GPU server** (e.g. Linode RTX 4000 Ada)

This architecture allows inexpensive local processing while optionally offloading heavy inference to a GPU server.

---

## Architecture

![Architecture](docs/architecture.png)

---

## Repo Contents

- `cascade_fast.py` — runs the two-stage cascade on a video (Stage2 local OR Stage2 API)
- `stage2_api.py` — FastAPI server for Stage2 inference (localhost now, GPU server later)
- `scripts/rotate_video.sh` — helper to rotate smartphone videos once with ffmpeg
- `docs/` — diagrams and demo assets

---

## Requirements

- Python 3.10+ recommended
- macOS Apple Silicon works well with **MPS** acceleration
- For remote GPU: CUDA-capable Linux machine (e.g. RTX 4000 Ada)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
