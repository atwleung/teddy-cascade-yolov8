# Teddy Cascade YOLOv8

A two-stage YOLOv8 pipeline for fine-grained teddy bear recognition in
video.

The system works as a cascade detector:

Stage 1 (YOLOv8 COCO) Detects teddy bear objects in video frames.

Stage 2 (Custom YOLOv8 model) Classifies each teddy ROI into specific
types: - apeach - formosa_black_bear - panda - ryan

Stage 2 can run either: - locally on the same machine - as an HTTP
inference API (FastAPI) - on a remote GPU server (e.g. Linode RTX 4000
Ada)

This architecture allows inexpensive local processing while optionally
offloading heavy inference to a GPU server.

------------------------------------------------------------------------

## ARCHITECTURE

See diagram:

![Architecture](docs/architecture.png)

------------------------------------------------------------------------

## REPO CONTENTS

cascade_fast.py Runs the two-stage cascade on a video. Stage 2 can be
local or HTTP API.

stage2_api.py FastAPI server providing Stage 2 inference.

scripts/rotate_video.sh Helper script to rotate smartphone videos using
ffmpeg.

docs/ Diagrams and demo images.

------------------------------------------------------------------------

## REQUIREMENTS

Python 3.10+ recommended

Install dependencies:

python -m venv .venv source .venv/bin/activate pip install -r
requirements.txt

------------------------------------------------------------------------

## STAGE 2 MODEL (best.pt)

This repository does NOT include trained weights.

After training YOLOv8 you will obtain:

runs/detect/train/weights/best.pt

Copy that file into the project root:

best.pt

or specify:

export MODEL_PATH=/path/to/best.pt

------------------------------------------------------------------------

## TRAIN YOUR OWN best.pt

You can train YOLOv8 using custom images and labels.

A good step-by-step tutorial:

https://www.youtube.com/watch?v=r0RspiLG260

Typical workflow:

1.  Collect images for each teddy class
2.  Annotate bounding boxes (Label Studio works well)
3.  Export dataset in YOLO format
4.  Train YOLOv8

Example training command:

yolo detect train model=yolov8n.pt data=/path/to/data.yaml imgsz=640
epochs=100 device=mps

Training output:

runs/detect/train/weights/best.pt

------------------------------------------------------------------------

## QUICKSTART

Optional: rotate smartphone videos first

bash scripts/rotate_video.sh input.mp4 input_rot.mp4

------------------------------------------------------------------------

## RUN CASCADE WITH LOCAL STAGE 2 MODEL

python cascade_fast.py -i input_rot.mp4 --stage2 best.pt

------------------------------------------------------------------------

## RUN STAGE 2 AS LOCAL API

Start API:

export MODEL_PATH=best.pt export DEVICE=mps export MAX_DET=10

uvicorn stage2_api:app --host 127.0.0.1 --port 8000

Test:

curl http://127.0.0.1:8000/health

Run cascade pointing to API:

python cascade_fast.py -i input_rot.mp4 --api
http://127.0.0.1:8000/predict --api-timeout 5

------------------------------------------------------------------------

## SAVE OUTPUT VIDEO

python cascade_fast.py -i input_rot.mp4 --api
http://127.0.0.1:8000/predict --api-timeout 5 -o out.mp4 --no-gui

------------------------------------------------------------------------

## DEPLOY STAGE 2 TO GPU SERVER

Example: Linode RTX 4000 Ada

On the server:

export MODEL_PATH=best.pt export DEVICE=0 export MAX_DET=10

uvicorn stage2_api:app --host 0.0.0.0 --port 8000

(Optional security)

export API_KEY=secret_token

On your Mac client:

python cascade_fast.py -i input_rot.mp4 --api
http://GPU_SERVER_IP:8000/predict --api-timeout 10 --api-key
secret_token

------------------------------------------------------------------------

## PERFORMANCE TIPS

The project includes several speed optimizations:

-   Stage1 uses yolov8n
-   Stage2 runs every N frames (default 5)
-   ROI batching
-   small inference size (320)

Further improvements:

Increase frame interval: --every 7 or 10

Reduce padding: --pad 0.06

Increase confidence: --s2-conf 0.40

Reduce Stage1 resolution: --s1-imgsz 448

Pre-resize ROIs before sending to API

Use tracking to reduce Stage2 calls

------------------------------------------------------------------------

## TODO / FUTURE IMPROVEMENTS

-   Deploy Stage2 to Linode RTX 4000 Ada GPU
-   Add Dockerfile for Stage2 API
-   Add Nginx + TLS reverse proxy example
-   Add object tracking (ByteTrack) to reduce Stage2 calls
-   Convert Stage2 to classification model for faster inference
-   Add benchmark script to measure FPS and latency
-   Add demo GIF to README
-   Provide automated deployment script

------------------------------------------------------------------------

LICENSE

MIT
