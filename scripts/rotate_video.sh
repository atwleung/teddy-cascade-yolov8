#!/usr/bin/env bash
set -euo pipefail

in="${1:?Usage: rotate_video.sh input.mp4 [output.mp4]}"
out="${2:-${in%.*}_rot.mp4}"

# Clockwise transpose (common for -90 rotation). If wrong direction, use transpose=cclock.
ffmpeg -y -i "$in" -vf "transpose=clock" -c:v libx264 -crf 18 -preset veryfast -an "$out"
echo "Wrote: $out"