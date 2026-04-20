#!/bin/bash
# Download and extract the 10-minute audio segment from the source video.
# Source: Lex Fridman × Narendra Modi podcast
# Segment: 2:20:00 to 2:30:00 (10 minutes within the 2:20–2:54 window)
# You can change START_TIME to any point between 2:20:00 and 2:44:00

VIDEO_URL="https://www.youtube.com/watch?v=ZPUtA3W-7_I"
START_TIME="02:20:00"
DURATION="600"   # 10 minutes in seconds
OUTPUT="audio/original_segment.wav"

mkdir -p audio

echo "Checking for yt-dlp..."
if ! command -v yt-dlp &> /dev/null; then
    echo "yt-dlp not found. Installing..."
    pip install yt-dlp
fi

echo "Checking for ffmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg is required. Install it with:"
    echo "  sudo apt install ffmpeg   (Ubuntu/Debian)"
    echo "  brew install ffmpeg       (macOS)"
    exit 1
fi

echo "Downloading audio from YouTube..."
yt-dlp \
    --extract-audio \
    --audio-format wav \
    --output "audio/full_download.%(ext)s" \
    "$VIDEO_URL"

echo "Cutting 10-minute segment starting at $START_TIME..."
ffmpeg -i audio/full_download.wav \
    -ss "$START_TIME" \
    -t "$DURATION" \
    -ar 16000 \
    -ac 1 \
    "$OUTPUT" \
    -y

if [ -f "$OUTPUT" ]; then
    DURATION_CHECK=$(ffprobe -i "$OUTPUT" -show_entries format=duration -v quiet -of csv="p=0" 2>/dev/null)
    echo "Segment saved: $OUTPUT"
    echo "Duration: ${DURATION_CHECK}s"
    rm -f audio/full_download.wav
else
    echo "ERROR: Output file not created."
    exit 1
fi

echo "Done. Now run: python pipeline.py --input $OUTPUT"
