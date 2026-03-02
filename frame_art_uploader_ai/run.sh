#!/bin/sh
set -e

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] launcher event=start component=frame_art_uploader_ai"
python3 /app/uploader.py
echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] launcher event=done component=frame_art_uploader_ai"
