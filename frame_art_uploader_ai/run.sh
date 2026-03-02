#!/bin/sh
set -e

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] launcher event=start component=frame_art_uploader_ai"

running=1
trap 'running=0' TERM INT

while [ "$running" -eq 1 ]; do
  if ! python3 /app/uploader.py; then
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] launcher event=worker_error component=frame_art_uploader_ai"
  fi
  sleep 2
done

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] launcher event=done component=frame_art_uploader_ai"
