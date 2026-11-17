#!/bin/sh
# Build every reel visual for a finished run:  sh viz/build.sh out/reel
set -e
run="$1"
uv run python -m viz.export "$run"
uv run python -m viz.dose "$run"
rm -rf "$run/viz/terrain_frames"
blender -b --python viz/terrain_blender.py -- "$run" > "$run/viz/blender.log" 2>&1
ffmpeg -y -loglevel error -framerate 30 -i "$run/viz/terrain_frames/f_%04d.png" \
  -c:v libx264 -pix_fmt yuv420p -crf 16 "$run/viz/terrain.mp4"
echo "visuals in $run/viz: terrain.mp4, terrain.blend, dose.mp4, dose_final.png"
