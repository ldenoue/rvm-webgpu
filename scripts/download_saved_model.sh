#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="$ROOT/.model-cache"
ARCHIVE="$CACHE/rvm_mobilenetv3_tf.zip"

mkdir -p "$CACHE"
curl -L \
  https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_tf.zip \
  -o "$ARCHIVE"
unzip -qo "$ARCHIVE" -d "$CACHE"
echo "SavedModel is ready at $CACHE/rvm_mobilenetv3_tf"
