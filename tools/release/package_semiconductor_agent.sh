#!/usr/bin/env bash
# [Maintainers] Pack a built release directory into shards for the Hugging Face dataset.
#   bash tools/release/package_semiconductor_agent.sh release/semiconductor_agent upload/semiconductor_agent
# Hardlinks created by --dedupe are preserved inside each archive, so the
# whole source_tree goes into one stream split into 20 GB parts.
set -euo pipefail
SRC=$(realpath "$1"); DST=$2
mkdir -p "$DST/archives"
cp "$SRC/tasks.jsonl" "$SRC/manifest.json" "$DST/"
tar -C "$SRC" -cf - environments rubrics | zstd -T0 -10 | split -b 20G -d -a 3 - "$DST/archives/environments.tar.zst.part-"
tar -C "$SRC" -cf - source_tree        | zstd -T0 -10 | split -b 20G -d -a 3 - "$DST/archives/runtimes.tar.zst.part-"
( cd "$DST/archives" && sha256sum *.part-* > SHA256SUMS )
echo "Upload with: hf upload <org>/Micius-Flash-Bench $DST semiconductor_agent --repo-type dataset"
