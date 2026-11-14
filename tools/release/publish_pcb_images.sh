#!/usr/bin/env bash
# [Maintainers] Re-tag the two pinned PCB Agent images and push them to a public registry.
#   SRC_GEN=<internal pcb-kimi600 v34 image>  SRC_VERIFY=<internal KiCad 10.0.6 reviewer image> \
#   DST=ghcr.io/zhiman-bj bash tools/release/publish_pcb_images.sh
set -euo pipefail
DST=${DST:-ghcr.io/zhiman-bj}
docker pull "$SRC_GEN" && docker tag "$SRC_GEN" "$DST/pcb-agent-env:v34"
docker pull "$SRC_VERIFY" && docker tag "$SRC_VERIFY" "$DST/pcb-agent-verifier:kicad-10.0.6"
docker push "$DST/pcb-agent-env:v34"
docker push "$DST/pcb-agent-verifier:kicad-10.0.6"
