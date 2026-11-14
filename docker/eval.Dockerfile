# Portable runner for SemiBench and Semiconductor Agent (generation + verification).
#
#   docker build -f docker/eval.Dockerfile -t micius-eval .
#   docker run --rm -it --network host \
#       -v $MICIUS_DATA:/data -e MICIUS_DATA=/data -e OPENAI_API_KEY \
#       -v $PWD/outputs:/workspace/Micius-Flash/outputs micius-eval \
#       bash scripts/run_semiconductor_agent.sh
#
# The tool runtimes are relocated to /data on first use
# (python -m micius_eval.semiconductor_agent.setup_data --data-root /data/semiconductor_agent).
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip bubblewrap ca-certificates curl git zstd ripgrep \
        libgomp1 libgl1 libglib2.0-0 libxrender1 libxext6 \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @openai/codex \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/Micius-Flash
COPY . .
RUN python -m pip install -e ".[download]"
CMD ["bash"]
