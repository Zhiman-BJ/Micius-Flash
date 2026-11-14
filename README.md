<div align="center">

# Micius-Flash

**Lightweight Agentic Intelligence for Engineering**

<a href="https://arxiv.org/abs/2611.10015"><b>📄 Technical Report</b></a> &nbsp;|&nbsp;
<a href="https://huggingface.co/zhiman-ai/Micius-Flash"><b>🤗 Model</b></a> &nbsp;|&nbsp;
<a href="https://huggingface.co/datasets/zhiman-ai/Micius-Flash-Bench"><b>📊 Benchmarks</b></a> &nbsp;|&nbsp;
<a href="README_zh.md"><b>中文</b></a>

</div>

## Introduction

**Micius-Flash** is a 27B-parameter model for engineering, with native multimodal input, deep reasoning and a 256K-token context window. It is trained with a knowledge-tree-driven data flywheel and **Multi-Teacher On-Policy Distillation (MOPD)**: four domain experts (knowledge & reasoning, hardware coding, multimodal, agentic) are trained by SFT, and the student is then distilled on its own on-policy trajectories, each trajectory receiving token-level supervision (top-32 forward KL) from the teacher of its task category.

- **Hardware coding:** 64.2% on average over 12 hardware-code benchmarks, +15.0 points over a same-size baseline; best or tied-best on RTLLM-v2.
- **Engineering agents:** 62.4% on Semiconductor Agent (+17.2 over the same-size baseline) and 83.3% on PCB Agent, second only to GPT-5.6 Sol among evaluated models.
- **Circuit vision:** best reported results on RFSchemBench (72.4%) and CircuitSense (71.0%).

| | |
|---|---|
| Parameters | 27.4B (BF16), vision-language |
| Architecture | 64 layers, hybrid attention (48 linear-attention + 16 full-attention layers), 27-layer vision encoder |
| Context | 262,144 tokens |
| Thinking | `reasoning_effort` = `low` / `medium` / `xhigh`, or `enable_thinking=false` |
| Tool calling | OpenAI function-calling API (Qwen3-Coder XML under the hood) |
| License | Apache 2.0 |

Details in [docs/inference.md](docs/inference.md). This repository contains:

| | |
|---|---|
| [`inference/`](inference) | Serving (SGLang / vLLM), OpenAI-compatible client examples (thinking effort, tool calling, images) and offline Transformers inference |
| [`model_card/`](model_card) | The Hugging Face model card |
| [`micius_eval/`](micius_eval) | Evaluation code for the three benchmarks introduced in the report: **SemiBench**, **Semiconductor Agent**, **PCB Agent** |
| [`data/`](data) | SemiBench (167 questions) and PCB Agent (30 tasks) inputs. Semiconductor Agent environments are downloaded separately (see below) |

## News

- **2026-11** — Micius-Flash weights released on [Hugging Face](https://huggingface.co/zhiman-ai/Micius-Flash), together with the technical report, inference code and the three Micius benchmarks.

## Evaluation Results

All models are evaluated at maximum reasoning effort (GPT-5.6 Sol: `xhigh`). "-" means not reported.

| Benchmark | Qwen3.8-27B | Muse Glimmer | MiniMax M3 | Tencent Hy3 | Kimi K2.6 | DeepSeek V4-Pro | Gemini 3.0 Flash | Claude Sonnet 5.0 | GPT-5.6 Sol | **Micius-Flash** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ***Reasoning & knowledge*** | | | | | | | | | | |
| SemiBench | 42.51 | 29.94 | 56.89 | 52.10 | 45.51 | 67.66 | 68.50 | 66.40 | 88.02 | **49.70** |
| SuperGPQA (Circuits) | 70.37 | 63.89 | 61.11 | 73.15 | 67.59 | 73.15 | 72.55 | 67.86 | 75.93 | **72.22** |
| SuperGPQA (Emfield) | 84.09 | 82.95 | 78.41 | 86.36 | 88.64 | 85.23 | 87.50 | 86.67 | 87.50 | **86.36** |
| SuperGPQA (Microelec) | 76.00 | 70.00 | 82.00 | 78.00 | 82.00 | 82.00 | 78.12 | 79.31 | 78.00 | **84.00** |
| SuperGPQA (Physics) | 63.93 | 63.93 | 63.93 | 70.49 | 72.13 | 70.49 | 85.37 | 72.22 | 65.57 | **70.49** |
| ***Vision*** | | | | | | | | | | |
| RFSchemBench | 66.80 | 54.87 | 54.56 | - | 53.01 | - | - | - | 50.84 | **72.40** |
| PCB-Bench | 80.00 | 72.17 | 77.39 | 47.83 | 77.39 | - | - | - | 86.96 | **80.60** |
| CircuitSense | 69.80 | 52.07 | 41.42 | 26.04 | 49.70 | - | - | - | 60.36 | **71.00** |
| ***Coding*** | | | | | | | | | | |
| CVDP-static | 43.70 | 39.40 | 57.95 | 59.60 | 50.99 | 57.28 | 57.00 | - | 65.80 | **58.90** |
| VerilogEval-v2 | 77.80 | 82.60 | 81.35 | 88.42 | 77.81 | 89.71 | 50.50 | 66.30 | 94.20 | **86.80** |
| RTLLM-v2 | 65.10 | 65.10 | 76.74 | 74.42 | 72.09 | 79.07 | 78.00 | 69.60 | 79.10 | **79.10** |
| ArchXBench | 50.70 | 56.30 | 59.15 | 64.79 | 53.52 | 66.20 | 67.80 | 82.60 | 80.00 | **53.50** |
| NetlistBench | 69.80 | 83.00 | 75.62 | 82.84 | 90.56 | 77.97 | 68.95 | 79.40 | 75.60 | **83.80** |
| VHDLBench | 70.90 | 73.80 | 71.84 | 75.24 | 72.33 | 83.50 | 86.50 | 83.20 | 84.50 | **76.20** |
| Pluto | 62.10 | 64.70 | 68.10 | 68.10 | 65.52 | 68.10 | 76.00 | 72.30 | 78.50 | **72.40** |
| NotSoTiny | 14.70 | 12.70 | 14.83 | 14.49 | 12.57 | 17.36 | - | - | 18.80 | **16.10** |
| Rule2DRC | 4.90 | 5.80 | 12.80 | 16.10 | 29.00 | 34.60 | 32.10 | 44.30 | 75.50 | **48.30** |
| PyHDL-Eval | 19.60 | 49.80 | 35.22 | 41.47 | 46.53 | 53.67 | 67.10 | 60.60 | 74.50 | **65.80** |
| CreativEval | 71.70 | 70.80 | 71.67 | 78.33 | 72.50 | 88.33 | 84.40 | 83.90 | 94.10 | **79.20** |
| ChipBench | 39.10 | 39.90 | 46.62 | 50.38 | 51.13 | 50.38 | 47.40 | 57.00 | 46.60 | **50.40** |
| ***Agentic*** | | | | | | | | | | |
| Semiconductor Agent | 45.22 | 28.80 | 39.20 | 33.80 | 22.70 | 57.29 | - | - | 87.25 | **62.41** |
| GameCraft-Bench | 32.10 | - | - | - | 30.65 | 2.15 | - | - | 60.50 | **50.50** |
| PCB Agent | 60.00 | - | 73.30 | 56.70 | 66.70 | 80.00 | - | - | 86.70 | **83.30** |
| ***General*** | | | | | | | | | | |
| HMMT | 81.82 | 57.58 | 87.88 | 87.88 | 75.76 | 90.91 | 78.80 | 90.00 | 93.94 | **87.90** |
| MMLU-Pro | 83.60 | 83.50 | 83.27 | 87.20 | 87.30 | 88.40 | - | - | 88.90 | **84.20** |

## Quick Start

### 1. Install

```bash
git clone https://github.com/Zhiman-BJ/Micius-Flash.git && cd Micius-Flash
pip install -e .                 # evaluation toolkit (pure Python, no GPU needed)
pip install "sglang[all]"        # or: pip install vllm   -- for serving
```

### 2. Serve the model

```bash
MODEL_PATH=zhiman-ai/Micius-Flash TP=8 bash inference/serve_sglang.sh     # OpenAI API on :30000
# or
MODEL_PATH=zhiman-ai/Micius-Flash TP=8 bash inference/serve_vllm.sh
```

Both scripts enable the `qwen3` reasoning parser and the `qwen3_coder` tool-call parser, so reasoning is returned in `reasoning_content` and tool calls in `tool_calls`.

### 3. Chat

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="Micius-Flash",
    messages=[{"role": "user", "content": "比较 FinFET 与平面 MOSFET 在短沟道效应上的差异。"}],
    temperature=1.0, top_p=0.95,
    extra_body={"top_k": 20, "chat_template_kwargs": {"reasoning_effort": "medium"}},
)
print(response.choices[0].message.reasoning_content)   # thinking
print(response.choices[0].message.content)             # answer
```

`reasoning_effort` ∈ {`low`, `medium`, `xhigh`} (default `xhigh`) controls how much the model thinks, and `enable_thinking=False` turns thinking off. Recommended sampling: `temperature=1.0, top_p=0.95, top_k=20`. See [`inference/chat_example.py`](inference/chat_example.py) for multi-turn tool calling and image input, and [`inference/hf_generate.py`](inference/hf_generate.py) for serverless Transformers inference. More in [docs/inference.md](docs/inference.md).

## Micius Benchmarks

Three benchmarks were built for the report. Each has a one-click script that takes any OpenAI-compatible endpoint.

| Benchmark | What it measures | Size | Metric | Needs |
|---|---|---|---|---|
| [**SemiBench**](docs/semibench.md) | Semiconductor knowledge & reasoning, single/multi-select (up to 10 options) | 167 | exact-set accuracy | model endpoint |
| [**Semiconductor Agent**](docs/semiconductor_agent.md) | Tool use in 197 executable semiconductor environments (process, device, materials, EDA, metrology) | 197 | pass rate (frozen-rubric verifier) | data download, Codex CLI + verifier key |
| [**PCB Agent**](docs/pcb_agent.md) | End-to-end KiCad schematic / PCB design and repair | 30 | all-three-axes pass rate | Docker, Codex CLI + verifier key |

```bash
export BASE_URL=http://<host>:30000/v1 MODEL=Micius-Flash

# SemiBench -- a few minutes
bash scripts/run_semibench.sh

# Semiconductor Agent -- download environments once, then run + verify
bash scripts/download_data.sh semiconductor_agent
export MICIUS_DATA=$PWD/data OPENAI_API_KEY=<verifier key>
bash scripts/run_semiconductor_agent.sh

# PCB Agent -- each task runs in its own container
bash scripts/run_pcb_agent.sh

python -m micius_eval.report outputs      # one table for all runs
```

Agentic benchmarks separate **generation** (the evaluated model solves tasks) from **verification** (an independent verifier agent grades the final state and answer). Both stages are resumable; re-running a script skips finished tasks.

## Repository Layout

```
Micius-Flash/
├── inference/                 # serve_sglang.sh, serve_vllm.sh, chat_example.py, hf_generate.py
├── micius_eval/
│   ├── semibench/             # prompt, answer extraction, scoring
│   ├── semiconductor_agent/   # agent loop, read-only file tools, runtime setup, frozen-rubric verifier
│   │   └── _vendor/agent_world/   # environment harness + verifier used to build the benchmark
│   ├── pcb_agent/             # containerized generation + three-axis KiCad review
│   └── report.py
├── model_card/                # Hugging Face model card
├── data/                      # semibench/, pcb_agent/  (+ semiconductor_agent/ after download)
├── scripts/                   # one-click entry points
├── docker/                    # portable evaluation image
└── docs/
```

## Citation

```bibtex
@misc{micius2026flash,
  title         = {Micius-Flash: Lightweight Agentic Intelligence for Engineering},
  author        = {Micius Team},
  year          = {2026},
  eprint        = {2611.10015},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  url           = {https://arxiv.org/abs/2611.10015}
}
```

## License

Code in this repository is released under the [Apache License 2.0](LICENSE). Benchmark data are provided for research evaluation; please do not use them for training.
