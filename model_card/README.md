---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
language:
  - zh
  - en
tags:
  - engineering
  - semiconductor
  - eda
  - hardware-design
  - agent
  - reasoning
---

<div align="center">

# Micius-Flash

**Lightweight Agentic Intelligence for Engineering**

<a href="https://arxiv.org/abs/2611.10015">📄 Technical Report</a> &nbsp;|&nbsp;
<a href="https://github.com/Zhiman-BJ/Micius-Flash">💻 GitHub</a> &nbsp;|&nbsp;
<a href="https://huggingface.co/datasets/zhiman-ai/Micius-Flash-Bench">📊 Benchmarks</a>

</div>

## Highlights

**Micius-Flash** is a 27B-parameter model for engineering with native multimodal input, deep reasoning and a 256K-token context window, trained with Multi-Teacher On-Policy Distillation (MOPD) from four domain experts (knowledge & reasoning, hardware coding, multimodal, agentic).

- **Hardware coding:** 64.2% average over 12 hardware-code benchmarks, +15.0 points over a same-size baseline.
- **Engineering agents:** 62.4% on Semiconductor Agent and 83.3% on PCB Agent.
- **Circuit vision:** 72.4% on RFSchemBench and 71.0% on CircuitSense.

## Model Overview

| | |
|---|---|
| Type | vision-language causal LM (`Qwen3_5ForConditionalGeneration`) |
| Parameters | 27.4B, BF16 |
| Layers | 64 (48 linear-attention + 16 full-attention, one every 4) |
| Hidden size / FFN | 5,120 / 17,408 |
| Full attention | 24 query heads, 4 KV heads, head dim 256 |
| Vision encoder | 27 layers, patch 16 |
| Context length | 262,144 tokens |
| Vocabulary | 248,320 |

## Selected Results

| Benchmark | Qwen3.8-27B | DeepSeek V4-Pro | GPT-5.6 Sol | **Micius-Flash** |
|---|---:|---:|---:|---:|
| SemiBench | 42.51 | 67.66 | 88.02 | **49.70** |
| SuperGPQA (Microelec) | 76.00 | 82.00 | 78.00 | **84.00** |
| RFSchemBench | 66.80 | - | 50.84 | **72.40** |
| CircuitSense | 69.80 | - | 60.36 | **71.00** |
| VerilogEval-v2 | 77.80 | 89.71 | 94.20 | **86.80** |
| RTLLM-v2 | 65.10 | 79.07 | 79.10 | **79.10** |
| NetlistBench | 69.80 | 77.97 | 75.60 | **83.80** |
| Semiconductor Agent | 45.22 | 57.29 | 87.25 | **62.41** |
| PCB Agent | 60.00 | 80.00 | 86.70 | **83.30** |
| HMMT | 81.82 | 90.91 | 93.94 | **87.90** |

Full results for 25 benchmarks and 10 models are in the [technical report](https://arxiv.org/abs/2611.10015) and on [GitHub](https://github.com/Zhiman-BJ/Micius-Flash#evaluation-results).

## Quickstart

### Serving (recommended)

```bash
# SGLang
python -m sglang.launch_server --model-path zhiman-ai/Micius-Flash --served-model-name Micius-Flash \
  --tp 8 --context-length 262144 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --port 30000

# vLLM
vllm serve zhiman-ai/Micius-Flash --served-model-name Micius-Flash --tensor-parallel-size 8 \
  --max-model-len 262144 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --port 30000
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="Micius-Flash",
    messages=[{"role": "user", "content": "比较 FinFET 与平面 MOSFET 在短沟道效应上的差异。"}],
    max_tokens=32768, temperature=1.0, top_p=0.95,
    extra_body={"top_k": 20, "chat_template_kwargs": {"reasoning_effort": "xhigh"}},
)
print(response.choices[0].message.reasoning_content)
print(response.choices[0].message.content)
```

### Transformers

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "zhiman-ai/Micius-Flash"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

messages = [{"role": "user", "content": [
    {"type": "image", "url": "https://example.com/schematic.png"},
    {"type": "text", "text": "分析这张原理图的功能。"},
]}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt",
                                       reasoning_effort="medium").to(model.device)
out = model.generate(**inputs, max_new_tokens=8192, do_sample=True, temperature=1.0, top_p=0.95, top_k=20)
print(processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Requires `transformers>=5.15`.

## Thinking and Tool Use

| `chat_template_kwargs` | values | effect |
|---|---|---|
| `reasoning_effort` | `xhigh` (default), `medium`, `low` | how thoroughly the model thinks |
| `enable_thinking` | `true` (default), `false` | `false` answers directly without reasoning |
| `preserve_thinking` | `true` (default), `false` | keep reasoning of earlier turns in the prompt |

Tools follow the OpenAI function-calling schema; the template renders them in the Qwen3-Coder XML format, parsed by `--tool-call-parser qwen3_coder`. In multi-turn agent loops, send each assistant message back **with its `reasoning_content`**.

**Best practices**

- Sampling: `temperature=1.0`, `top_p=0.95`, `top_k=20`.
- Output length: `max_tokens` includes reasoning; allow 32K tokens for hard questions and up to 128K per turn for agents.
- Use `xhigh` for difficult reasoning, coding and agentic tasks; `low` or `enable_thinking=false` for quick replies.

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
