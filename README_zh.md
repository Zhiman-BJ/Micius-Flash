<div align="center">

# Micius-Flash

**面向工程的轻量级智能体模型**

<a href="https://arxiv.org/abs/2611.10015"><b>📄 技术报告</b></a> &nbsp;|&nbsp;
<a href="https://huggingface.co/zhiman-ai/Micius-Flash"><b>🤗 模型</b></a> &nbsp;|&nbsp;
<a href="https://huggingface.co/datasets/zhiman-ai/Micius-Flash-Bench"><b>📊 评测集</b></a> &nbsp;|&nbsp;
<a href="README.md"><b>English</b></a>

</div>

## 简介

**Micius-Flash** 是一个面向工程领域的 27B 参数模型，原生支持多模态输入、深度推理与 256K 上下文。我们以知识树驱动的数据飞轮构造训练数据，并采用 **多教师在线策略蒸馏（Multi-Teacher On-Policy Distillation, MOPD）**：先通过 SFT 训练知识与推理、硬件编程、多模态、智能体四个领域教师，再让学生模型在自身生成的轨迹上，按任务类别接受对应教师的逐 token 监督（保留 top-32 的前向 KL），把各教师能力整合到一个模型中。

- **硬件编程**：12 个硬件代码基准平均 64.2%，比同尺寸基线高 15.0 分；RTLLM-v2 与最强模型并列第一。
- **工程智能体**：Semiconductor Agent 62.4%（比同尺寸基线高 17.2 分），PCB Agent 83.3%，在已评测模型中均仅次于 GPT-5.6 Sol。
- **电路视觉**：RFSchemBench（72.4%）与 CircuitSense（71.0%）为已报告结果中最高。

| | |
|---|---|
| 参数量 | 27.4B（BF16），视觉-语言模型 |
| 结构 | 64 层，混合注意力（48 层线性注意力 + 16 层全注意力），27 层视觉编码器 |
| 上下文 | 262,144 tokens |
| 思考控制 | `reasoning_effort` = `low` / `medium` / `xhigh`，或 `enable_thinking=false` 关闭思考 |
| 工具调用 | OpenAI function calling 接口（底层为 Qwen3-Coder XML 格式） |
| 许可 | Apache 2.0 |

详见 [docs/inference.md](docs/inference.md)。本仓库包含：

| | |
|---|---|
| [`inference/`](inference) | SGLang / vLLM 部署脚本，OpenAI 兼容客户端示例（思考档位、工具调用、图像输入），Transformers 离线推理 |
| [`micius_eval/`](micius_eval) | 技术报告中新构建的三个评测：**SemiBench**、**Semiconductor Agent**、**PCB Agent** |
| [`data/`](data) | SemiBench（167 题）与 PCB Agent（30 题）输入；Semiconductor Agent 环境需单独下载 |

## 动态

- **2026-11** — 在 [Hugging Face](https://huggingface.co/zhiman-ai/Micius-Flash) 发布 Micius-Flash 模型权重，同时发布技术报告、推理代码与三个 Micius 评测。

## 评测结果

完整结果见 [README.md](README.md#evaluation-results)。三个自建评测：

| 评测 | Qwen3.8-27B | DeepSeek V4-Pro | GPT-5.6 Sol | **Micius-Flash** |
|---|---:|---:|---:|---:|
| SemiBench | 42.51 | 67.66 | 88.02 | **49.70** |
| Semiconductor Agent | 45.22 | 57.29 | 87.25 | **62.41** |
| PCB Agent | 60.00 | 80.00 | 86.70 | **83.30** |

## 快速开始

```bash
git clone https://github.com/Zhiman-BJ/Micius-Flash.git && cd Micius-Flash
pip install -e .

# 部署（OpenAI 兼容接口，端口 30000）
MODEL_PATH=zhiman-ai/Micius-Flash TP=8 bash inference/serve_sglang.sh
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="Micius-Flash",
    messages=[{"role": "user", "content": "比较 FinFET 与平面 MOSFET 在短沟道效应上的差异。"}],
    temperature=1.0, top_p=0.95,
    extra_body={"top_k": 20, "chat_template_kwargs": {"reasoning_effort": "medium"}},
)
print(response.choices[0].message.reasoning_content)   # 思考过程
print(response.choices[0].message.content)             # 最终回答
```

`reasoning_effort` 取 `low` / `medium` / `xhigh`（默认）控制思考深度，`enable_thinking=False` 可关闭思考。推荐采样参数 `temperature=1.0, top_p=0.95, top_k=20`。多轮工具调用、图像输入示例见 [`inference/chat_example.py`](inference/chat_example.py)，详细说明见 [docs/inference.md](docs/inference.md)。

## Micius 评测集

| 评测 | 考察内容 | 规模 | 指标 | 依赖 |
|---|---|---|---|---|
| [**SemiBench**](docs/semibench.md) | 半导体领域知识与推理，单选/多选（最多 10 个选项） | 167 | 答案集合完全匹配准确率 | 模型服务 |
| [**Semiconductor Agent**](docs/semiconductor_agent.md) | 在 197 个可执行半导体环境（工艺、器件、材料、EDA、量测）中调用工具完成任务 | 197 | 通过率（冻结 rubric 验收） | 下载环境数据，Codex CLI + 验收模型 key |
| [**PCB Agent**](docs/pcb_agent.md) | 基于 KiCad 的原理图 / PCB 端到端设计与修复 | 30 | 三轴（交付/结构/功能）全部通过率 | Docker，Codex CLI + 验收模型 key |

```bash
export BASE_URL=http://<host>:30000/v1 MODEL=Micius-Flash

bash scripts/run_semibench.sh                              # SemiBench，数分钟

bash scripts/download_data.sh semiconductor_agent          # 首次：下载并配置环境
export MICIUS_DATA=$PWD/data OPENAI_API_KEY=<验收模型 key>
bash scripts/run_semiconductor_agent.sh                    # 生成 + 验收

bash scripts/run_pcb_agent.sh                              # 每题一个容器

python -m micius_eval.report outputs                       # 汇总成一张表
```

智能体评测分为 **生成**（被测模型解题）与 **验收**（独立验收智能体检查最终状态与回答）两个阶段，均支持断点续跑。

## 引用

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

## 许可

代码采用 [Apache License 2.0](LICENSE)。评测数据仅供研究评测使用，请勿用于训练。
