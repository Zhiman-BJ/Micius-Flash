# Inference

## Model

| | |
|---|---|
| Hugging Face | [`zhiman-ai/Micius-Flash`](https://huggingface.co/zhiman-ai/Micius-Flash) |
| Architecture | `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`), vision-language |
| Parameters | 27.4B (BF16 checkpoint, 54.7 GB, 11 safetensors shards) |
| Language model | 64 layers, hidden size 5,120, FFN 17,408; hybrid attention: 48 linear-attention layers + 16 full-attention layers (one every 4 layers; 24 query / 4 KV heads, head dim 256, output gate) |
| Vision encoder | 27 layers, hidden size 1,152, patch 16, 2×2 spatial merge; images and video |
| Position encoding | interleaved M-RoPE, θ = 10⁷, partial rotary 0.25 |
| Vocabulary | 248,320 |
| Context | 262,144 tokens |
| Chat format | ChatML; reasoning in `<think>…</think>`; tool calls in Qwen3-Coder XML (`<tool_call><function=…><parameter=…>`) |
| Recommended sampling | `temperature=1.0, top_p=0.95, top_k=20` (shipped in `generation_config.json`) |

### Thinking control

Pass these through `chat_template_kwargs` (OpenAI API) or as keyword arguments of
`apply_chat_template` (Transformers):

| kwarg | values | effect |
|---|---|---|
| `reasoning_effort` | `xhigh` (default), `medium`, `low` | `xhigh` and `low` add a one-line system instruction on how thoroughly to think; `medium` adds none |
| `enable_thinking` | `true` (default), `false` | `false` pre-fills an empty `<think></think>` block, so the model answers directly |
| `preserve_thinking` | `true` (default), `false` | whether reasoning of earlier turns is kept in the prompt; the current turn's reasoning is always kept |

All benchmark numbers in the report use thinking with `xhigh`.

## Serving

```bash
MODEL_PATH=zhiman-ai/Micius-Flash TP=8 bash inference/serve_sglang.sh   # port 30000
MODEL_PATH=zhiman-ai/Micius-Flash TP=8 bash inference/serve_vllm.sh
```

Environment variables: `MODEL_PATH` (HF id or local directory), `SERVED_NAME`
(default `Micius-Flash`), `TP`, `PORT`, `CONTEXT`, `MEM_FRACTION` (SGLang).
Extra arguments are passed through, e.g. `bash inference/serve_sglang.sh --dp 2`.

Both scripts enable the `qwen3` reasoning parser (reasoning returned in
`message.reasoning_content`) and the `qwen3_coder` tool-call parser. Weights take
~55 GB: `TP=1` runs on one 80 GB GPU with a reduced `CONTEXT`; use `TP=4` or `8`
for the full 256K context and agentic workloads. Agent benchmarks in this
repository were run with SGLang, `--tp 8 --context-length 262144`.

## Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="Micius-Flash", messages=messages, tools=tools,
    temperature=1.0, top_p=0.95,
    extra_body={"top_k": 20, "chat_template_kwargs": {"reasoning_effort": "xhigh"}},
)
msg = response.choices[0].message   # msg.reasoning_content, msg.content, msg.tool_calls
```

For multi-turn tool use, append each assistant message **with its
`reasoning_content`** before the tool results. A runnable example covering
thinking efforts, non-thinking mode, a tool-calling loop and image input:

```bash
python inference/chat_example.py --base-url http://127.0.0.1:30000/v1 --model Micius-Flash
```

## Transformers

```bash
pip install "transformers>=5.15" torch accelerate pillow
python inference/hf_generate.py --model zhiman-ai/Micius-Flash --reasoning-effort low \
    --prompt "解释 FinFET 相比平面 MOSFET 抑制短沟道效应的原因。"
python inference/hf_generate.py --model zhiman-ai/Micius-Flash --image inference/assets/inverter.png \
    --prompt "这是什么电路？"
python inference/hf_generate.py --model zhiman-ai/Micius-Flash --no-thinking --prompt "1+1=?"
```

Use `--device-map cpu` to run without a GPU (slow, but useful to check a download).

## Notes

* `max_tokens` includes reasoning. At `xhigh` hard problems can think for tens of
  thousands of tokens; budget accordingly (SemiBench uses 32,768, agents 131,072 per turn).
* If `ALL_PROXY` points to a SOCKS proxy, the OpenAI SDK needs
  `pip install "httpx[socks]"`, or unset the proxy variables for local endpoints.
