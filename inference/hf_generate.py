"""Offline generation with Hugging Face Transformers (no server).

    pip install "transformers>=5.15" torch accelerate
    python inference/hf_generate.py --model zhiman-ai/Micius-Flash \
        --prompt "解释 FinFET 相比平面 MOSFET 抑制短沟道效应的原因。" --reasoning-effort low

Use SGLang / vLLM (see serve_*.sh) for throughput, long context and tool use;
this script is meant for quick checks and debugging.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="用三句话解释 MOSFET 的阈值电压。")
    ap.add_argument("--image", default=None, help="Optional image path or URL.")
    ap.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "xhigh"])
    ap.add_argument("--no-thinking", action="store_true", help="Answer directly (enable_thinking=False).")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--device-map", default="auto", help="'auto' for GPUs, 'cpu' to run on CPU.")
    ap.add_argument("--greedy", action="store_true", help="Greedy decoding instead of the recommended sampling.")
    a = ap.parse_args()

    processor = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map=a.device_map)

    content = [{"type": "text", "text": a.prompt}]
    if a.image:
        content.insert(0, {"type": "image", "url": a.image})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        reasoning_effort=a.reasoning_effort, enable_thinking=not a.no_thinking,
    ).to(model.device)

    sampling = {"do_sample": False} if a.greedy else {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20}
    started = time.time()
    output = model.generate(**inputs, max_new_tokens=a.max_new_tokens, **sampling)
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    text = processor.decode(new_tokens, skip_special_tokens=True)
    # The reasoning comes first and is closed by </think>; the answer follows.
    reasoning, _, answer = text.rpartition("</think>")
    print("----- reasoning -----\n" + reasoning.replace("<think>", "").strip())
    print("----- answer -----\n" + (answer.strip() or text.strip()))
    print(f"\n[{len(new_tokens)} tokens in {time.time() - started:.1f}s]")


if __name__ == "__main__":
    main()
