#!/usr/bin/env python3
"""[Maintainers] Assemble the Hugging Face model repository from a trained checkpoint.

    python tools/release/prepare_hf_model.py --checkpoint <.../actor/huggingface> --output release/Micius-Flash
    hf upload zhiman-ai/Micius-Flash release/Micius-Flash . --repo-type model

Copies weights, tokenizer, processor and chat template; writes the model card
(model_card/README.md) and the recommended generation_config.json; then checks
that every shard listed in the index exists and that the chat template renders
all thinking modes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REQUIRED = ["config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
            "preprocessor_config.json", "processor_config.json", "model.safetensors.index.json"]
OPTIONAL = ["video_preprocessor_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]
GENERATION_CONFIG = {
    "bos_token_id": 248044, "pad_token_id": 248044, "eos_token_id": [248046, 248044],
    "do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--link", action="store_true", help="hardlink weight shards instead of copying")
    a = ap.parse_args()
    src, out = a.checkpoint.resolve(), a.output.resolve()
    missing = [f for f in REQUIRED if not (src / f).is_file()]
    if missing:
        sys.exit(f"checkpoint is missing {missing}")
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    for name in REQUIRED + [f for f in OPTIONAL if (src / f).is_file()]:
        shutil.copy2(src / name, out / name)
    for shard in shards:
        target = out / shard
        if target.exists() and target.stat().st_size == (src / shard).stat().st_size:
            continue
        if a.link:
            target.unlink(missing_ok=True)
            target.hardlink_to(src / shard)
        else:
            shutil.copyfile(src / shard, target)
        print(f"  {shard}", flush=True)
    generation = json.loads((src / "generation_config.json").read_text()) if (src / "generation_config.json").is_file() else {}
    generation.update(GENERATION_CONFIG)
    (out / "generation_config.json").write_text(json.dumps(generation, indent=2) + "\n")
    shutil.copy2(REPO / "model_card" / "README.md", out / "README.md")
    shutil.copy2(REPO / "LICENSE", out / "LICENSE")

    # checks
    size = sum((out / s).stat().st_size for s in shards)
    declared = index.get("metadata", {}).get("total_size")
    print(f"{len(shards)} shards, {size / 1e9:.1f} GB on disk (index total_size {declared})")
    if declared and size < int(declared):
        print("WARNING: shards are smaller than the tensor bytes declared in the index (incomplete copy?)")
    from transformers import AutoTokenizer  # noqa: E402

    tok = AutoTokenizer.from_pretrained(out)
    msgs = [{"role": "user", "content": "hi"}]
    for kwargs in ({}, {"reasoning_effort": "low"}, {"reasoning_effort": "medium"}, {"enable_thinking": False}):
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs)
        assert rendered.rstrip().endswith(("<think>", "</think>")), kwargs
    manifest = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                for name in REQUIRED + ["generation_config.json"] if (out / name).is_file()}
    out.with_name(out.name + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"ready: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
