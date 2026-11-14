"""Talk to a served Micius-Flash through the OpenAI SDK.

    pip install openai
    python inference/chat_example.py --base-url http://127.0.0.1:30000/v1 --model Micius-Flash

Covers: thinking with a selectable effort, non-thinking mode, multi-turn tool
calling and image input. ``reasoning_effort`` (low | medium | xhigh, default
xhigh) sets how much the model thinks; ``enable_thinking=False`` turns thinking
off. The reasoning is returned separately in ``message.reasoning_content`` when
the server runs a reasoning parser.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from openai import OpenAI

# Recommended sampling (also shipped in generation_config.json).
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "extra_body": {"top_k": 20}}


def effort(level: str) -> dict:
    return {"chat_template_kwargs": {"reasoning_effort": level}}


NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def thinking_demo(client: OpenAI, model: str) -> None:
    print("=== 1. Thinking (reasoning_effort=medium) ===")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "一个 NMOS 的阈值电压随衬底掺杂浓度如何变化？请简要说明物理原因。"}],
        max_tokens=8192,
        temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
        extra_body={**SAMPLING["extra_body"], **effort("medium")},
    )
    message = response.choices[0].message
    reasoning = getattr(message, "reasoning_content", None) or ""
    print(f"[reasoning: {len(reasoning)} chars]\n{message.content}\n")


def no_thinking_demo(client: OpenAI, model: str) -> None:
    print("=== 2. Non-thinking mode ===")
    response = client.chat.completions.create(
        model=model, max_tokens=1024,
        messages=[{"role": "user", "content": "用一句话说明 CMOS 反相器为什么静态功耗低。"}],
        temperature=0.7, top_p=0.8, extra_body={"top_k": 20, **NO_THINKING},
    )
    print(response.choices[0].message.content, "\n")


TOOLS = [{
    "type": "function",
    "function": {
        "name": "mosfet_saturation_current",
        "description": "Square-law saturation drain current I_D = 0.5 * k_n * (W/L) * (V_GS - V_TH)^2 * (1 + lambda * V_DS).",
        "parameters": {
            "type": "object",
            "properties": {
                "k_n_uA_per_V2": {"type": "number", "description": "Process transconductance, uA/V^2"},
                "w_over_l": {"type": "number"},
                "v_gs": {"type": "number"},
                "v_th": {"type": "number"},
                "v_ds": {"type": "number"},
                "channel_length_modulation": {"type": "number", "default": 0.0},
            },
            "required": ["k_n_uA_per_V2", "w_over_l", "v_gs", "v_th", "v_ds"],
        },
    },
}]


def mosfet_saturation_current(k_n_uA_per_V2, w_over_l, v_gs, v_th, v_ds, channel_length_modulation=0.0):
    vov = max(v_gs - v_th, 0.0)
    i_d = 0.5 * k_n_uA_per_V2 * 1e-6 * w_over_l * vov ** 2 * (1 + channel_length_modulation * v_ds)
    return {"i_d_A": i_d, "v_ov": vov, "saturated": v_ds >= vov}


def tool_demo(client: OpenAI, model: str) -> None:
    print("=== 3. Tool calling ===")
    messages = [{"role": "user", "content":
                 "k'n=200 uA/V^2, W/L=10, VTH=0.45 V, VGS=1.0 V, VDS=1.2 V, lambda=0.05 1/V。用工具算漏极电流，并判断是否饱和。"}]
    for _ in range(4):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, max_tokens=8192,
            temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
            extra_body={**SAMPLING["extra_body"], **effort("low")},
        )
        message = response.choices[0].message
        assistant = {"role": "assistant", "content": message.content or ""}
        # Keep the reasoning in the history so interleaved thinking stays coherent.
        if getattr(message, "reasoning_content", None):
            assistant["reasoning_content"] = message.reasoning_content
        if message.tool_calls:
            assistant["tool_calls"] = [c.model_dump() for c in message.tool_calls]
        messages.append(assistant)
        if not message.tool_calls:
            print(message.content, "\n")
            return
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            result = mosfet_saturation_current(**args)
            print(f"  -> {call.function.name}({args}) = {result}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})


def image_demo(client: OpenAI, model: str, image: Path) -> None:
    print("=== 4. Image input ===")
    data = base64.b64encode(image.read_bytes()).decode()
    mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
    response = client.chat.completions.create(
        model=model, max_tokens=8192,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
            {"type": "text", "text": "描述这张电路/版图图片中的关键元件和连接关系。"},
        ]}],
        temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
        extra_body={**SAMPLING["extra_body"], **effort("low")},
    )
    print(response.choices[0].message.content, "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model", default="Micius-Flash")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--image", type=Path, default=Path(__file__).with_name("assets") / "inverter.png")
    a = ap.parse_args()
    client = OpenAI(base_url=a.base_url, api_key=a.api_key, timeout=1800)
    thinking_demo(client, a.model)
    no_thinking_demo(client, a.model)
    tool_demo(client, a.model)
    if a.image.is_file():
        image_demo(client, a.model, a.image)


if __name__ == "__main__":
    main()
