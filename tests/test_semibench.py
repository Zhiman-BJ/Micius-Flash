import json
from pathlib import Path

from micius_eval.semibench.run import DEFAULT_DATA, build_prompt, extract_answer


def test_data_integrity():
    rows = [json.loads(l) for l in Path(DEFAULT_DATA).read_text().splitlines()]
    assert len(rows) == 167 and len({r["id"] for r in rows}) == 167
    for r in rows:
        assert r["answer"] and set(r["answer"]) <= set(r["options"])
        assert r["question_type"] != "single_choice" or len(r["answer"]) == 1


def test_prompt_contract():
    row = {"question_type": "multiple_choice", "question": "Q", "options": {"A": "a", "B": "b", "C": "c"}}
    prompt = build_prompt(row)
    assert "本题为多选题" in prompt and "A–C" in prompt and prompt.endswith("C. c")


def test_extraction():
    ok = [("推理……\n答案：B", "single_choice", ["B"]), ("x\n答案：A、C、F", "multiple_choice", ["A", "C", "F"]),
          ("**答案：DCB**", "multiple_choice", ["B", "C", "D"]), ("final answer: (C)", "single_choice", ["C"]),
          ("<think>a</think>\n答案：A", "single_choice", ["A"])]
    for content, qtype, expected in ok:
        assert extract_answer(content, "stop", qtype) == (expected, None), content
    bad = [("答案：A", "length", "single_choice", "truncated_response"),
           ("答案：AB", "stop", "single_choice", "multiple_answers_for_single_choice"),
           ("答案：AA", "stop", "multiple_choice", "duplicate_answer_letters"),
           ("答案是A，理由如下", "stop", "single_choice", "missing_or_ambiguous_final_answer"),
           ("", "stop", "single_choice", "missing_final_answer")]
    for content, finish, qtype, reason in bad:
        assert extract_answer(content, finish, qtype) == (None, reason), content
