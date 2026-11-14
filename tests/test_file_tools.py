import pytest

from micius_eval.semiconductor_agent.file_tools import WorkspaceFiles


@pytest.fixture()
def ws(tmp_path):
    (tmp_path / "eos").mkdir()
    (tmp_path / "eos" / "data.json").write_text('{"a": 1}\n{"b": 2}\n')
    (tmp_path / "notes.md").write_text("alpha\nbeta stress\ngamma\n")
    (tmp_path / ".env").write_text("SECRET=1")
    return WorkspaceFiles(tmp_path)


def test_read_and_paths(ws):
    out, err = ws.call("Read", {"path": "/workspace/notes.md", "line_offset": 2, "n_lines": 1})
    assert not err and out["output"] == "2\tbeta stress"
    out, err = ws.call("Read", {"path": "../etc/passwd"})
    assert err
    out, err = ws.call("Read", {"path": ".env"})
    assert err


def test_glob_grep(ws):
    out, _ = ws.call("Glob", {"pattern": "*.json"})
    assert out["output"] == "eos/data.json"
    out, _ = ws.call("Grep", {"pattern": "stress", "output_mode": "content"})
    assert out["output"] == "notes.md:2:beta stress"
    out, _ = ws.call("Grep", {"pattern": "\"b\""})
    assert out["output"] == "eos/data.json"


def test_listing(ws):
    assert ws.listing() == "├── eos/\n│   └── data.json\n├── .env\n└── notes.md\n"
