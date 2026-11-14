import json

from micius_eval.semiconductor_agent.agent import AgentConfig, run_agent
from micius_eval.semiconductor_agent.file_tools import WorkspaceFiles


class FakeClient:
    """Calls the business tool until refused, reads a file, then answers."""

    def __init__(self):
        self.requests = []

    def chat(self, messages, **params):
        self.requests.append((messages, params))
        n = len(self.requests)
        if n <= 3:
            call = {"id": f"c{n}", "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"key": n})}}
            msg = {"content": "", "reasoning_content": "think", "tool_calls": [call]}
        elif n == 4:
            call = {"id": "r", "type": "function", "function": {"name": "Read", "arguments": '{"path": "a.txt"}'}}
            msg = {"content": "", "tool_calls": [call]}
        else:
            msg = {"content": "final answer"}
        return {"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def test_loop(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    seen = []

    def call_env_tool(name, arguments):
        seen.append(arguments["key"])
        return {"success": True, "data": arguments}, False

    client = FakeClient()
    result = run_agent(client=client, task_text="do it", files=WorkspaceFiles(tmp_path),
                       env_tools=[{"name": "lookup", "description": "d", "inputSchema": {"type": "object"}}],
                       call_env_tool=call_env_tool, config=AgentConfig(max_tool_calls=2))
    assert result.status == "completed" and result.final_answer == "final answer"
    assert seen == [1, 2] and result.environment_calls == 2 and result.native_calls == 1
    refused = [m for m in result.messages if m["role"] == "tool" and "limit reached" in m["content"]]
    assert len(refused) == 1
    first = client.requests[0][0]
    assert first[0]["role"] == "system" and "a.txt" in first[0]["content"] and first[1]["content"] == "do it"
    assert {t["function"]["name"] for t in client.requests[0][1]["tools"]} == {"lookup", "Read", "Grep", "Glob"}
    assert any(m.get("reasoning_content") == "think" for m in result.messages if m["role"] == "assistant")
