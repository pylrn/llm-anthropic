import json

import anthropic
import httpx2
import llm
import pytest
from llm.parts import ToolResultPart

import llm_anthropic


def _stream_response(model, blocks):
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        }
    ]
    for index, block in enumerate(blocks):
        start_block = dict(block)
        delta = None
        if block["type"] == "text":
            start_block["text"] = ""
            delta = {"type": "text_delta", "text": block["text"]}
        elif block["type"] == "mcp_tool_use":
            start_block["input"] = {}
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"]),
            }
        events.append(
            {
                "type": "content_block_start",
                "index": index,
                "content_block": start_block,
            }
        )
        if delta is not None:
            events.append(
                {"type": "content_block_delta", "index": index, "delta": delta}
            )
        events.append({"type": "content_block_stop", "index": index})
    events.extend(
        [
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        ]
    )
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(
            "event: {}\ndata: {}\n\n".format(event["type"], json.dumps(event))
            for event in events
        ).encode(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "buffer"])
@pytest.mark.parametrize(
    "content",
    [
        "A result with a newline\nand Unicode: café",
        "",
        '{"answer": 42}',
        "null",
        [{"type": "text", "text": "List result", "citations": None}],
    ],
    ids=["text", "empty", "json-object-text", "json-null-text", "list"],
)
async def test_mcp_result_content_round_trips(
    monkeypatch, asynchronous, stream, content
):
    requests = []
    tool_call = {
        "type": "mcp_tool_use",
        "id": "mcptoolu_test",
        "name": "lookup",
        "server_name": "test-server",
        "input": {"query": "example"},
    }
    tool_result = {
        "type": "mcp_tool_result",
        "tool_use_id": "mcptoolu_test",
        "is_error": False,
        "content": content,
    }

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        blocks = (
            [tool_call, tool_result, {"type": "text", "text": "Done"}]
            if len(requests) == 1
            else [{"type": "text", "text": "Follow-up"}]
        )
        return _stream_response(body["model"], blocks)

    transport = httpx2.MockTransport(handle)
    tools = [
        llm_anthropic.AnthropicMCP(
            url="https://mcp.example.com/mcp", name="test-server"
        )
    ]
    if asynchronous:
        async with anthropic.AsyncAnthropic(
            api_key="sk-test", http_client=httpx2.AsyncClient(transport=transport)
        ) as client:
            monkeypatch.setattr(
                llm_anthropic, "AsyncAnthropic", lambda **kwargs: client
            )
            conversation = llm.get_async_model("claude-sonnet-4.6").conversation()
            response = await conversation.prompt(
                "Look this up", tools=tools, stream=stream, key="sk-test"
            )
            assert await response.text() == "Done"
            messages = await response.messages()
            follow_up = await conversation.prompt(
                "Continue", tools=tools, key="sk-test"
            )
            assert await follow_up.text() == "Follow-up"
    else:
        with anthropic.Anthropic(
            api_key="sk-test", http_client=httpx2.Client(transport=transport)
        ) as client:
            monkeypatch.setattr(llm_anthropic, "Anthropic", lambda **kwargs: client)
            conversation = llm.get_model("claude-sonnet-4.6").conversation()
            response = conversation.prompt(
                "Look this up", tools=tools, stream=stream, key="sk-test"
            )
            assert response.text() == "Done"
            messages = response.messages()
            follow_up = conversation.prompt("Continue", tools=tools, key="sk-test")
            assert follow_up.text() == "Follow-up"

    results = [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolResultPart)
    ]
    assert len(results) == 1
    assert results[0].server_executed
    assert results[0].tool_call_id == tool_call["id"]
    assert json.loads(results[0].output) == content
    assert response.response_json["content"][1]["content"] == content
    assert len(requests) == 2
    assert requests[0]["stream"] is True
    history = requests[1]["messages"][1]
    assert history["role"] == "assistant"
    assert history["content"][0] == tool_call
    assert history["content"][1]["type"] == "mcp_tool_result"
    assert history["content"][1]["tool_use_id"] == tool_call["id"]
    assert history["content"][1]["content"] == content
