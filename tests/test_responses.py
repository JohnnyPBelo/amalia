"""Tests for the Responses-API worker path (gpt-5.x). No network — pure parsing."""
import pytest
from amalia.workers import Worker, _extract_responses_text


def test_worker_rejects_bad_api_type():
    with pytest.raises(ValueError):
        Worker("x", "m", "http://x/v1", api_type="grpc")


def test_extract_from_output_text_string():
    assert _extract_responses_text({"output_text": "hello"}) == "hello"


def test_extract_from_output_text_list():
    assert _extract_responses_text({"output_text": ["a", "b"]}) == "ab"


def test_extract_walks_message_blocks_and_skips_reasoning():
    data = {
        "output": [
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "thinking..."}]},
            {"type": "message", "content": [
                {"type": "output_text", "text": "the answer is 391"},
            ]},
        ]
    }
    assert _extract_responses_text(data) == "the answer is 391"


def test_extract_empty_when_no_text():
    assert _extract_responses_text({"output": [{"type": "reasoning", "content": []}]}) == ""


def test_extract_prefers_output_text_over_blocks():
    data = {"output_text": "fast path", "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "slow path"}]}]}
    assert _extract_responses_text(data) == "fast path"
