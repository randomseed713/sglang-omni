# SPDX-License-Identifier: Apache-2.0
"""Native text and beam generation contracts."""

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sglang.srt.managers.schedule_batch import Req

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest
from sglang_omni.models.sid_gr.request_builders import SIDGRRequestBuilder
from sglang_omni.models.sid_gr.stages import (
    create_decode_executor,
    create_preprocessing_executor,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.serve.openai_api import (
    build_chat_generate_request,
    chat_stream_events,
    create_app,
)
from sglang_omni.serve.protocol import ChatCompletionRequest


class Tokenizer:
    eos_token_id = 9
    additional_stop_token_ids = None

    def encode(self, text: str) -> list[int]:
        return [1, 2]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        return_dict: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        assert return_dict is False
        return [1, 2, 3]


def test_beam_parameters_reach_native_request() -> None:
    api_request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        beam_width=4,
        n=3,
        max_tokens=8,
        stop_token_ids=[7],
    )
    request = Client.build_omni_request(build_chat_generate_request(api_request))
    payload = StagePayload(
        request_id="beam", request=request, data={"token_ids": [1, 2]}
    )
    built = SIDGRRequestBuilder(Tokenizer(), vocab_size=10)(payload)
    assert isinstance(built.req, Req)
    assert built.req.sampling_params.beam_width == 4
    assert built.req.sampling_params.n == 3
    assert built.req.sampling_params.max_new_tokens == 8
    assert 7 in built.req.sampling_params.stop_token_ids


@pytest.mark.parametrize(
    "inputs", ["hello", [1, 2], [{"role": "user", "content": "hello"}]]
)
def test_text_inputs_follow_client_contract(
    monkeypatch: pytest.MonkeyPatch, inputs: str | list[int] | list[dict[str, str]]
) -> None:
    monkeypatch.setattr(
        "sglang_omni.models.sid_gr.stages.get_tokenizer", lambda path: Tokenizer()
    )
    executor = create_preprocessing_executor("test-model")
    result = executor.fn(
        StagePayload(request_id="text", request=OmniRequest(inputs=inputs), data=None)
    )
    assert result.data["token_ids"] in ([1, 2], [1, 2, 3])


@pytest.mark.parametrize(
    "parameters", [{"beam_width": 0}, {"beam_width": 2, "n": 3}, {"n": 2}]
)
def test_invalid_beam_parameters_are_rejected(parameters: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}], **parameters
        )


class CandidateCoordinator:
    async def submit(self, request_id: str, request: OmniRequest) -> dict[str, object]:
        return {
            "text": "first",
            "candidates": [
                {
                    "text": "first",
                    "token_ids": [1, 2],
                    "sequence_score": -0.1,
                    "finish_reason": "length",
                },
                {
                    "text": "second",
                    "token_ids": [3],
                    "sequence_score": -0.2,
                    "finish_reason": "stop",
                },
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        }


def test_client_and_sse_preserve_all_candidates() -> None:
    client = Client(CandidateCoordinator())
    request = GenerateRequest(prompt="hello", stream=False)
    result = asyncio.run(client.completion(request, request_id="beam"))
    assert [candidate.text for candidate in result.candidates] == ["first", "second"]
    assert result.usage.completion_tokens == 3

    async def collect_client_stream() -> list[str]:
        return [
            candidate.text
            async for chunk in client.completion_stream(request, request_id="beam")
            for candidate in chunk.candidates
        ]

    assert asyncio.run(collect_client_stream()) == ["first", "second"]
    api_request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}], beam_width=2, n=2
    )

    async def collect() -> list[str]:
        return [
            event
            async for event in chat_stream_events(
                client, request, "beam", "chat-beam", 0, "test", api_request, "wav"
            )
        ]

    events = asyncio.run(collect())
    assert events[-1] == "data: [DONE]\n\n"
    response = json.loads(events[0].removeprefix("data: "))
    assert [choice["index"] for choice in response["choices"]] == [0, 1]
    assert [choice["sequence_score"] for choice in response["choices"]] == [-0.1, -0.2]
    assert response["usage"]["total_tokens"] == 7


@pytest.mark.skipif(
    not os.environ.get("OMNI_BEAM_TEST_URL"), reason="Requires a running sid-gr server"
)
def test_gpu_beam_mixed_concurrency_and_sse() -> None:
    endpoint = os.environ["OMNI_BEAM_TEST_URL"].rstrip("/") + "/v1/chat/completions"

    def generate(parameters: tuple[int, int, int]) -> dict[str, object]:
        width, count, length = parameters
        response = httpx.post(
            endpoint,
            json={
                "messages": [{"role": "user", "content": "Say hello"}],
                "beam_width": width,
                "n": count,
                "max_tokens": length,
                "temperature": 0,
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        assert len(result["choices"]) == count
        scores = [choice["sequence_score"] for choice in result["choices"]]
        if width > 1:
            assert scores == sorted(scores, reverse=True)
        else:
            pass
        assert 0 < result["usage"]["completion_tokens"] <= count * length
        return result

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(generate, [(4, 3, 1), (4, 3, 8), (1, 1, 8), (8, 5, 32)] * 2)
        )
    assert results[0]["usage"]["completion_tokens"] == 3
    response = httpx.post(
        endpoint,
        json={
            "messages": [{"role": "user", "content": "Say hello"}],
            "beam_width": 4,
            "n": 3,
            "max_tokens": 8,
            "stream": True,
        },
        timeout=60,
    )
    response.raise_for_status()
    events = [
        line[6:] for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    assert len(json.loads(events[0])["choices"]) == 3


@pytest.mark.parametrize("supported", [False, True])
def test_http_beam_capability_and_candidates(supported: bool) -> None:
    app = create_app(Client(CandidateCoordinator()), supports_beam_search=supported)
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "beam_width": 2,
            "n": 2,
        },
    )
    if supported:
        assert response.status_code == 200, response.text
        assert [
            choice["message"]["content"] for choice in response.json()["choices"]
        ] == ["first", "second"]
    else:
        assert response.status_code == 400


def test_stop_token_is_trimmed_from_text_but_counted_in_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StopTokenizer:
        def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
            assert token_ids == [1, 2]
            return "hello"

    monkeypatch.setattr(
        "sglang_omni.models.sid_gr.stages.get_tokenizer", lambda path: StopTokenizer()
    )
    executor = create_decode_executor("test-model")
    result = executor.fn(
        StagePayload(
            request_id="stop",
            request=OmniRequest(inputs="hi"),
            data={
                "prompt_tokens": 4,
                "candidates": [
                    {
                        "token_ids": [1, 2, 9],
                        "sequence_score": -0.5,
                        "finish_reason": "stop",
                        "matched_stop_token": 9,
                    }
                ],
            },
        )
    )
    assert result.data["candidates"][0]["text"] == "hello"
    assert result.data["usage"]["completion_tokens"] == 3


@pytest.mark.skipif(
    not os.environ.get("OMNI_BEAM_TEST_URL"),
    reason="Requires a running sid-gr server with 64 request slots",
)
def test_gpu_long_prompt_and_cancelled_beam_recovery() -> None:
    endpoint = os.environ["OMNI_BEAM_TEST_URL"].rstrip("/")
    messages = [
        {
            "role": "user",
            "content": "Write a detailed long essay about computing, at least 2000 words.",
        }
    ]
    cancellation_messages = [
        {
            "role": "user",
            "content": messages[0]["content"] + " Background context." * 200,
        }
    ]
    with httpx.Client(base_url=endpoint, timeout=60) as client:
        for _ in range(3):
            try:
                client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": cancellation_messages,
                        "beam_width": 32,
                        "n": 2,
                        "max_tokens": 512,
                    },
                    timeout=0.1,
                )
            except httpx.ReadTimeout:
                pass
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": messages,
                "beam_width": 64,
                "n": 2,
                "max_tokens": 100,
            },
        )
        response.raise_for_status()
        result = response.json()
        assert len(result["choices"]) == 2
        assert result["usage"]["completion_tokens"] == 200
        assert client.get("/health").json()["pending_completions"] == 0
