# SPDX-License-Identifier: Apache-2.0
"""Stage factories for native text generation."""

from collections.abc import Mapping

import torch
from pydantic import BaseModel
from sglang.srt.utils.hf_transformers import get_tokenizer
from transformers import AutoConfig, PreTrainedTokenizerBase

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.sid_gr.request_builders import (
    SIDGRRequestBuilder,
    SIDGRResultAdapter,
    TokenizedPrompt,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.bootstrap import InfrastructureOptions
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SGLangGenerationEngineBuilder,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


class TextMessage(BaseModel):
    role: str
    content: str


class TextInputs(BaseModel):
    prompt: str | None = None
    prompt_token_ids: list[int] | None = None
    messages: list[TextMessage] | None = None


class Candidate(BaseModel):
    token_ids: list[int]
    sequence_score: float | None
    finish_reason: str
    matched_stop_token: int | None = None


class CandidateBatch(BaseModel):
    candidates: list[Candidate]
    prompt_tokens: int


def create_preprocessing_executor(
    model_path: str,
) -> SimpleScheduler[StagePayload, StagePayload]:
    tokenizer = get_tokenizer(model_path)

    def preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        if isinstance(inputs, str):
            inputs = {"prompt": inputs}
        elif isinstance(inputs, list):
            inputs = (
                {"prompt_token_ids": inputs}
                if all(isinstance(token, int) for token in inputs)
                else {"messages": inputs}
            )
        else:
            pass
        prompt = TextInputs.model_validate(inputs)
        if prompt.prompt_token_ids is not None:
            token_ids = prompt.prompt_token_ids
        elif prompt.messages:
            token_ids = tokenizer.apply_chat_template(
                [message.model_dump() for message in prompt.messages],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        elif prompt.prompt is not None:
            token_ids = tokenizer.encode(prompt.prompt)
        else:
            raise ValueError("A text prompt, messages or prompt_token_ids is required")
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=TokenizedPrompt(token_ids=token_ids).model_dump(),
        )

    return SimpleScheduler(preprocess)


def create_decode_executor(
    model_path: str,
) -> SimpleScheduler[StagePayload, StagePayload]:
    tokenizer = get_tokenizer(model_path)

    def decode(payload: StagePayload) -> StagePayload:
        batch = CandidateBatch.model_validate(payload.data)
        candidates = [
            {
                **candidate.model_dump(exclude={"matched_stop_token"}),
                "text": tokenizer.decode(
                    (
                        candidate.token_ids[:-1]
                        if candidate.matched_stop_token is not None
                        else candidate.token_ids
                    ),
                    skip_special_tokens=True,
                ),
                "finish_reason": (
                    "length" if candidate.finish_reason == "length" else "stop"
                ),
            }
            for candidate in batch.candidates
        ]
        completion_tokens = sum(
            len(candidate.token_ids) for candidate in batch.candidates
        )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": candidates[0]["text"],
                "finish_reason": candidates[0]["finish_reason"],
                "candidates": candidates,
                "usage": {
                    "prompt_tokens": batch.prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": batch.prompt_tokens + completion_tokens,
                },
            },
        )

    return SimpleScheduler(decode)


class SIDGREngineBuilder(SGLangGenerationEngineBuilder[SGLangARRequestData]):
    model_name = "sid-gr"
    supports_context_length_override = True
    tokenizer: PreTrainedTokenizerBase
    vocab_size: int

    def __init__(self, *, tp_rank: int, nccl_port: int | None) -> None:
        self.tp_rank = tp_rank
        self.nccl_port = nccl_port

    def infra_kwargs(self) -> InfrastructureOptions:
        return {"tp_rank": self.tp_rank, "nccl_port": self.nccl_port}

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        configuration = AutoConfig.from_pretrained(checkpoint_dir)
        self.context_length = configuration.max_position_embeddings
        self.vocab_size = configuration.vocab_size
        self.tokenizer = get_tokenizer(checkpoint_dir)

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            "dtype": dtype,
            "max_running_requests": 64,
            "disable_overlap_schedule": True,
            "enable_mixed_chunk": False,
        }

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> ModelRunner[SGLangARRequestData]:
        return ModelRunner(model_worker, output_proc)

    def make_adapters(
        self, model: torch.nn.Module
    ) -> tuple[SIDGRRequestBuilder, SIDGRResultAdapter]:
        return (
            SIDGRRequestBuilder(self.tokenizer, self.vocab_size),
            SIDGRResultAdapter(),
        )


def create_generation_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    server_args_overrides: Mapping[str, object] | None = None,
) -> OmniScheduler[SGLangARRequestData]:
    overrides = dict(server_args_overrides or {})
    overrides.setdefault("page_size", 1)
    overrides["tp_size"] = tp_size
    return SIDGREngineBuilder(tp_rank=tp_rank, nccl_port=nccl_port).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=overrides,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )
