# SPDX-License-Identifier: Apache-2.0
"""Text request and multi-candidate result adapters."""

from array import array

import torch
from pydantic import BaseModel, Field
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import PreTrainedTokenizerBase

from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


class TokenizedPrompt(BaseModel):
    token_ids: list[int] = Field(min_length=1)


class GenerationParameters(BaseModel):
    max_new_tokens: int = Field(default=32, ge=1)
    beam_width: int | None = Field(default=None, ge=1)
    n: int = Field(default=1, ge=1)
    temperature: float = Field(default=1.0, ge=0)
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    stop: list[str] = Field(default_factory=list)
    stop_token_ids: list[int] = Field(default_factory=list)
    seed: int | None = None


class SIDGRRequestBuilder:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, vocab_size: int) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    def __call__(self, payload: StagePayload) -> SGLangARRequestData:
        prompt = TokenizedPrompt.model_validate(payload.data)
        parameters = GenerationParameters.model_validate(
            {
                name: value
                for name, value in payload.request.params.items()
                if value is not None
            }
        )
        if parameters.n > (parameters.beam_width or 1):
            raise ValueError(
                "n must not exceed beam_width; parallel sampling is unsupported"
            )
        else:
            pass
        if any(token < 0 or token >= self.vocab_size for token in prompt.token_ids):
            raise ValueError("Prompt token IDs are outside the model vocabulary")
        else:
            pass
        sampling = SamplingParams(
            **parameters.model_dump(exclude={"seed"}), sampling_seed=parameters.seed
        )
        sampling.normalize(self.tokenizer)
        sampling.verify(self.vocab_size)
        request = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=array("q", prompt.token_ids),
            sampling_params=sampling,
            eos_token_ids={self.tokenizer.eos_token_id},
            vocab_size=self.vocab_size,
        )
        request.tokenizer = self.tokenizer
        return SGLangARRequestData(
            req=request,
            input_ids=torch.tensor(prompt.token_ids, dtype=torch.long),
            stage_payload=payload,
            enforce_request_limits=True,
            max_new_tokens=parameters.max_new_tokens,
        )


class SIDGRResultAdapter:
    def __call__(self, request: SGLangARRequestData) -> StagePayload:
        payload = request.stage_payload
        assert payload is not None
        if request.beam_output is not None:
            candidates = [
                {
                    "token_ids": sequence.tokens,
                    "sequence_score": sequence.beam_score,
                    "finish_reason": sequence.finish_reason["type"],
                    "matched_stop_token": sequence.finish_reason.get("matched"),
                }
                for sequence in request.beam_output.sequences
            ]
        else:
            candidates = [
                {
                    "token_ids": request.output_ids,
                    "sequence_score": None,
                    "finish_reason": request.finish_reason,
                }
            ]
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "candidates": candidates,
                "prompt_tokens": len(request.req.origin_input_ids),
            },
        )
