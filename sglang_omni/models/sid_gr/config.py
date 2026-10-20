# SPDX-License-Identifier: Apache-2.0
"""Native causal language model pipeline with beam generation."""

from typing import ClassVar

from sglang_omni.config.schema import EngineStageConfig, PipelineConfig, StageConfig


class SIDGRPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "Qwen3ForCausalLM"
    supports_beam_search: ClassVar[bool] = True
    model_path: str
    entry_stage: str = "preprocessing"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "generation": EngineStageConfig,
    }
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="text",
            factory_path="sglang_omni.models.sid_gr.stages.create_preprocessing_executor",
            next="generation",
        ),
        EngineStageConfig(
            name="generation",
            process="generation",
            factory_path="sglang_omni.models.sid_gr.stages.create_generation_executor",
            gpu=0,
            next="decode",
        ),
        StageConfig(
            name="decode",
            process="text",
            factory_path="sglang_omni.models.sid_gr.stages.create_decode_executor",
            terminal=True,
        ),
    ]


EntryClass = SIDGRPipelineConfig
