# SPDX-License-Identifier: Apache-2.0
"""End-to-end SID recommendation quality over the public HTTP API."""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

import httpx
import pytest

from benchmarks.eval.benchmark_sid_beam_compare import extract_sid, quality_metrics


class SIDSampleResult(TypedDict):
    id: str
    ground_truth_sids: list[str]
    candidates: list[str]
    sequence_scores: list[float | None]
    metrics: dict[str, float]


SID_TOKEN_PATTERN = re.compile(r"<s_[abc]_\d+>")


def sid_hit_at_k(candidates: list[str], ground_truth: set[str], cutoff: int) -> float:
    """Return whether any of the first K candidates matches a target SID."""
    predictions = [
        "".join(SID_TOKEN_PATTERN.findall(text)[:3]) for text in candidates[:cutoff]
    ]
    return float(bool(set(predictions) & ground_truth))


def test_sid_hit_rate_counts_samples_not_candidates() -> None:
    target = "<s_a_1><s_b_2><s_c_3>"
    other = "<s_a_4><s_b_5><s_c_6>"
    assert sid_hit_at_k([other, target, target], {target}, 1) == 0
    assert sid_hit_at_k([other, target, target], {target}, 3) == 1
    assert sid_hit_at_k([f"<|sid_begin|>{target}<|sid_end|>"], {target}, 1) == 1
    assert sid_hit_at_k(["invalid", "<s_a_1>"], {target}, 2) == 0
    assert sum(sid_hit_at_k(row, {target}, 2) for row in [[target], [other]]) / 2 == 0.5


@pytest.mark.skipif(
    not os.environ.get("OMNI_BEAM_TEST_URL") or not os.environ.get("OMNI_SID_SAMPLES"),
    reason="Requires a running sid-gr server and labeled SID samples",
)
def test_gpu_sid_hit_rate(tmp_path: Path) -> None:
    """Evaluate labeled recommendations through the complete HTTP pipeline."""
    samples = json.loads(Path(os.environ["OMNI_SID_SAMPLES"]).read_text())
    beam_width = int(os.environ.get("OMNI_SID_BEAM_WIDTH", "32"))
    max_tokens = int(os.environ.get("OMNI_SID_MAX_TOKENS", "8"))
    concurrency = int(os.environ.get("OMNI_SID_CONCURRENCY", "8"))
    minimum_hit_rate = float(os.environ["OMNI_SID_MIN_HIT_RATE"])
    sample_offset = int(os.environ.get("OMNI_SID_SAMPLE_OFFSET", "0"))
    sample_count = int(os.environ.get("OMNI_SID_SAMPLE_COUNT", str(len(samples))))
    assert sample_offset >= 0 and sample_count > 0
    assert sample_offset + sample_count <= len(samples)
    samples = samples[sample_offset : sample_offset + sample_count]
    assert isinstance(samples, list) and samples, "Samples must be a nonempty JSON list"
    assert beam_width >= 1 and max_tokens >= 1 and concurrency >= 1
    assert 0 < minimum_hit_rate <= 1, "Set a meaningful positive quality threshold"
    cutoffs = sorted({1, beam_width} | {k for k in (4, 8, 16, 32) if k <= beam_width})
    endpoint = os.environ["OMNI_BEAM_TEST_URL"].rstrip("/") + "/v1/chat/completions"

    def evaluate(sample_index: int) -> SIDSampleResult:
        sample = samples[sample_index]
        ground_truth = set(sample["ground_truth_sids"])
        assert ground_truth and all(
            re.fullmatch(r"<s_a_\d+><s_b_\d+><s_c_\d+>", sid) for sid in ground_truth
        ), f"Invalid ground truth at sample {sample_index}"
        assert sample["messages"], f"Missing messages at sample {sample_index}"
        response = httpx.post(
            endpoint,
            json={
                "messages": sample["messages"],
                "beam_width": beam_width,
                "n": beam_width,
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        choices = sorted(response.json()["choices"], key=lambda choice: choice["index"])
        assert len(choices) == beam_width
        candidates = [choice["message"]["content"] for choice in choices]
        assert all(isinstance(text, str) for text in candidates)
        return {
            "id": str(sample.get("id", sample_offset + sample_index)),
            "ground_truth_sids": sorted(ground_truth),
            "candidates": candidates,
            "sequence_scores": [choice["sequence_score"] for choice in choices],
            "metrics": {
                f"hit@{cutoff}": sid_hit_at_k(candidates, ground_truth, cutoff)
                for cutoff in cutoffs
            },
        }

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(evaluate, range(len(samples))))
    metrics = {
        f"hit@{cutoff}": sum(row["metrics"][f"hit@{cutoff}"] for row in results)
        / len(samples)
        for cutoff in cutoffs
    }
    report_path = Path(
        os.environ.get("OMNI_SID_REPORT", str(tmp_path / "hit_rate.json"))
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "samples": len(samples),
                "sample_offset": sample_offset,
                "samples_path": os.environ["OMNI_SID_SAMPLES"],
                "beam_width": beam_width,
                "max_tokens": max_tokens,
                "concurrency": concurrency,
                "minimum_hit_rate": minimum_hit_rate,
                "metrics": metrics,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    assert (
        metrics[f"hit@{beam_width}"] >= minimum_hit_rate
    ), f"Hit rate below threshold: {metrics}; report: {report_path}"


def test_comparison_metrics_expose_duplicate_and_invalid_candidates() -> None:
    target = "<s_a_1><s_b_2><s_c_3>"
    missing = "<s_a_4><s_b_5><s_c_6>"
    assert extract_sid(f"<|sid_begin|>{target}<|sid_end|>") == target
    metrics = quality_metrics([target, target, ""], {target, missing}, {target}, [1, 3])
    assert metrics["hit@3"] == 1
    assert metrics["recall@3"] == 1
    assert metrics["unique_recall@3"] == 0.5
    assert metrics["invalid_rate"] == pytest.approx(1 / 3)
    assert metrics["unique_candidates"] == 2
