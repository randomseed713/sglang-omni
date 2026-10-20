# SPDX-License-Identifier: Apache-2.0
"""Compare ordinary SID beam outputs from two live HTTP services."""

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

SID_PATTERN = re.compile(r"<s_[abc]_\d+>")


class Sample(BaseModel):
    id: str
    messages: list[dict[str, str]]
    ground_truth_sids: list[str] = Field(min_length=1)


class Evaluation(BaseModel):
    id: str
    sids: list[str]
    texts: list[str]
    scores: list[float | None]
    finish_reasons: list[str]
    metrics: dict[str, float]


def extract_sid(text: str) -> str:
    return "".join(SID_PATTERN.findall(text)[:3])


def quality_metrics(
    sids: list[str], ground_truth: set[str], catalog: set[str], cutoffs: list[int]
) -> dict[str, float]:
    metrics = {}
    for cutoff in cutoffs:
        matches = set(sids[:cutoff]) & ground_truth
        metrics[f"hit@{cutoff}"] = float(bool(matches))
        metrics[f"recall@{cutoff}"] = sum(
            sid in ground_truth for sid in sids[:cutoff]
        ) / len(ground_truth)
        metrics[f"unique_recall@{cutoff}"] = len(matches) / len(ground_truth)
    metrics["invalid_rate"] = sum(sid not in catalog for sid in sids) / len(sids)
    metrics["unique_candidates"] = float(len(set(sids)))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid-gr-url", required=True)
    parser.add_argument("--reference-url", required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="Catalog JSON with comma-separated a,b,c[,end] keys.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Checkpoint used by both servers, recorded for provenance.",
    )
    parser.add_argument("--offset", type=int, default=8)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    if (
        min(
            arguments.count,
            arguments.beam_width,
            arguments.max_tokens,
            arguments.concurrency,
        )
        < 1
        or arguments.offset < 0
    ):
        parser.error("Counts must be positive and offset nonnegative")
    else:
        pass
    samples_bytes = arguments.samples.read_bytes()
    raw_samples = json.loads(samples_bytes)
    if arguments.offset + arguments.count > len(raw_samples):
        parser.error("Requested sample range exceeds the input file")
    else:
        pass
    samples = [
        Sample.model_validate(sample)
        for sample in raw_samples[arguments.offset : arguments.offset + arguments.count]
    ]
    catalog = set()
    for key in json.loads(arguments.catalog.read_text()):
        parts = key.split(",")
        if len(parts) not in (3, 4) or not all(part.isdigit() for part in parts):
            parser.error(f"Invalid catalog key: {key}")
        else:
            catalog.add(
                "".join(
                    f"<s_{layer}_{value}>" for layer, value in zip("abc", parts[:3])
                )
            )
    if not catalog:
        parser.error("Catalog must be nonempty")
    else:
        pass
    for sample in samples:
        if not sample.messages or any(
            re.fullmatch(r"<s_a_\d+><s_b_\d+><s_c_\d+>", sid) is None
            for sid in sample.ground_truth_sids
        ):
            parser.error(f"Invalid sample: {sample.id}")
        else:
            pass
    cutoffs = sorted(
        {arguments.beam_width}
        | {cutoff for cutoff in (1, 4, 8, 16, 32) if cutoff <= arguments.beam_width}
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, list[Evaluation]] = {}
    summaries: dict[str, dict[str, float]] = {}
    for engine, url in (
        ("sid-gr", arguments.sid_gr_url),
        ("flashrec", arguments.reference_url),
    ):

        def evaluate(sample: Sample) -> Evaluation:
            response = httpx.post(
                url.rstrip("/") + "/v1/chat/completions",
                json={
                    "messages": sample.messages,
                    "n": arguments.beam_width,
                    "beam_width": arguments.beam_width,
                    "max_tokens": arguments.max_tokens,
                    "temperature": 0,
                    "chat_template_kwargs": {
                        "return_dict": False,
                        "enable_thinking": False,
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            choices = sorted(
                response.json()["choices"], key=lambda choice: choice["index"]
            )
            if [choice["index"] for choice in choices] != list(
                range(arguments.beam_width)
            ):
                raise ValueError(f"Missing or duplicate choices: {engine}/{sample.id}")
            else:
                pass
            texts = [choice["message"]["content"] for choice in choices]
            sids = [extract_sid(text) for text in texts]
            return Evaluation(
                id=sample.id,
                sids=sids,
                texts=texts,
                scores=[
                    choice.get(
                        "sequence_score", choice.get("sglext", {}).get("sequence_score")
                    )
                    for choice in choices
                ],
                finish_reasons=[choice["finish_reason"] for choice in choices],
                metrics=quality_metrics(
                    sids, set(sample.ground_truth_sids), catalog, cutoffs
                ),
            )

        with ThreadPoolExecutor(max_workers=arguments.concurrency) as executor:
            rows = list(executor.map(evaluate, samples))
        outputs[engine] = rows
        summaries[engine] = {
            key: sum(row.metrics[key] for row in rows) / len(rows)
            for key in rows[0].metrics
        }
        (arguments.output_dir / f"{engine}.json").write_text(
            json.dumps([row.model_dump() for row in rows], ensure_ascii=False, indent=2)
            + "\n"
        )
        print(engine, json.dumps(summaries[engine]), flush=True)
    paired = []
    for left, right in zip(outputs["sid-gr"], outputs["flashrec"]):
        common = set(left.sids) & set(right.sids)
        differences = []
        for sid in common:
            left_score = left.scores[left.sids.index(sid)]
            right_score = right.scores[right.sids.index(sid)]
            if left_score is not None and right_score is not None:
                differences.append(abs(left_score - right_score))
            else:
                pass
        paired.append(
            {
                "id": left.id,
                "top1_equal": left.sids[0] == right.sids[0],
                "overlap": len(common) / arguments.beam_width,
                "max_score_difference": max(differences, default=None),
            }
        )
    summary = {
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
        "samples_sha256": hashlib.sha256(samples_bytes).hexdigest(),
        "metrics": summaries,
        "all_candidates_length_limited": {
            engine: all(
                reason == "length" for row in rows for reason in row.finish_reasons
            )
            for engine, rows in outputs.items()
        },
        "top1_agreement": sum(row["top1_equal"] for row in paired) / len(paired),
        "mean_topk_overlap": sum(row["overlap"] for row in paired) / len(paired),
        "paired": paired,
    }
    (arguments.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"Top-1 agreement: {summary['top1_agreement']:.4f}; "
        f"top-K overlap: {summary['mean_topk_overlap']:.4f}"
    )


if __name__ == "__main__":
    main()
else:
    pass
