# Ordinary SID beam quality comparison

Measured on 2026-10-20 with OneRec-1.7B on NVIDIA L20. This comparison covers
ordinary, unconstrained beam search only. It does not establish equivalence with
SID-range or trie-constrained recommendation.

## Configuration

- Checkpoint: `OpenOneRec/OneRec-1.7B`, revision
  `cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b`; both servers load the same local files.
- Reference: repository `third_party/FlashRec`, revision
  `1089682a087ae5493f3b357ed1ec501cb684717b`.
- Input: existing RecIF video export
  `opensource-reference/FlashRec/results/onerec_aligned_bf16_ranges_L20_20260909/samples.json`,
  rows `[8:508]`. The export has 5,008 samples, including eight warmup rows.
  The new runs do not reuse the old constrained inference results.
- BF16 weights and KV cache, beam width 32, returned candidates 32, maximum
  generated tokens 5, request concurrency 8, length penalty 1, TP 1.
- Both tokenizers produced identical prompt token IDs on all 500 inputs.
- Both engines returned 32 candidates per sample. Every candidate ended at the
  five-token length limit in both runs, so the differing EOS policy was inactive.
- FlashRec: no catalog, SID ranges, or SID layout passed to the server; the
  startup log reports depth 0. FlashInfer FA2 attention. CUDA graphs enabled.
- sid-gr: FA3 in the first run; FlashInfer FA2 in the second. CUDA graphs enabled.
  Native request-slot limit remains 64; FlashRec beam-slot budget is 256, so
  internal batch formation is not identical despite equal request concurrency.
- Environment: torch `2.13.0+cu129`, SGLang `0.5.21+cu129`, FlashInfer `0.6.18`,
  Transformers `5.12.1`.

## Results

All percentages below are sample-averaged, except invalid rate, which averages
invalid candidate fractions. Each run evaluates the same 500 labeled inputs.

| Metric | sid-gr FA3 | FlashRec run 1 | sid-gr FA2 | FlashRec run 2 |
| --- | ---: | ---: | ---: | ---: |
| Hit@1 | 4.0% | 3.8% | 3.8% | 3.8% |
| Hit@4 | 8.6% | 8.4% | 8.6% | 8.2% |
| Hit@8 | 12.8% | 12.6% | 13.0% | 12.4% |
| Hit@16 | 15.6% | 15.4% | 15.4% | 16.0% |
| Hit@32 | 17.4% | 16.4% | 17.4% | 17.0% |
| Recall@32 | 3.3694% | 3.2279% | 3.3806% | 3.3786% |
| Deduplicated recall@32 | 2.8364% | 2.7234% | 2.8475% | 2.8206% |
| Invalid candidates | 27.1500% | 26.8688% | 26.8688% | 27.0188% |

Recall follows the reference convention of counting matching beam entries;
deduplicated recall additionally reports unique matched SIDs. Invalid candidates
are measured against the existing `sid2pid_beamrec_l4.json` catalog, used only
by the evaluation client.

| Paired measure | FA3 comparison | FA2 comparison |
| --- | ---: | ---: |
| Top-1 SID agreement | 96.8% | 96.6% |
| Mean top-32 SID set intersection / 32 | 91.16% | 91.24% |
| Samples with equal hit@32 outcome | 493 / 500 | 494 / 500 |
| Only sid-gr hits | 6 | 4 |
| Only FlashRec hits | 1 | 2 |
| Largest score difference on a shared SID | 0.09947 | 0.15613 |

Shared-SID score comparisons use the first occurrence of that SID. They do not
establish equality of the full generated token sequence. FlashRec itself has
98.4% top-1 agreement and 93.44% mean top-32 overlap across these two runs;
its hit@32 changes by 0.6 percentage points without changing server settings.
Batch composition and numerical execution remain possible contributors; this
experiment does not isolate a particular operator as the cause.

The evidence supports similar ordinary-beam quality on this slice, not exact
candidate equivalence, statistical equivalence, or superiority. Switching both
engines to FA2 does not eliminate output differences. A five-token evaluation
also differs from the earlier eight-token smoke test; their hit rates should
not be directly compared.

## Reproduction and environment compatibility

Use [the comparison command](./beam_search.md#compare-ordinary-beam-quality-with-a-reference-service).
Keep both servers isolated from production workloads. The reference server was
started with:

```bash
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
PYTHONPATH=third_party/FlashRec/python \
python results/sid_gr_comparison_20261020/flashrec_compat_launcher.py \
  --serve --model-path "$MODEL_PATH" \
  --quantization bf16 --kv-cache-dtype bfloat16 \
  --beam-width 32 --max-tokens 5 --max-seq-len 8192 \
  --cuda-graph-max-bs 256 --batch-slots 256 \
  --warmup-iters 0 --disable-warmup --host 127.0.0.1 --port 13427
```

The client passes `chat_template_kwargs.return_dict=false` to accommodate
Transformers 5. The installed sgl-kernel lacks
`apply_rope_with_cos_sin_cache_inplace`. An evaluation-only launcher assigns
FlashInfer's same-named function to that missing entry before calling
`flashrec.cli.main()`, using the fallback already present in FlashRec's rotary
module. This is an environment compatibility change, not an unmodified reference
binary; neither model weights nor beam-search logic were changed. No FlashRec
code was added to the sid-gr runtime.

Local reports, candidate lists, server logs, and the compatibility launcher are
saved under `results/sid_gr_comparison_20261020/{fa3,fa2}` and its parent directory.
The results directory is ignored by Git. Each summary records the sample file's
SHA-256 and evaluation settings. Both evaluation services were stopped after
completion.
