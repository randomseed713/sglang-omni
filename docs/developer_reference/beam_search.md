# Native text beam search

The `sid-gr` pipeline currently provides ordinary, unconstrained beam search for
`Qwen3ForCausalLM` checkpoints. SID constraints and specialized recommendation
batching are not implemented yet.

## Architecture

```text
HTTP → Client → Coordinator
  → preprocessing (SimpleScheduler: tokenizer)
  → generation (OmniScheduler → ModelRunner → native SGLang model)
  → decode (SimpleScheduler: candidate decoding)
  → Client → HTTP
```

The shared scheduler initializes native beam groups, runs joint selection after
token relay, and commits results through the native result processor. The runner
preserves raw logits before sampling and separates member rows from ordinary
sampling rows. Native SGLang owns beam history, KV aliasing, remapping and release.
Failure cleanup releases member suffixes before releasing the leader.
Chunked prefill retains owned prompt progress as native int64 prefix indices
without inserting beam suffixes into the radix cache.

## Run on L20 / CUDA 12.9

Use the environment described in `Build.md`. Set `MODEL_PATH` to a local
Qwen3-0.6B checkpoint or another Qwen3 causal LM checkpoint.

```bash
source .venv-py310-cu129/bin/activate
export MODEL_PATH=/path/to/Qwen3-0.6B

CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 sgl-omni serve \
  --model-path "$MODEL_PATH" --model-name sid-gr \
  --generation.engine.attention_backend fa3 \
  --generation.engine.sampling_backend pytorch \
  --generation.engine.enable_torch_compile false \
  --generation.engine.context_length 2048 \
  --generation.engine.max_total_tokens 8192 \
  --host 127.0.0.1 --port 8000
```

The PyTorch sampler avoids a first-use FlashInfer sampling JIT build. Beam scores
come from native joint selection using preserved raw logits.

For TP2, expose two GPUs and add:

```bash
--generation.tp_size 2 --generation.gpu '[0,1]'
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sid-gr",
    "messages": [{"role": "user", "content": "Say hello"}],
    "beam_width": 4,
    "n": 3,
    "max_tokens": 8
  }'
```

`beam_width` controls search width; `n` controls returned candidates and must not
exceed the width. Each choice includes `sequence_score`. Completion usage sums
tokens across returned candidates, including generated stop tokens. Custom stop
tokens are removed from displayed text. Length normalization currently uses the
native default exponent of 1.

With `stream: true`, the server sends finalized candidates together in one SSE
chunk followed by `[DONE]`; intermediate beam paths are not stable output.
Without beam search, `n` must be 1.

The initial implementation uses synchronous scheduling and page size 1. It rejects
async decode and native unsupported combinations. Stop strings are unsupported
for beam search; use `stop_token_ids`. Temperature does not enable stochastic
beam search. Only the `sid-gr` pipeline currently advertises beam support.

## Verification

CPU contracts and regression tests:

```bash
python -m pytest tests/unit_test/sid_gr tests/unit_test/model_runner/test_sglang_execution.py \
  tests/unit_test/pipeline/test_scheduler.py
```

Against an already running server with at least 64 request slots (add
`--generation.engine.chunked_prefill_size 16` at startup to exercise chunk boundaries):

```bash
OMNI_BEAM_TEST_URL=http://127.0.0.1:8000 \
  python -m pytest tests/unit_test/sid_gr/test_generation.py
```

## End-to-end SID hit rate

Use the OneRec-1.7B recommendation checkpoint and the RecIF video samples JSON
used by `third_party/FlashRec/scripts/eval_beam_matrix.py`. Each sample contains
`id`, `messages`, and `ground_truth_sids` (canonical strings such as
`<s_a_1><s_b_2><s_c_3>`). Labels must come from the evaluation dataset, not model
predictions. The test sends requests through `/v1/chat/completions` and exercises
all three stages. No FlashRec runtime is imported.

Start the server using the command above with the OneRec checkpoint; use
`--generation.engine.context_length 8192` and
`--generation.engine.max_total_tokens 32768` for its longer history prompts.
Then run:

```bash
OMNI_BEAM_TEST_URL=http://127.0.0.1:8000 \
OMNI_SID_SAMPLES=/path/to/samples.json \
OMNI_SID_SAMPLE_OFFSET=8 OMNI_SID_SAMPLE_COUNT=50 \
OMNI_SID_BEAM_WIDTH=32 OMNI_SID_CONCURRENCY=8 OMNI_SID_MAX_TOKENS=8 \
OMNI_SID_MIN_HIT_RATE=0.01 \
OMNI_SID_REPORT=/tmp/sid-hit-rate.json \
python -m pytest tests/unit_test/sid_gr/test_hit_rate.py -q
```

The offset can exclude warmup samples from an existing evaluation export. If
omitted, the test uses the entire file. Hit@K is the fraction of samples with at
least one matching SID among the first K candidates, extracting the first three
SID tokens from each response, matching the reference evaluator's convention.
Repeated candidates do not increase hit rate. Candidate order follows response
choice indices. HTTP errors and missing candidates fail the test.

The report includes per-sample candidates, scores, labels, and aggregate hit@K.
`OMNI_SID_MIN_HIT_RATE` is required when running the live test; the value above
is a smoke-test threshold, not a calibrated regression baseline. Set it to an
agreed threshold for a fixed checkpoint, dataset slice, and generation settings.
This evaluates ordinary open-vocabulary beam search, not SID-constrained beam,
so its results must not be treated as equivalent to constrained search metrics.

## Compare ordinary beam quality with a reference service

`benchmarks/eval/benchmark_sid_beam_compare.py` sends the same labeled messages
to sid-gr and a separate FlashRec HTTP service. Start both with the same
checkpoint, BF16 weights and KV cache, length penalty 1, and no SID constraints.
For FlashRec, omit all SID catalog/range/layout options and set
`--quantization bf16 --kv-cache-dtype bfloat16`. Do not reuse measurements made
with trie or SID-range constraints. The reference service is only an evaluation
baseline; it is not imported by the sid-gr runtime or the comparison client.

```bash
python benchmarks/eval/benchmark_sid_beam_compare.py \
  --sid-gr-url http://127.0.0.1:8000 \
  --reference-url http://127.0.0.1:8001 \
  --model-path "$MODEL_PATH" \
  --samples /path/to/samples.json \
  --catalog /path/to/sid2pid_beamrec_l4.json \
  --offset 8 --count 500 --beam-width 32 --max-tokens 5 --concurrency 8 \
  --output-dir /tmp/sid-beam-comparison
```

The catalog is used only for invalid-candidate measurement. Outputs include both
engines' candidates, scores, finish reasons, hit@K, recall@K, deduplicated recall,
invalid rate, unique-candidate counts, paired top-1 agreement and top-K overlap.
Recall counts matching beam entries for compatibility with the reference metric;
deduplicated recall prevents repeated SIDs from inflating that number. Top-K
overlap uses the intersection of SID sets divided by K. Score differences match
shared SIDs at their first occurrence. These measures do not imply token-level
identity. Request errors abort the comparison instead of dropping samples.

The client sets `return_dict=false` and `enable_thinking=false` in the reference
chat template options. Verify prompt token IDs and finish reasons before drawing
conclusions: FlashRec ignores EOS, whereas the current sid-gr stops at EOS. A
fixed five-token SID evaluation is aligned only when all candidates reach the
length limit. The supplied model path records provenance; the client cannot
verify what checkpoints remote servers actually loaded.

See [the L20 comparison report](./sid_gr_quality_comparison.md) for the measured
500-sample results, environment adaptations, and remaining limitations.
