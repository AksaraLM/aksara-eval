# 📊 aksara-eval

Evaluation suite for AksaraLLM models — **Aksara-Indo-Bench**.

Evaluates any HuggingFace-compatible causal LM (including AksaraLLM
checkpoints exported via `aksarallm.hf_export`, see the aksaraLLM repo) —
the runner is model-agnostic, not tied to any specific base model.

## Benchmark Tracks

| Track | Task | Metric |
|---|---|---|
| Knowledge | IndoMMLU | 5-shot accuracy |
| Reasoning | COPAL-ID | accuracy |
| Regional | NusaX-sentiment (11 langs) | macro-F1 |
| Safety | Aksara-Safety (custom) | pass-rate |

See [`aksara_indo_bench/README.md`](aksara_indo_bench/README.md) for the full
design (composite score weighting, planned tracks, data sources).

## Quick Start

```bash
pip install transformers datasets torch

python -m aksara_indo_bench.run --model <hf-repo-or-local-dir> --tasks all --out results.json
```

`--model` accepts any HF-format model directory or repo id — a checkpoint
exported from aksaraLLM works the same as anything else on the Hub.

## Contributing

To add a task, create a file in `aksara_indo_bench/tasks/` with a class that
inherits `BenchTask` (see [`aksara_indo_bench/README.md`](aksara_indo_bench/README.md#contributing)
for the exact interface).

## License
Apache 2.0
