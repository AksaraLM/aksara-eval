# Aksara-Indo-Bench v0

A benchmark suite for Indonesian language models, covering **general knowledge, cultural knowledge, pragmatic reasoning, regional languages, and Indonesia-specific safety**.

## Design principles

1. **Tidak translate benchmark asing.** Gunakan benchmark yang native Indonesian atau hand-authored for Indonesia.
2. **Per-track scores + composite**, bukan single-number. Memungkinkan analisa kekuatan/kelemahan.
3. **Reproducible**: semua task dari HuggingFace datasets atau tersimpan di repo `AksaraLLM/aksara-eval`.
4. **Lightweight runner** compatible dengan `lm-evaluation-harness` (tasks adapter).
5. **Safety track terpisah** dengan isu lokal Indonesia (SARA, politik, hoax).

## Task tracks (v0)

| Track | Task | Source | Metric | # items | Notes |
|---|---|---|---|---|---|
| **Knowledge** | IndoMMLU | [IndoNLP/IndoMMLU](https://huggingface.co/datasets/IndoNLP/indommlu) | 5-shot accuracy | 14,981 | 63 subjects |
| **Knowledge** | IndoStoryCloze | [jakartaresearch/indo-story-cloze](https://huggingface.co/datasets/jakartaresearch/indo-story-cloze) | accuracy | ~1,100 | story completion |
| **Culture** | IndoCulture | [IndoNLP/IndoCulture](https://huggingface.co/datasets/IndoNLP/indoculture) | accuracy | ~2,400 | cultural norms |
| **Culture** | LocalValueBench-ID | custom (this repo) | accuracy | ~500 (target) | hand-authored |
| **Reasoning** | COPAL-ID | [haryoaw/COPAL](https://huggingface.co/datasets/haryoaw/COPAL) | accuracy | ~600 | pragmatic (causal) |
| **Reasoning** | XCOPA-id | [xcopa](https://huggingface.co/datasets/xcopa) | accuracy | 500 | commonsense |
| **Reading** | TyDiQA-id (gold) | [khalidalt/tydiqa-goldp](https://huggingface.co/datasets/khalidalt/tydiqa-goldp) | F1 | ~1,000 | extractive QA |
| **Reading** | Indo-MMLU-pro (target) | TBD | accuracy | - | harder QA |
| **Regional** | NusaX-sentiment (11 langs) | [indonlp/NusaX-senti](https://huggingface.co/datasets/indonlp/NusaX-senti) | macro-F1 | 5,388 | sentiment |
| **Regional** | NusaTranslation (target) | [indonlp/NusaTranslation](https://huggingface.co/datasets/indonlp/NusaTranslation) | chrF / BLEU | - | ID ↔ 11 langs |
| **Safety** | Aksara-Safety (custom) | this repo | pass-rate | 150 (target) | refuse / comply |
| **Generation** | IndoSum (news) | [id_liputan6](https://huggingface.co/datasets/id_liputan6) | ROUGE-L | 200 (sampled) | summarization |

### Composite score

```
aksara_score = 0.25 * knowledge_mean      # IndoMMLU, IndoStoryCloze
             + 0.15 * culture_mean        # IndoCulture, LocalValueBench
             + 0.15 * reasoning_mean      # COPAL-ID, XCOPA
             + 0.15 * reading_mean        # TyDiQA
             + 0.20 * regional_mean       # NusaX, NusaTranslation
             + 0.05 * safety              # Aksara-Safety
             + 0.05 * generation          # IndoSum
```

## Directory structure

```
aksara-eval/
├── README.md
├── aksara_indo_bench/
│   ├── __init__.py
│   ├── run.py                    # CLI: python -m aksara_indo_bench.run --model X
│   ├── tasks/
│   │   ├── indommlu.py
│   │   ├── indoculture.py
│   │   ├── copal_id.py
│   │   ├── tydiqa_id.py
│   │   ├── nusax_senti.py
│   │   └── aksara_safety.py
│   └── data/
│       ├── local_value_bench_id/   # hand-authored culture items
│       └── aksara_safety/          # hand-authored safety items
└── baselines/                       # results on comparison models
    ├── sealion-8b.json
    ├── sahabat-ai-v2.json
    ├── merak-7b-v4.json
    ├── aksarallm-kiel-pro-0.5b.json
    └── README.md                    # narrative comparison
```

## Roadmap

- **v0.1** (current): Scaffold tasks + runner, reuse existing benchmarks.
- **v0.2**: LocalValueBench-ID (100 items hand-authored).
- **v0.3**: Aksara-Safety (50 items hand-authored).
- **v1.0**: Full 500 + 150 item custom tracks, public leaderboard.

## Contributing

Untuk menambahkan task, buat file baru di `tasks/` dengan class yang inherit `BenchTask`:

```python
from aksara_indo_bench.base import BenchTask

class MyTask(BenchTask):
    name = "mytask"
    track = "knowledge"
    dataset = "my/dataset"

    def format_prompt(self, example):
        return f"Pertanyaan: {example['question']}\nJawaban:"

    def extract_answer(self, output):
        return output.strip().split("\n")[0]

    def score(self, pred, example):
        return float(pred == example["answer"])
```
