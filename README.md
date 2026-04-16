# 📊 aksara-eval

Evaluation suite and benchmarks for AksaraLLM models.

## Benchmarks

| Category | Tests | Description |
|---|---|---|
| 🆔 Identity | 8 | Self-awareness & anti-impersonation |
| 🛡️ Safety | 5 | Harmful content refusal |
| 🇮🇩 Indonesian Knowledge | 8 | Pancasila, history, geography |
| 💡 General QA | 5 | Science, politics, environment |
| 🔢 Math | 5 | Arithmetic & word problems |
| 💻 Coding | 4 | Python, JavaScript |
| 📝 Fluency | 4 | Long-form writing quality |

## Features
- ✅ Perplexity measurement on Indonesian text
- ✅ Comparison with base model (Qwen2.5-1.5B)
- ✅ Auto-upload results to HuggingFace
- ✅ Model card auto-update with scores
- ✅ Auto-detect new models and evaluate

## Quick Start
```bash
pip install transformers datasets huggingface_hub
python3 benchmark.py
```

## License
Apache 2.0
