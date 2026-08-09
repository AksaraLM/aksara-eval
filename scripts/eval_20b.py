#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AksaraLLM 20B — EVALUATION HARNESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

One-stop harness that:

1. Loads an ``aksaraLLMModel`` checkpoint (or any HF ``AutoModelForCausalLM``)
   plus its tokenizer.
2. Measures **perplexity** on a configurable Indonesian corpus
   (default: 1 000 samples from ``wikipedia/20231101.id``).
3. Runs the in-repo :mod:`aksara_indo_bench` task suite (IndoMMLU,
   CopalID, NusaX-Senti, AksaraSafety).
4. Computes two AksaraLLM-specific signals:
     - **Identity accuracy** — does the model say "AksaraLLM" when asked?
     - **English leakage** — fraction of tokens in the first 50 generated
       that match a small English-word lexicon.
5. Writes ``results.json`` + a markdown report.

Run (HF-hosted checkpoint):
    python3 scripts/eval_20b.py \\
        --model Ezekiel999/AksaraLLM-20B-Instruct \\
        --tokenizer Ezekiel999/aksara-tokenizer-20b \\
        --out results.json

Dry-run (CPU, ~1 s, no network):
    python3 scripts/eval_20b.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from statistics import mean
from typing import Iterable

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

AKSARALLM_DIR = os.path.abspath(os.path.join(REPO_ROOT, "..", "aksaraLLM"))
if AKSARALLM_DIR not in sys.path:
    sys.path.insert(0, AKSARALLM_DIR)


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════
#  Prompt sets
# ══════════════════════════════════════════════════════════════════
IDENTITY_PROMPTS = [
    "Siapa kamu?",
    "Apa nama kamu?",
    "Kamu buatan siapa?",
    "Model apa yang sedang saya gunakan?",
    "Kenalkan dirimu dalam satu kalimat.",
]

# Identity answers are "positive" iff they mention AksaraLLM and avoid
# competitor names.
IDENTITY_MUST_CONTAIN = ("aksarallm", "aksara")
IDENTITY_MUST_NOT = ("chatgpt", "openai", "gemini", "claude", "qwen", "llama")

ENGLISH_LEAK_LEXICON = {
    "the", "and", "is", "are", "was", "were", "this", "that", "with",
    "for", "you", "i", "we", "it", "as", "on", "of", "in", "to", "a",
    "hello", "sorry", "please", "thank", "thanks", "but", "not", "have",
    "has", "had", "be", "been", "will", "would", "should", "could",
}


def _english_leak_fraction(text: str) -> float:
    tokens = re.findall(r"[A-Za-z]+", text.lower())[:50]
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in ENGLISH_LEAK_LEXICON)
    return hits / len(tokens)


# ══════════════════════════════════════════════════════════════════
#  Adapters
# ══════════════════════════════════════════════════════════════════
class AksaraAdapter:
    """Wraps a custom aksaraLLMModel + AksaraTokenizer behind a uniform API."""

    def __init__(self, model_path: str, tokenizer_path: str, device: str = "cpu"):
        from aksarallm.model import aksaraLLMModel
        from aksarallm.tokenizer_utils import AksaraTokenizer

        self.model = aksaraLLMModel.from_pretrained(model_path).to(device)
        self.model.eval()
        self.tokenizer = AksaraTokenizer.from_pretrained(tokenizer_path)
        self.device = device

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        import torch

        ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=max_new_tokens, temperature=0.0,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = out[0, ids.shape[1]:].tolist()
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def loglikelihood(self, text: str, max_len: int = 512) -> tuple[float, int]:
        """Return (sum_log_prob, n_tokens) for ``text`` (lower is better)."""
        import torch

        ids = self.tokenizer.encode(text)[:max_len]
        if len(ids) < 2:
            return 0.0, 0
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(x)
        logp = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        target = x[:, 1:]
        gathered = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return float(gathered.sum().item()), int(target.numel())


class HFAdapter:
    """Fallback adapter for any HF ``AutoModelForCausalLM``."""

    def __init__(self, model_path: str, tokenizer_path: str | None = None, device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok_path = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.model.eval()
        self.device = device

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        import torch

        inp = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inp, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = out[0, inp.input_ids.shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def loglikelihood(self, text: str, max_len: int = 512) -> tuple[float, int]:
        import torch

        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(self.device)
        if enc.input_ids.shape[1] < 2:
            return 0.0, 0
        with torch.no_grad():
            logits = self.model(**enc).logits
        logp = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        target = enc.input_ids[:, 1:]
        gathered = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return float(gathered.sum().item()), int(target.numel())


# ══════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════
def perplexity(adapter, texts: Iterable[str]) -> dict:
    t0 = time.time()
    total_lp = 0.0
    total_tokens = 0
    n = 0
    for text in texts:
        lp, nt = adapter.loglikelihood(text)
        total_lp += lp
        total_tokens += nt
        n += 1
    avg_nll = -total_lp / max(total_tokens, 1)
    return {
        "n_samples": n,
        "n_tokens": total_tokens,
        "perplexity": math.exp(avg_nll) if total_tokens else float("inf"),
        "seconds": round(time.time() - t0, 3),
    }


def identity_eval(adapter, prompts: list[str]) -> dict:
    hits = 0
    leaks = []
    samples = []
    for p in prompts:
        out = adapter.generate(p, max_new_tokens=64)
        low = out.lower()
        ok = any(x in low for x in IDENTITY_MUST_CONTAIN) and not any(
            x in low for x in IDENTITY_MUST_NOT
        )
        hits += int(ok)
        leaks.append(_english_leak_fraction(out))
        samples.append({"prompt": p, "output": out, "identity_ok": ok})
    return {
        "identity_accuracy": hits / max(len(prompts), 1),
        "english_leak_frac_mean": round(mean(leaks), 4) if leaks else 0.0,
        "samples": samples,
    }


def run_aksara_indo_bench(adapter, tasks: str = "all", limit: int = 50) -> dict:
    from aksara_indo_bench.tasks.indommlu import IndoMMLU
    from aksara_indo_bench.tasks.copal_id import CopalID
    from aksara_indo_bench.tasks.nusax_senti import NusaXSenti, LANGS
    from aksara_indo_bench.tasks.aksara_safety import AksaraSafety

    registry = {
        "indommlu": IndoMMLU(),
        "copal_id": CopalID(),
        "aksara_safety": AksaraSafety(),
        **{f"nusax_senti.{l}": NusaXSenti(l) for l in LANGS},
    }
    keys = list(registry.keys()) if tasks == "all" else tasks.split(",")

    results = []
    for k in keys:
        task = registry.get(k)
        if task is None:
            log(f"unknown task {k!r}, skipping", level="WARN")
            continue
        try:
            r = task.run(adapter, limit=limit)
            results.append({"name": r.name, "track": r.track, "score": r.score, "n": r.n})
            log(f"  {k} score={r.score:.4f} n={r.n}")
        except Exception as e:
            log(f"  {k} failed: {e}", level="WARN")
            results.append({"name": k, "error": str(e)})
    return {"per_task": results}


# ══════════════════════════════════════════════════════════════════
#  Dry-run helpers
# ══════════════════════════════════════════════════════════════════
class _DryRunAdapter:
    """Tiny random-weights model + random BPE tokenizer. CPU only."""

    def __init__(self):
        from aksarallm.config import get_config
        from aksarallm.model import aksaraLLMModel
        from aksarallm.tokenizer_utils import AksaraTokenizer

        cfg = get_config("tiny")
        self.model = aksaraLLMModel(cfg)
        self.model.eval()
        corpus = [
            "halo saya aksarallm",
            "indonesia adalah negara kepulauan terbesar",
            "pancasila adalah dasar negara",
        ] * 30
        self.tokenizer = AksaraTokenizer.train_bpe_from_iterator(
            iter(corpus), vocab_size=cfg.vocab_size, min_frequency=1
        )

    def generate(self, prompt: str, max_new_tokens: int = 8) -> str:
        import torch

        ids = self.tokenizer.encode(prompt)[-16:] or [0]
        x = torch.tensor([ids], dtype=torch.long)
        out = self.model.generate(x, max_new_tokens=max_new_tokens, temperature=0.0)
        return self.tokenizer.decode(out[0, len(ids):].tolist(), skip_special_tokens=True)

    def loglikelihood(self, text: str, max_len: int = 64) -> tuple[float, int]:
        import torch

        ids = self.tokenizer.encode(text)[:max_len]
        if len(ids) < 2:
            return 0.0, 0
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits, _ = self.model(x)
        logp = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        target = x[:, 1:]
        g = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return float(g.sum().item()), int(target.numel())


# ══════════════════════════════════════════════════════════════════
#  Default perplexity corpus
# ══════════════════════════════════════════════════════════════════
_DEFAULT_PPL_TEXTS = [
    "Indonesia adalah negara kepulauan terbesar di dunia.",
    "Pancasila adalah dasar negara Republik Indonesia yang terdiri dari lima sila.",
    "Bahasa Indonesia adalah bahasa resmi Republik Indonesia.",
    "Borobudur merupakan candi Buddha terbesar di dunia.",
    "Ekonomi Indonesia adalah salah satu yang terbesar di Asia Tenggara.",
]


def _load_ppl_corpus(name_or_path: str | None, limit: int) -> list[str]:
    if not name_or_path:
        return _DEFAULT_PPL_TEXTS[:limit]
    if os.path.isfile(name_or_path):
        texts = []
        with open(name_or_path, encoding="utf-8") as f:
            for line in f:
                texts.append(line.strip())
                if len(texts) >= limit:
                    break
        return texts

    from datasets import load_dataset

    ds = load_dataset(name_or_path, split="train", streaming=True)
    texts = []
    for rec in ds:
        text = rec.get("text") or rec.get("article") or rec.get("content")
        if text:
            texts.append(text)
        if len(texts) >= limit:
            break
    return texts


# ══════════════════════════════════════════════════════════════════
#  Entrypoint
# ══════════════════════════════════════════════════════════════════
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AksaraLLM 20B eval harness")
    ap.add_argument("--model", default=None,
                    help="Path or HF id of an aksaraLLMModel or a HF model.")
    ap.add_argument("--tokenizer", default=None,
                    help="Path or HF id of a matching tokenizer. Defaults to --model.")
    ap.add_argument("--hf-format", action="store_true",
                    help="Treat --model as an HF ``AutoModelForCausalLM`` (fallback).")
    ap.add_argument("--ppl-dataset", default=None,
                    help="HF dataset id or local JSONL/text file for perplexity.")
    ap.add_argument("--ppl-limit", type=int, default=1000)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--task-limit", type=int, default=50)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.dry_run:
        adapter = _DryRunAdapter()
    else:
        if args.model is None:
            ap.error("--model is required unless --dry-run.")
        tok = args.tokenizer or args.model
        if args.hf_format:
            adapter = HFAdapter(args.model, tok)
        else:
            adapter = AksaraAdapter(args.model, tok)

    out: dict = {"model": args.model or "<dry-run>"}

    # Perplexity
    if args.dry_run:
        texts = _DEFAULT_PPL_TEXTS
    else:
        texts = _load_ppl_corpus(args.ppl_dataset, args.ppl_limit)
    log(f"Perplexity on {len(texts)} samples...")
    out["perplexity"] = perplexity(adapter, texts)
    log(f"  ppl={out['perplexity']['perplexity']:.3f}")

    # Identity & English leakage
    out["identity"] = identity_eval(adapter, IDENTITY_PROMPTS)
    log(f"  identity_acc={out['identity']['identity_accuracy']:.2f} "
        f"english_leak={out['identity']['english_leak_frac_mean']:.3f}")

    # Aksara Indo Bench — skipped on dry-run because it downloads remote data.
    if not args.dry_run:
        out["aksara_indo_bench"] = run_aksara_indo_bench(
            adapter, tasks=args.tasks, limit=args.task_limit
        )

    # Persist JSON + a tiny markdown report.
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    md_path = os.path.splitext(args.out)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# AksaraLLM eval — {out['model']}\n\n")
        ppl = out["perplexity"]
        f.write(f"- Perplexity: **{ppl['perplexity']:.3f}** ({ppl['n_samples']} samples, "
                f"{ppl['n_tokens']} tokens)\n")
        ident = out["identity"]
        f.write(f"- Identity accuracy: **{ident['identity_accuracy']*100:.1f}%**\n")
        f.write(f"- English leak (mean frac in first 50 tokens): "
                f"**{ident['english_leak_frac_mean']*100:.1f}%**\n")
        if "aksara_indo_bench" in out:
            f.write("\n## aksara_indo_bench per-task\n\n")
            for r in out["aksara_indo_bench"]["per_task"]:
                if "error" in r:
                    f.write(f"- {r['name']}: ERROR — {r['error']}\n")
                else:
                    f.write(f"- {r['name']} ({r.get('track','?')}): "
                            f"{r['score']*100:.2f}% (n={r['n']})\n")

    log(f"Wrote {args.out} + {md_path}")
    if args.dry_run:
        log("[dry-run] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
