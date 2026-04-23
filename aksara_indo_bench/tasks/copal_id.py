"""COPAL-ID — pragmatic/causal reasoning for Indonesian."""
from datasets import load_dataset
from aksara_indo_bench.base import BenchTask


class CopalID(BenchTask):
    name = "copal_id"
    track = "reasoning"
    dataset = "haryoaw/COPAL"

    def iter_examples(self):
        ds = load_dataset(self.dataset, split="test")
        for ex in ds:
            yield ex

    def format_prompt(self, example):
        premise = example["premise"]
        q = "Apa penyebabnya?" if example.get("question") == "cause" else "Apa akibatnya?"
        return (
            f"Premis: {premise}\n"
            f"{q}\n"
            f"A. {example['choice1']}\n"
            f"B. {example['choice2']}\n"
            f"Jawaban:"
        )

    def extract_answer(self, output, example):
        for ch in output.strip():
            if ch in "AB":
                return 0 if ch == "A" else 1
        return -1

    def score_one(self, pred, example):
        return float(pred == example.get("label", -999))
