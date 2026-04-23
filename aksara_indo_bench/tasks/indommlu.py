"""IndoMMLU — 63-subject multiple choice knowledge benchmark for Indonesian."""
from datasets import load_dataset
from aksara_indo_bench.base import BenchTask


class IndoMMLU(BenchTask):
    name = "indommlu"
    track = "knowledge"
    dataset = "IndoNLP/indommlu"

    def iter_examples(self):
        ds = load_dataset(self.dataset, split="test")
        for ex in ds:
            yield ex

    def format_prompt(self, example):
        q = example["question"]
        choices = [example[k] for k in ("A", "B", "C", "D") if k in example]
        labels = "ABCD"[: len(choices)]
        lines = [f"Pertanyaan: {q}"]
        for lab, ch in zip(labels, choices):
            lines.append(f"{lab}. {ch}")
        lines.append("Jawaban:")
        return "\n".join(lines)

    def extract_answer(self, output, example):
        for ch in output.strip():
            if ch in "ABCD":
                return ch
        return "X"

    def score_one(self, pred, example):
        return float(pred == example.get("answer", ""))
