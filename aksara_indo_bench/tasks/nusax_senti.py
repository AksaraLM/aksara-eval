"""NusaX sentiment — 11 regional languages, 3-way sentiment classification."""
from datasets import load_dataset
from aksara_indo_bench.base import BenchTask

LANGS = ["ace", "ban", "bbc", "bjn", "bug", "ind", "jav", "mad", "min", "nij", "sun"]


class NusaXSenti(BenchTask):
    name = "nusax_senti"
    track = "regional"
    dataset = "indonlp/NusaX-senti"

    def __init__(self, lang="ind"):
        assert lang in LANGS
        self.lang = lang

    def iter_examples(self):
        ds = load_dataset(self.dataset, self.lang, split="test")
        for ex in ds:
            yield ex

    def format_prompt(self, example):
        return (
            "Klasifikasikan sentimen teks berikut ke salah satu: positive, negative, neutral.\n"
            f"Teks: {example['text']}\n"
            "Sentimen:"
        )

    def extract_answer(self, output, example):
        o = output.strip().lower()
        for lab in ("positive", "negative", "neutral"):
            if lab in o:
                return lab
        return "unknown"

    def score_one(self, pred, example):
        return float(pred == example.get("label", ""))
