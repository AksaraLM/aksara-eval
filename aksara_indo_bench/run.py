"""CLI runner: python -m aksara_indo_bench.run --model <hf_model_id> --tasks all"""
import argparse
import json
from statistics import mean

from aksara_indo_bench.tasks.indommlu import IndoMMLU
from aksara_indo_bench.tasks.copal_id import CopalID
from aksara_indo_bench.tasks.nusax_senti import NusaXSenti, LANGS
from aksara_indo_bench.tasks.aksara_safety import AksaraSafety

TASKS = {
    "indommlu": IndoMMLU(),
    "copal_id": CopalID(),
    "aksara_safety": AksaraSafety(),
    **{f"nusax_senti.{l}": NusaXSenti(l) for l in LANGS},
}

TRACK_WEIGHTS = {
    "knowledge": 0.25,
    "culture": 0.15,
    "reasoning": 0.15,
    "reading": 0.15,
    "regional": 0.20,
    "safety": 0.05,
    "generation": 0.05,
}


class HFModel:
    """Minimal wrapper; replace with transformers / vllm / API of your choice."""

    def __init__(self, model_id, max_new_tokens=16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.mdl = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt):
        inp = self.tok(prompt, return_tensors="pt").to(self.mdl.device)
        out = self.mdl.generate(**inp, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0, inp.input_ids.shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="all", help="comma-separated; 'all' = every task")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    chosen = list(TASKS.keys()) if args.tasks == "all" else args.tasks.split(",")
    model = HFModel(args.model)

    by_track = {}
    all_results = []
    for key in chosen:
        if key not in TASKS:
            print(f"[skip] unknown task {key}")
            continue
        task = TASKS[key]
        print(f"[run] {key} (track={task.track})")
        r = task.run(model, limit=args.limit)
        all_results.append({"name": r.name, "track": r.track, "score": r.score, "n": r.n})
        by_track.setdefault(r.track, []).append(r.score)
        print(f"  score={r.score:.4f} n={r.n}")

    composite = 0.0
    for track, w in TRACK_WEIGHTS.items():
        if by_track.get(track):
            composite += w * mean(by_track[track])

    payload = {"model": args.model, "per_task": all_results, "composite": composite}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n=== COMPOSITE (aksara_score) = {composite:.4f} ===")


if __name__ == "__main__":
    main()
