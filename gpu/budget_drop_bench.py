#!/usr/bin/env python3
"""Sudden budget drop mid-thinking: does warning the model help, versus just cutting it off?

Scenario: a request has been thinking freely. At token p (default 50% of its natural thinking length L) the
serving system's load spikes and the request's remaining thinking allowance drops to W tokens.
Arms (all answer-forcing uses Qwen's early-stop phrase + </think>, unless noted):
  cut_now        end the thinking at p immediately, then answer
  cut_now_hard   same, but with a bare </think> (the abrupt cut many systems use)
  silent_a       do not tell the model; it keeps thinking as it would have, answer forced at p+W
  warn_soft_a    inject "running out of time, wrap up" into the thinking at p; the model continues for at
                 most W tokens; if it closes </think> itself it answers, otherwise the answer is forced at p+W
  warn_count_a   same, but the notice says how many tokens are left
  (a = W as a fraction of L, default 0.25 and 0.1)
silent and warn share the same cap p+W, so their accuracy is directly comparable.

Uses the natural traces (phase_a.jsonl) from an earlier thinking_budget_bench.py run, so no new natural
generation is needed. Needs thinking_budget_bench.py in the same folder (grading, data, prompts).

Usage:  python budget_drop_bench.py --traces ../results/gpu_hard --out drop        (Qwen3-4B, ~25 min on an RTX 4090)
        python budget_drop_bench.py --traces hard --out drop_quick --limit 20
        python budget_drop_bench.py --traces hard --mock --limit 5    (no GPU, checks the pipeline)
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import thinking_budget_bench as TB

WARNS = {
    "warn_soft": "\n\nWait, I'm running out of time. I need to wrap up my reasoning quickly and give the "
                 "final answer soon.\n",
    "warn_count": "\n\nI only have about {W} tokens of thinking left, so I must finish my reasoning quickly "
                  "and move to the final answer.\n",
}
HARD_CLOSE = "\n</think>\n\n"


class Engine:
    def __init__(self, model, max_len, gpu_util, seed, max_num_seqs):
        import os
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM
        from transformers import AutoTokenizer
        self.llm = LLM(model=model, enable_prefix_caching=True, gpu_memory_utilization=gpu_util,
                       max_model_len=max_len, seed=seed, max_num_seqs=max_num_seqs)
        self.tok = AutoTokenizer.from_pretrained(model)

    def generate(self, prompts, max_tokens, seed, stop_ids=None):
        """max_tokens: one int per prompt. Returns (token_ids, finish_reason) per prompt."""
        from vllm import SamplingParams
        sps = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=int(m), seed=seed,
                              stop_token_ids=stop_ids) for m in max_tokens]
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], sps)
        return [(list(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outs]


class Mock:
    """No GPU: real tokenizer, fake model. Only checks that the pipeline runs end to end."""

    def __init__(self, model, problems, seed):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.problems, self.seed = problems, seed
        self.t_close = self.tok.convert_tokens_to_ids("</think>")
        self.filler = self.tok.encode(" step", add_special_tokens=False)[0]

    def generate(self, prompts, max_tokens, seed, stop_ids=None):
        out = []
        for ids, m in zip(prompts, max_tokens):
            h = random.Random(hash((len(ids), seed)) & 0xFFFFFFFF)
            if stop_ids:                                   # thinking continuation
                n = h.randint(1, max(1, int(m)))
                if n < m and h.random() < 0.6:
                    out.append(([self.filler] * n + [self.t_close], "stop"))
                else:
                    out.append(([self.filler] * int(m), "length"))
                continue
            text = self.tok.decode(ids[:400], skip_special_tokens=False)
            gold = next((p["gold"] for p in self.problems if p["q"][:80] in text), "0")
            ans = gold if h.random() < 0.7 else "wrong"
            out.append((self.tok.encode(f"So the answer is \\boxed{{{ans}}}.", add_special_tokens=False), "stop"))
        return out


def boot(d, B=2000):
    g = random.Random(1)
    n = len(d)
    m = sorted(sum(d[g.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [round(m[int(.025 * B)] * 100, 1), round(m[int(.975 * B)] * 100, 1)]


def main():
    # Keep the established --traces workflow intact; --sweep opts into the multi-sample experiment.
    if "--sweep" in sys.argv[1:]:
        sys.argv.remove("--sweep")
        from budget_drop_sweep import main as sweep_main
        return sweep_main()

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--traces", type=Path, default=Path("hard"), help="folder with phase_a.jsonl")
    ap.add_argument("--drop", type=float, default=0.5, help="budget drops after this share of natural thinking")
    ap.add_argument("--allow", type=float, nargs="+", default=[0.25, 0.1],
                    help="remaining allowance after the drop, as a share of natural thinking")
    ap.add_argument("--min-L", type=int, default=400)
    ap.add_argument("--max-think", type=int, default=8192)
    ap.add_argument("--answer-max", type=int, default=4096)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("drop"))
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    assert all(a.drop + x < 1.0 for x in a.allow), "drop + allow must stay below 1"
    a.out.mkdir(parents=True, exist_ok=True)

    gens = {}
    for line in (a.traces / "phase_a.jsonl").open():
        r = json.loads(line)
        gens[r["id"]] = r["gen"]
    dataset = "gsm8k" if next(iter(gens)).startswith("gsm8k/") else "math500"
    items = {p["id"]: p for p in TB.load_problems(dataset, 10 ** 6, 0)}
    problems = [items[i] for i in gens if i in items]
    if a.limit:
        problems = problems[:a.limit]
    max_len = 1536 + a.max_think + 256 + a.answer_max
    eng = Mock(a.model, problems, a.seed) if a.mock else Engine(a.model, max_len, a.gpu_util, a.seed,
                                                                  a.max_num_seqs)
    tok = eng.tok
    t_open, t_close = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    suffix, hard, nl2 = enc(TB.EARLY_STOP), enc(HARD_CLOSE), enc("\n\n")

    def prompt_ids(q):
        text = tok.apply_chat_template([{"role": "user", "content": q + TB.INSTR}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        return enc(text)

    recs = []
    for p in problems:
        g = gens[p["id"]]
        if t_close not in g:
            continue                                          # capped: natural length unknown
        start = g.index(t_open) + 1 if t_open in g else 0
        end = g.index(t_close)
        L = end - start
        if L < a.min_L:
            continue
        nat = TB.grade(dataset, tok.decode(g[end + 1:], skip_special_tokens=True), p["gold"])
        recs.append({"p": p, "prompt": prompt_ids(p["q"]), "gen": g, "start": start, "L": L, "nat": nat})
    print(f"{len(recs)} problems with a natural thinking length >= {a.min_L} tokens ({dataset})", flush=True)

    # ---------------- stage 1: warned continuations
    cont_jobs = []                                            # (rec, key, W, prompt)
    for i, r in enumerate(recs):
        p = int(a.drop * r["L"])
        for x in a.allow:
            W = max(16, int(x * r["L"]))
            for name, text in WARNS.items():
                cont_jobs.append((i, f"{name}_{x}", W, r["prompt"] + r["gen"][:r["start"] + p] +
                                  enc(text.format(W=W))))
    t0 = time.time()
    conts = eng.generate([j[3] for j in cont_jobs], [j[2] for j in cont_jobs], a.seed + 7, stop_ids=[t_close])
    print(f"stage 1: {len(cont_jobs)} warned continuations in {time.time() - t0:.0f}s", flush=True)

    # ---------------- stage 2: answers for every arm
    ans_jobs = []                                             # (rec, key, think_used, closed_itself, prompt)
    for i, r in enumerate(recs):
        p = int(a.drop * r["L"])
        base = r["prompt"] + r["gen"][:r["start"] + p]
        ans_jobs.append((i, "cut_now", p, None, base + suffix))
        ans_jobs.append((i, "cut_now_hard", p, None, base + hard))
        for x in a.allow:
            W = max(16, int(x * r["L"]))
            ans_jobs.append((i, f"silent_{x}", p + W, None, r["prompt"] + r["gen"][:r["start"] + p + W] + suffix))
    for (i, key, W, cprompt), (ids, why) in zip(cont_jobs, conts):
        p = int(a.drop * recs[i]["L"])
        warn_len = len(cprompt) - len(recs[i]["prompt"]) - recs[i]["start"] - p
        if ids and ids[-1] == t_close:
            ans_jobs.append((i, key, p + warn_len + len(ids) - 1, True, cprompt + ids + nl2))
        elif why == "stop":                                   # stopped on </think> but token not returned
            ans_jobs.append((i, key, p + warn_len + len(ids), True, cprompt + ids + [t_close] + nl2))
        else:
            ans_jobs.append((i, key, p + warn_len + len(ids), False, cprompt + ids + suffix))
    t0 = time.time()
    outs = eng.generate([j[4] for j in ans_jobs], [a.answer_max] * len(ans_jobs), a.seed + 1)
    print(f"stage 2: {len(ans_jobs)} answers in {time.time() - t0:.0f}s", flush=True)

    res = {}
    with (a.out / "jobs.jsonl").open("w") as f:
        for (i, key, used, closed, _), (ids, _) in zip(ans_jobs, outs):
            text = tok.decode(ids, skip_special_tokens=True)
            ok = TB.grade(dataset, text, recs[i]["p"]["gold"])
            row = {"id": recs[i]["p"]["id"], "key": key, "L": recs[i]["L"], "think_used": used,
                   "closed_itself": closed, "ans_tokens": len(ids), "correct": ok, "tail": text[-200:]}
            res.setdefault(key, {})[i] = row
            f.write(json.dumps(row) + "\n")

    n = len(recs)
    keys = ["cut_now", "cut_now_hard"] + [f"{k}_{x}" for x in a.allow for k in ("silent", *WARNS)]
    summ = {"model": a.model, "dataset": dataset, "n": n, "drop_at": a.drop, "allow": a.allow, "mock": a.mock,
            "natural_accuracy": round(100 * sum(r["nat"] for r in recs) / n, 1), "arms": {}, "paired_ci_pp": {}}
    for k in keys:
        rows = [res[k][i] for i in range(n)]
        arm = {"accuracy": round(100 * sum(r["correct"] for r in rows) / n, 1),
               "think_tokens_vs_natural": round(statistics.mean(r["think_used"] / r["L"] for r in rows), 3),
               "mean_answer_tokens": round(statistics.mean(r["ans_tokens"] for r in rows))}
        if k.startswith("warn"):
            arm["closed_itself_share"] = round(sum(bool(r["closed_itself"]) for r in rows) / n, 3)
        summ["arms"][k] = arm
    d = lambda x, y: boot([int(res[x][i]["correct"]) - int(res[y][i]["correct"]) for i in range(n)])
    summ["paired_ci_pp"]["cut_now - cut_now_hard"] = d("cut_now", "cut_now_hard")
    for x in a.allow:
        for w in WARNS:
            summ["paired_ci_pp"][f"{w}_{x} - silent_{x}"] = d(f"{w}_{x}", f"silent_{x}")
    (a.out / "summary.json").write_text(json.dumps(summ, indent=2))

    print(f"\nnatural accuracy (full thinking): {summ['natural_accuracy']}%   n={n}")
    print(f"budget drops at {a.drop:.0%} of natural thinking")
    for k in keys:
        s = summ["arms"][k]
        extra = f"  closed by itself {s['closed_itself_share']:.0%}" if "closed_itself_share" in s else ""
        print(f"  {k:18s} {s['accuracy']:5.1f}%   thinking used {s['think_tokens_vs_natural']:.0%} of natural{extra}")
    for k, v in summ["paired_ci_pp"].items():
        print(f"  {k:32s} 95% CI {v[0]:+.1f} to {v[1]:+.1f} points")
    print(f"results in {a.out.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
