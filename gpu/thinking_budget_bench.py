#!/usr/bin/env python3
"""Lifeline GPU kill test: how much accuracy does a reasoning model lose when its thinking is cut short?

Every Lifeline result so far assumed an accuracy-vs-thinking curve. This script measures the real one.

Method (one GPU, vLLM, prefix caching):
  Phase A  For each problem, let the model think naturally (up to --max-think tokens) and answer.
           Records the natural thinking length L and whether the natural answer is right.
  Phase B  Re-use that same thinking trace, cut it short, force the answer, and grade again:
             * absolute budgets  (--budgets, tokens)            -> accuracy vs budget
             * fractions of L    (--fracs, e.g. 0.25 0.5 0.75)  -> accuracy vs share of natural thinking
           The forced answer uses Qwen's documented early-stop phrase followed by </think>.

Kill rule (from the Lifeline report): if cutting thinking to 50% of its natural length loses more than
10 accuracy points on average, the "elastic reasoning" premise is too weak.

Outputs in --out:
  rows.jsonl     one line per problem: natural length, natural correctness, correctness at each budget/fraction
  summary.json   curves, kill-rule verdict, natural-length stats, throughput
  phase_a.jsonl  raw Phase A traces (lets --resume skip Phase A)

Usage (Linux or WSL2 with vLLM installed):
  pip install datasets math-verify
  python thinking_budget_bench.py --n 200                       # Qwen3-4B on MATH-500, ~45-60 min on an RTX 4090
  python thinking_budget_bench.py --n 60 --out quick            # smoke run, ~15 min
  python thinking_budget_bench.py --model Qwen/Qwen3-8B --max-think 6144 --gpu-util 0.95
  python thinking_budget_bench.py --mock --n 30                 # no GPU: checks the pipeline end to end
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
import time
from pathlib import Path

INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."
EARLY_STOP = ("\n\nConsidering the limited time by the user, I have to give the solution "
              "based on the thinking directly now.\n</think>\n\n")


# ----------------------------------------------------------------------------- grading
def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} or \\fbox{...}, with nested braces."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


def _norm(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""),
                 ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""),
                 ("\\%", ""), ("%", ""), (" ", "")):
        s = s.replace(a, b)
    s = s.rstrip(".")
    return s


def _num(s: str):
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


try:
    from math_verify import parse as mv_parse, verify as mv_verify
    HAVE_MV = True
except Exception:          # math-verify missing: fall back to string/number matching
    HAVE_MV = False


def grade(dataset: str, response: str, gold: str) -> bool:
    if dataset == "gsm8k":
        pred = last_boxed(response)
        if pred is None:
            nums = re.findall(r"-?\d[\d,]*\.?\d*", response)
            pred = nums[-1] if nums else None
        if pred is None:
            return False
        a, b = _num(_norm(pred)), _num(gold)
        return a is not None and b is not None and abs(a - b) < 1e-6
    pred = last_boxed(response)
    if pred is None:
        return False
    if _norm(pred) == _norm(gold):
        return True
    a, b = _num(_norm(pred)), _num(_norm(gold))
    if a is not None and b is not None:
        return abs(a - b) < 1e-6
    if HAVE_MV:
        try:
            return bool(mv_verify(mv_parse("$" + gold + "$"), mv_parse("$" + pred + "$")))
        except Exception:
            return False
    return False


# ----------------------------------------------------------------------------- data
def load_problems(name: str, n: int, seed: int, levels=None):
    from datasets import load_dataset
    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [{"id": r["unique_id"], "q": r["problem"], "gold": r["answer"], "level": r["level"]} for r in ds]
    elif name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = [{"id": f"gsm8k/{i}", "q": r["question"], "gold": r["answer"].split("####")[-1].strip(),
                  "level": 0} for i, r in enumerate(ds)]
    else:
        raise SystemExit(f"unknown dataset {name}")
    if levels:
        items = [it for it in items if it["level"] in levels]
    random.Random(seed).shuffle(items)
    return items[:n]


# ----------------------------------------------------------------------------- engines
class VLLMEngine:
    def __init__(self, model, max_len, gpu_util, seed, max_num_seqs=16):
        import os
        # FlashInfer's top-k/top-p sampler JIT-compiles with nvcc, which WSL usually lacks.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM
        from transformers import AutoTokenizer
        # Capping concurrent sequences stops vLLM from admitting more long traces than the KV cache
        # holds, which otherwise causes repeated preempt-and-recompute (slow Phase A).
        self.llm = LLM(model=model, enable_prefix_caching=True, gpu_memory_utilization=gpu_util,
                       max_model_len=max_len, seed=seed, max_num_seqs=max_num_seqs)
        self.tok = AutoTokenizer.from_pretrained(model)

    def generate(self, prompts, max_tokens, seed, temperature=0.6, top_p=0.95, top_k=20):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k,
                            max_tokens=max_tokens, seed=seed)
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], sp)
        return [list(o.outputs[0].token_ids) for o in outs]


class MockEngine:
    """No GPU. Uses the real tokenizer and fakes a model whose accuracy grows with thinking length."""

    def __init__(self, model, problems, seed):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.problems = problems
        self.rng = random.Random(seed)
        self.t_open = self.tok.convert_tokens_to_ids("<think>")
        self.t_close = self.tok.convert_tokens_to_ids("</think>")
        self.filler = self.tok.encode(" step", add_special_tokens=False)[0]

    def _problem(self, text):
        for p in self.problems:
            if p["q"] in text:
                return p
        raise RuntimeError("mock: problem not found in prompt")

    def generate(self, prompts, max_tokens, seed, **_):
        out = []
        for ids in prompts:
            text = self.tok.decode(ids, skip_special_tokens=False)
            p = self._problem(text)
            h = random.Random(hash((p["id"], seed)) & 0xFFFFFFFF)
            tau = h.uniform(300, 3000)
            if self.t_open in ids:                       # Phase B: thinking already in the prompt
                t = ids.count(self.filler)
                ok = h.random() < 1 - 0.85 * math.exp(-t / tau)
                ans = p["gold"] if ok else "wrong"
                out.append(self.tok.encode(f"So the answer is \\boxed{{{ans}}}.", add_special_tokens=False))
                continue
            L = int(h.uniform(150, 11000))               # natural thinking length
            ok = h.random() < 1 - 0.85 * math.exp(-L / tau)
            ans = p["gold"] if ok else "wrong"
            seq = [self.t_open] + [self.filler] * L + [self.t_close] + \
                self.tok.encode(f"\n\nSo the answer is \\boxed{{{ans}}}.", add_special_tokens=False)
            out.append(seq[:max_tokens])
        return out


# ----------------------------------------------------------------------------- pipeline
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dataset", default="math500", choices=["math500", "gsm8k"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-think", type=int, default=8192)
    ap.add_argument("--answer-max", type=int, default=4096)
    ap.add_argument("--budgets", type=int, nargs="+", default=[0, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--levels", type=int, nargs="+", default=None, help="MATH-500 levels to keep, e.g. 4 5")
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("thinking_budget_results"))
    ap.add_argument("--resume", action="store_true", help="reuse phase_a.jsonl if present")
    ap.add_argument("--mock", action="store_true", help="no GPU: fake model, real tokenizer and data")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    problems = load_problems(a.dataset, a.n, a.seed, a.levels)
    max_len = 1536 + a.max_think + 64 + a.answer_max
    eng = MockEngine(a.model, problems, a.seed) if a.mock else VLLMEngine(a.model, max_len, a.gpu_util, a.seed,
                                                                            a.max_num_seqs)
    tok = eng.tok
    t_open = tok.convert_tokens_to_ids("<think>")
    t_close = tok.convert_tokens_to_ids("</think>")
    suffix = tok.encode(EARLY_STOP, add_special_tokens=False)

    def prompt_ids(q):
        text = tok.apply_chat_template([{"role": "user", "content": q + INSTR}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        return tok.encode(text, add_special_tokens=False)

    prompts = [prompt_ids(p["q"]) for p in problems]
    keep = [i for i, pr in enumerate(prompts) if len(pr) + a.max_think + len(suffix) + 2 + a.answer_max <= max_len]
    if len(keep) < len(problems):
        print(f"skipping {len(problems) - len(keep)} problems whose prompt is too long for the context window")
        problems, prompts = [problems[i] for i in keep], [prompts[i] for i in keep]

    # ---------------- Phase A: natural thinking
    pa_path = a.out / "phase_a.jsonl"
    if a.resume and pa_path.exists():
        gens = {json.loads(l)["id"]: json.loads(l)["gen"] for l in pa_path.open()}
        gens = [gens[p["id"]] for p in problems]
        a_secs, a_tok = None, None
        print(f"Phase A: reused {len(gens)} traces from {pa_path}")
    else:
        t0 = time.time()
        gens = eng.generate(prompts, a.max_think + a.answer_max, a.seed)
        a_secs, a_tok = time.time() - t0, sum(len(g) for g in gens)
        with pa_path.open("w") as f:
            for p, g in zip(problems, gens):
                f.write(json.dumps({"id": p["id"], "gen": g}) + "\n")
        print(f"Phase A: {len(gens)} traces, {a_tok} tokens in {a_secs:.0f}s ({a_tok / a_secs:.0f} tok/s)")

    recs = []
    for p, pr, g in zip(problems, prompts, gens):
        start = g.index(t_open) + 1 if t_open in g else 0
        end = g.index(t_close) if t_close in g else None
        capped = end is None or end - start > a.max_think
        L = a.max_think if capped else end - start
        recs.append({"p": p, "prompt": pr, "gen": g, "start": start, "end": end, "L": L, "capped": capped})

    # ---------------- Phase B: cut thinking, force the answer
    jobs = []          # (rec index, key, thinking tokens kept)
    for i, r in enumerate(recs):
        if r["capped"]:
            jobs.append((i, "natural", r["L"]))
        for b in a.budgets:
            if b < r["L"]:
                jobs.append((i, f"b{b}", b))
        for fr in a.fracs:
            jobs.append((i, f"f{fr}", int(fr * r["L"])))
        if not r["capped"]:
            jobs.append((i, "full", r["L"]))   # control: same full thinking, but answer forced
    batch = [recs[i]["prompt"] + recs[i]["gen"][:recs[i]["start"] + k] + suffix for i, _, k in jobs]
    t0 = time.time()
    outs = eng.generate(batch, a.answer_max, a.seed + 1) if batch else []
    b_secs, b_tok = time.time() - t0, sum(len(o) for o in outs)
    print(f"Phase B: {len(batch)} forced answers, {b_tok} tokens in {b_secs:.0f}s")

    res = {i: {} for i in range(len(recs))}
    detail = {i: {} for i in range(len(recs))}
    with (a.out / "phase_b.jsonl").open("w") as f:
        for (i, key, k), o in zip(jobs, outs):
            text = tok.decode(o, skip_special_tokens=True)
            ok = grade(a.dataset, text, recs[i]["p"]["gold"])
            res[i][key] = ok
            detail[i][key] = {"kept": k, "ans_tokens": len(o), "hit_cap": len(o) >= a.answer_max,
                              "boxed": last_boxed(text) is not None, "correct": ok}
            f.write(json.dumps({"id": recs[i]["p"]["id"], "key": key, **detail[i][key],
                                "tail": text[-300:]}) + "\n")

    rows = []
    for i, r in enumerate(recs):
        if r["capped"]:
            nat = res[i]["natural"]
        else:
            nat = grade(a.dataset, tok.decode(r["gen"][r["end"] + 1:], skip_special_tokens=True), r["p"]["gold"])
        nat_ans = (detail[i]["natural"]["ans_tokens"] if r["capped"] else len(r["gen"]) - r["end"] - 1)
        cost = {k: d["kept"] + d["ans_tokens"] for k, d in detail[i].items()}
        cost["natural"] = r["L"] + nat_ans
        row = {"id": r["p"]["id"], "level": r["p"]["level"], "L": r["L"], "capped": r["capped"],
               "natural": nat, "forced_full": res[i]["natural"] if r["capped"] else res[i]["full"],
               "tokens": {"natural": cost["natural"],
                          **{f"b{b}": (cost[f"b{b}"] if b < r["L"] else cost["natural"]) for b in a.budgets},
                          **{f"f{fr}": cost[f"f{fr}"] for fr in a.fracs}},
               "hit_cap": {k: d["hit_cap"] for k, d in detail[i].items()},
               "abs": {str(b): (res[i][f"b{b}"] if b < r["L"] else nat) for b in a.budgets},
               "frac": {str(fr): res[i][f"f{fr}"] for fr in a.fracs}}
        rows.append(row)
    with (a.out / "rows.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # ---------------- summary
    n = len(rows)
    pct = lambda xs: round(100 * sum(xs) / max(1, len(xs)), 1)
    acc_nat = pct([r["natural"] for r in rows])
    abs_curve = {str(b): pct([r["abs"][str(b)] for r in rows]) for b in a.budgets}
    frac_curve = {str(fr): pct([r["frac"][str(fr)] for r in rows]) for fr in a.fracs}
    frac_curve["1.0"] = acc_nat
    drop_half = round(acc_nat - frac_curve.get("0.5", float("nan")), 1) if "0.5" in frac_curve else None
    acc_forced_full = pct([r["forced_full"] for r in rows])
    drop_half_ctrl = round(acc_forced_full - frac_curve["0.5"], 1) if "0.5" in frac_curve else None
    mean_tok = lambda key: round(statistics.mean(r["tokens"][key] for r in rows))
    tokens = {"natural": mean_tok("natural"), **{f"b{b}": mean_tok(f"b{b}") for b in a.budgets},
              **{f"f{fr}": mean_tok(f"f{fr}") for fr in a.fracs}}
    keys = sorted({k for r in rows for k in r["hit_cap"]})
    cap_share = {k: round(sum(r["hit_cap"].get(k, False) for r in rows) /
                          max(1, sum(k in r["hit_cap"] for r in rows)), 3) for k in keys}
    Ls = sorted(r["L"] for r in rows)
    by_level = {}
    for lv in sorted({r["level"] for r in rows}):
        rs = [r for r in rows if r["level"] == lv]
        by_level[str(lv)] = {"n": len(rs), "natural": pct([r["natural"] for r in rs]),
                             **({"half": pct([r["frac"]["0.5"] for r in rs])} if "0.5" in frac_curve else {}),
                             "median_L": statistics.median(r["L"] for r in rs)}
    summary = {
        "model": a.model, "dataset": a.dataset, "n": n, "max_think": a.max_think, "mock": a.mock,
        "grader": "math-verify + string/number match" if HAVE_MV else "string/number match only",
        "accuracy_natural": acc_nat,
        "accuracy_by_budget_tokens": abs_curve,
        "accuracy_by_fraction_of_natural": frac_curve,
        "accuracy_forced_at_full_length": acc_forced_full,
        "drop_at_half_points": drop_half,
        "drop_at_half_vs_forced_full_points": drop_half_ctrl,
        "mean_total_tokens_think_plus_answer": tokens,
        "forced_answer_hit_cap_share": cap_share,
        "answer_max": a.answer_max,
        "kill_rule": None if drop_half is None else (
            "ELASTIC PREMISE TOO WEAK (drop > 10 points)" if drop_half > 10 else "PASSES (drop <= 10 points)"),
        "natural_thinking_tokens": {"median": statistics.median(Ls), "p10": Ls[int(0.1 * (n - 1))],
                                    "p90": Ls[int(0.9 * (n - 1))],
                                    "capped_share": round(sum(r["capped"] for r in rows) / n, 3)},
        "by_level": by_level,
        "throughput": {"phase_a_tok_per_s": round(a_tok / a_secs) if a_secs else None,
                       "phase_b_secs": round(b_secs, 1), "phase_b_tokens": b_tok},
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== accuracy vs thinking budget ===")
    print(f"natural (median {summary['natural_thinking_tokens']['median']} tokens): {acc_nat}%")
    for b, v in abs_curve.items():
        print(f"  budget {b:>5} tokens : {v}%")
    for fr, v in frac_curve.items():
        print(f"  {float(fr):>4.0%} of natural  : {v}%")
    print(f"  forced at 100% (control): {acc_forced_full}%")
    print("mean tokens (think + answer):", tokens)
    print("share of forced answers hitting the answer cap:", cap_share)
    print(f"\nkill rule: drop at 50% = {drop_half} points vs natural, "
          f"{drop_half_ctrl} vs forced-at-100% -> {summary['kill_rule']}")
    print(f"results in {a.out.resolve()}  (send me summary.json and rows.jsonl)")


if __name__ == "__main__":
    sys.exit(main())
