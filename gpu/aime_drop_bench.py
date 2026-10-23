#!/usr/bin/env python3
"""Sudden thinking-budget drop on AIME 2026, with several samples per problem.

Same test as budget_drop_bench.py, but on problems the model cannot have seen in training (AIME 2026 was held
in February 2026, after Qwen3 was released) and that are hard enough to avoid the ceiling of the MATH run.
Every problem is sampled --samples times; each sample gets its own natural trace and its own budget drop.

  Phase A  natural thinking for every (problem, sample), up to the model's context window
  Stage 1  warned continuations after the drop (same arms as budget_drop_bench.py)
  Stage 2  forced answers for every arm, plus no_think (answer with no thinking at all)

Samples run in rounds (default: 2 samples of every problem per round), and each round finishes Phase A and the
whole drop test for its traces before the next starts. A run stopped after N rounds therefore already has
complete results for those rounds, and --resume continues from there. CIs are bootstrapped over problems (all samples of a problem move
together), because samples of the same problem are not independent.

Usage:  python aime_drop_bench.py --out aime_drop                        (30 problems x 8 samples, overnight)
        python aime_drop_bench.py --out aime_quick --samples 1 --limit 2 (quick check)
        python aime_drop_bench.py --out aime_drop --resume               (continue a stopped run)
        python aime_drop_bench.py --analyze aime_drop                    (print the analysis again)
        python aime_drop_bench.py --mock --samples 2 --limit 4 --out mock (no GPU, checks the pipeline)
        python aime_drop_bench.py --traces aime_drop --out aime_final --answer-mode final
                                  (reuse finished natural traces; every arm, natural included, may only
                                   write the final answer)

--answer-mode final: after </think> every arm, natural included, is given "The final answer is \boxed{"
and may write only --final-max tokens, so the model cannot redo its reasoning in the answer. The prompt is
unchanged, so natural traces from a full-mode run can be reused with --traces.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import thinking_budget_bench as TB
from budget_drop_bench import HARD_CLOSE, WARNS

FINAL_PREFIX = "The final answer is \\boxed{"


def load_aime(name: str):
    from datasets import load_dataset
    ds = load_dataset(name, split="train")
    tag = name.split("/")[-1]
    return [{"id": f"{tag}/{r['problem_idx']}", "q": r["problem"], "gold": str(r["answer"])} for r in ds]


class Engine:
    def __init__(self, model, max_len, gpu_util, max_num_seqs):
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM
        from transformers import AutoTokenizer
        self.llm = LLM(model=model, enable_prefix_caching=True, gpu_memory_utilization=gpu_util,
                       max_model_len=max_len, max_num_seqs=max_num_seqs)
        self.tok = AutoTokenizer.from_pretrained(model)

    def generate(self, prompts, max_tokens, seeds, stop_ids=None):
        """One max_tokens and one seed per prompt. Returns (token_ids, finish_reason) per prompt."""
        from vllm import SamplingParams
        sps = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=int(m), seed=int(s),
                              stop_token_ids=stop_ids) for m, s in zip(max_tokens, seeds)]
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], sps)
        return [(list(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outs]


class Mock:
    """No GPU: real tokenizer and data, fake model. Only checks that the pipeline runs end to end."""

    def __init__(self, model, problems):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.problems = problems
        self.t_open = self.tok.convert_tokens_to_ids("<think>")
        self.t_close = self.tok.convert_tokens_to_ids("</think>")
        self.filler = self.tok.encode(" step", add_special_tokens=False)[0]

    def _ans(self, ids, ok):
        text = self.tok.decode(ids[:800], skip_special_tokens=False)
        gold = next((p["gold"] for p in self.problems if p["q"][:60] in text), "0")
        return self.tok.encode(f"\n\nSo the answer is \\boxed{{{gold if ok else 'wrong'}}}.",
                               add_special_tokens=False)

    def generate(self, prompts, max_tokens, seeds, stop_ids=None):
        out = []
        for ids, m, s in zip(prompts, max_tokens, seeds):
            h = random.Random((len(ids) * 1_000_003 + int(s)) & 0xFFFFFFFF)
            m = int(m)
            if stop_ids:                                       # thinking continuation
                n = h.randint(1, max(1, m))
                out.append(([self.filler] * n + [self.t_close], "stop") if n < m and h.random() < 0.6
                           else ([self.filler] * m, "length"))
            elif self.t_open not in ids:                       # Phase A: natural trace
                L = h.randint(200, 9000)
                seq = [self.t_open] + [self.filler] * L + [self.t_close] + self._ans(ids, h.random() < 0.6)
                out.append((seq[:m], "stop" if len(seq) <= m else "length"))
            else:                                              # forced answer
                t = ids.count(self.filler)
                out.append((self._ans(ids, h.random() < min(0.9, 0.25 + t / 9000)), "stop"))
        return out


def arm_keys(allow, final=False):
    keys = ["no_think", "cut_now", "cut_now_hard"] + [f"{k}_{x}" for x in allow for k in ("silent", *WARNS)]
    return (["natural_final"] + keys) if final else keys


# ----------------------------------------------------------------------------- analysis
def cluster_ci(diff_by_problem, B=10000, seed=0):
    """diff_by_problem: {problem: [per-sample differences]}. Resample problems, keep their samples together."""
    groups = list(diff_by_problem.values())
    allv = [x for g in groups for x in g]
    rng = random.Random(seed)
    stats = []
    for _ in range(B):
        tot = cnt = 0
        for _ in groups:
            g = groups[rng.randrange(len(groups))]
            tot += sum(g)
            cnt += len(g)
        stats.append(100 * tot / cnt)
    stats.sort()
    return round(100 * sum(allv) / len(allv), 1), round(stats[int(0.025 * B)], 1), round(stats[int(0.975 * B)], 1)


def analyze(out: Path):
    cfg = json.loads((out / "config.json").read_text())
    nat = [json.loads(l) for l in (out / "natural.jsonl").open()] if (out / "natural.jsonl").exists() else []
    jobs = [json.loads(l) for l in (out / "jobs.jsonl").open()] if (out / "jobs.jsonl").exists() else []
    final = cfg.get("answer_mode", "full") == "final"
    keys = arm_keys(cfg["allow"], final)
    by = collections.defaultdict(dict)
    for j in jobs:
        by[j["sid"]][j["key"]] = j
    sids = sorted(s for s in by if all(k in by[s] for k in keys))
    if not nat or not sids:
        print("nothing to analyse yet")
        return
    n_prob = len({r["id"] for r in nat})
    capped = [r for r in nat if r["capped"]]
    fin = [r for r in nat if not r["capped"]]
    summ = {"model": cfg["model"], "dataset": cfg["dataset"], "problems": n_prob, "samples_planned": cfg["samples"],
            "traces": len(nat), "capped_share": round(len(capped) / len(nat), 3),
            "natural_accuracy_all_traces": round(100 * sum(r["nat"] for r in nat) / len(nat), 1),
            "natural_accuracy_finished_traces": round(100 * sum(r["nat"] for r in fin) / max(1, len(fin)), 1),
            "median_natural_thinking_finished": statistics.median(r["L"] for r in fin) if fin else None,
            "answer_mode": cfg.get("answer_mode", "full"),
            "drop_at": cfg["drop"], "allow": cfg["allow"], "tested_traces": len(sids),
            "tested_problems": len({by[s][keys[0]]["id"] for s in sids}), "arms": {}, "paired_ci_pp": {}}

    def rows(k):
        return [by[s][k] for s in sids]
    L = {s: by[s][keys[0]]["L"] for s in sids}
    nat_ans = {s: by[s][keys[0]]["nat_ans"] for s in sids}
    nat_ok = {s: by[s][keys[0]]["nat_correct"] for s in sids}
    nat_total = sum(L[s] + nat_ans[s] for s in sids)
    summ["arms"]["natural"] = {"accuracy": round(100 * sum(nat_ok.values()) / len(sids), 1),
                               "think_vs_natural": 1.0, "mean_answer_tokens": round(statistics.mean(nat_ans.values())),
                               "total_tokens_vs_natural": 1.0}
    for k in keys:
        if k == "natural_final":                              # already reported as "natural"
            continue
        rr = rows(k)
        arm = {"accuracy": round(100 * sum(r["correct"] for r in rr) / len(rr), 1),
               "think_vs_natural": round(statistics.mean(r["think_used"] / r["L"] for r in rr), 3),
               "mean_answer_tokens": round(statistics.mean(r["ans_tokens"] for r in rr)),
               "total_tokens_vs_natural": round(sum(r["think_used"] + r["ans_tokens"] for r in rr) / nat_total, 3),
               "answers_hit_cap": sum(r["ans_hit_cap"] for r in rr)}
        if k.startswith("warn"):
            w = float(k.rsplit("_", 1)[1])
            arm["closed_itself_share"] = round(sum(bool(r["closed_itself"]) for r in rr) / len(rr), 3)
            arm["grace_window_used_median"] = round(statistics.median(
                (r["think_used"] - cfg["drop"] * r["L"]) / (w * r["L"]) for r in rr), 2)
        summ["arms"][k] = arm

    def ok(s, k):
        return int(nat_ok[s]) if k == "natural" else int(by[s][k]["correct"])

    pairs = [("natural", "no_think"), ("natural", "cut_now")]
    for x in cfg["allow"]:
        pairs.append((f"silent_{x}", "cut_now"))
    hi, lo = max(cfg["allow"]), min(cfg["allow"])
    pairs.append((f"warn_soft_{hi}", "cut_now"))
    for x in cfg["allow"]:
        for w in WARNS:
            pairs.append((f"{w}_{x}", f"silent_{x}"))
    pairs += [(f"warn_soft_{hi}", f"silent_{lo}"), ("natural", f"warn_soft_{hi}"), ("cut_now", "cut_now_hard")]
    for a_, b_ in pairs:
        d = collections.defaultdict(list)
        for s in sids:
            d[by[s][keys[0]]["id"]].append(ok(s, a_) - ok(s, b_))
        summ["paired_ci_pp"][f"{a_} - {b_}"] = cluster_ci(d)
    (out / "summary.json").write_text(json.dumps(summ, indent=2))

    print(f"\n{summ['dataset']}: {n_prob} problems, {len(nat)} natural traces "
          f"({cfg['samples']} samples per problem planned)")
    print(f"  natural accuracy{' with full answers' if final else ''} {summ['natural_accuracy_all_traces']}% of all traces "
          f"({summ['natural_accuracy_finished_traces']}% of the {len(fin)} that finished thinking; "
          f"{len(capped)} ran out of context)")
    print(f"  drop test on {len(sids)} finished traces from {summ['tested_problems']} problems, "
          f"budget drops at {cfg['drop']:.0%} of natural thinking"
          + ("; every arm may write only the final answer" if final else "") + "\n")
    print(f"  {'arm':16s} {'acc %':>6s} {'think':>6s} {'answer':>7s} {'total':>6s} {'closed itself':>14s}")
    for k, s in summ["arms"].items():
        c = f"{s['closed_itself_share']:.0%}" if "closed_itself_share" in s else ""
        print(f"  {k:16s} {s['accuracy']:6.1f} {s['think_vs_natural']:6.2f} {s['mean_answer_tokens']:7d} "
              f"{s['total_tokens_vs_natural']:6.2f} {c:>14s}")
    print("\n  paired difference (pp), 95% CI bootstrapped over problems")
    for k, (m, a_, b_) in summ["paired_ci_pp"].items():
        print(f"  {k:32s} {m:+6.1f}  [{a_:+.1f}, {b_:+.1f}]")
    print(f"\nsummary in {(out / 'summary.json').resolve()}")


# ----------------------------------------------------------------------------- pipeline
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dataset", default="MathArena/aime_2026")
    ap.add_argument("--samples", type=int, default=8, help="natural traces per problem")
    ap.add_argument("--max-len", type=int, default=40960, help="context window (Qwen3-4B: 40,960)")
    ap.add_argument("--drop", type=float, default=0.5)
    ap.add_argument("--allow", type=float, nargs="+", default=[0.25, 0.1])
    ap.add_argument("--min-L", type=int, default=400)
    ap.add_argument("--answer-max", type=int, default=4096)
    ap.add_argument("--answer-mode", choices=["full", "final"], default="full",
                    help="final: after </think> only the final answer may be written (see top of file)")
    ap.add_argument("--final-max", type=int, default=48, help="answer tokens allowed in --answer-mode final")
    ap.add_argument("--traces", type=Path, default=None,
                    help="folder of an earlier run whose phase_a.jsonl (natural traces) should be reused")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=6, help="long traces: keep low to avoid KV preemption")
    ap.add_argument("--round", type=int, default=2, help="samples per problem in each round")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only the first N problems")
    ap.add_argument("--out", type=Path, default=Path("aime_drop"))
    ap.add_argument("--resume", action="store_true", help="keep finished work in --out and continue")
    ap.add_argument("--analyze", type=Path, default=None, help="only print the analysis of this folder")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze(a.analyze)
        return
    assert all(a.drop + x < 1.0 for x in a.allow), "drop + allow must stay below 1"
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps({"model": a.model, "dataset": a.dataset, "samples": a.samples,
                                                   "drop": a.drop, "allow": a.allow, "min_L": a.min_L,
                                                   "max_len": a.max_len, "answer_mode": a.answer_mode,
                                                   "final_max": a.final_max, "mock": a.mock}, indent=2))
    problems = load_aime(a.dataset)
    if a.limit:
        problems = problems[:a.limit]
    eng = Mock(a.model, problems) if a.mock else Engine(a.model, a.max_len, a.gpu_util, a.max_num_seqs)
    tok = eng.tok
    t_open, t_close = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    suffix, hard, nl2 = enc(TB.EARLY_STOP), enc(HARD_CLOSE), enc("\n\n")

    def prompt_ids(q):
        text = tok.apply_chat_template([{"role": "user", "content": q + TB.INSTR}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        return enc(text)

    P = {p["id"]: prompt_ids(p["q"]) for p in problems}
    pidx = {p["id"]: i for i, p in enumerate(problems)}
    byid = {p["id"]: p for p in problems}
    final = a.answer_mode == "final"
    keys = arm_keys(a.allow, final)
    fp = enc(FINAL_PREFIX) if final else []                  # appended after every </think> in final mode
    pa_path, jobs_path = a.out / "phase_a.jsonl", a.out / "jobs.jsonl"
    if a.traces:                                             # reuse natural traces from an earlier run
        import shutil
        if not pa_path.exists():
            shutil.copyfile(a.traces / "phase_a.jsonl", pa_path)
        a.resume = True
    sid = lambda p, k: f"{p['id']}#s{k}"

    done, have = {}, collections.defaultdict(set)            # finished Phase A traces / finished arms
    if a.resume:
        if pa_path.exists():
            for line in pa_path.open():
                r = json.loads(line)
                done[r["sid"]] = r
        if jobs_path.exists():
            rows = [json.loads(l) for l in jobs_path.open()]
            for j in rows:
                have[j["sid"]].add(j["key"])
            full = {s for s, ks in have.items() if all(k in ks for k in keys)}
            jobs_path.write_text("".join(json.dumps(j) + "\n" for j in rows if j["sid"] in full))
            have = collections.defaultdict(set, {s: set(keys) for s in full})
    else:
        for f in (pa_path, jobs_path):
            if f.exists():
                f.unlink()

    def natural(p, k):
        """Parse one Phase A trace. Returns (natural row, drop-test record or None)."""
        r = done[sid(p, k)]
        g = r["gen"]
        if t_close not in g:
            return {"sid": r["sid"], "id": p["id"], "capped": True, "L": None, "nat": False}, None
        start = g.index(t_open) + 1 if t_open in g else 0
        end = g.index(t_close)
        ok = TB.grade("math500", tok.decode(g[end + 1:], skip_special_tokens=True), p["gold"])
        row = {"sid": r["sid"], "id": p["id"], "capped": False, "L": end - start, "nat": ok,
               "nat_ans": len(g) - end - 1}
        rec = None
        if end - start >= a.min_L:
            rec = {"sid": r["sid"], "p": byid[p["id"]], "prompt": P[p["id"]], "gen": g, "start": start,
                   "L": end - start, "nat": ok, "nat_ans": len(g) - end - 1}
        return row, rec

    # Samples run in rounds (default 2 samples of every problem per round). Each round does Phase A and the
    # whole drop test for its traces, so a run stopped after N rounds already has complete results.
    rounds = [list(range(r, min(r + a.round, a.samples))) for r in range(0, a.samples, a.round)]
    print(f"{len(problems)} problems x {a.samples} samples in {len(rounds)} rounds; "
          f"{len(done)} traces and {len(have)} drop tests reused", flush=True)
    t_all = time.time()
    for ri, ks in enumerate(rounds, 1):
        t0 = time.time()
        # ---------------- Phase A: natural traces
        part = [(p, k) for k in ks for p in problems if sid(p, k) not in done]
        if part:
            outs = eng.generate([P[p["id"]] for p, _ in part],
                                [a.max_len - len(P[p["id"]]) - 8 for p, _ in part],
                                [a.seed + 1000 * k + pidx[p["id"]] for p, k in part])
            with pa_path.open("a") as f:
                for (p, k), (ids, why) in zip(part, outs):
                    r = {"sid": sid(p, k), "id": p["id"], "k": k, "finish": why, "gen": ids}
                    done[r["sid"]] = r
                    f.write(json.dumps(r) + "\n")
            ntok = sum(len(o[0]) for o in outs)
            print(f"round {ri}/{len(rounds)} Phase A: {len(part)} traces, {ntok} tokens, "
                  f"{ntok / (time.time() - t0):.0f} tok/s", flush=True)
        with (a.out / "natural.jsonl").open("w") as f:
            for k in range(a.samples):
                for p in problems:
                    if sid(p, k) in done:
                        f.write(json.dumps(natural(p, k)[0]) + "\n")
        todo = [rec for k in ks for p in problems for rec in [natural(p, k)[1]]
                if rec is not None and rec["sid"] not in have]
        if not todo:
            continue

        # ---------------- stage 1: warned continuations
        cont_jobs = []                                         # (rec, key, W, prompt)
        for r in todo:
            p = int(a.drop * r["L"])
            for x in a.allow:
                W = max(16, int(x * r["L"]))
                for name, text in WARNS.items():
                    cont_jobs.append((r, f"{name}_{x}", W, r["prompt"] + r["gen"][:r["start"] + p] +
                                      enc(text.format(W=W))))
        conts = eng.generate([j[3] for j in cont_jobs], [j[2] for j in cont_jobs],
                             [a.seed + 7] * len(cont_jobs), stop_ids=[t_close])

        # ---------------- stage 2: answers for every arm
        ans_jobs = collections.defaultdict(list)               # sid -> [(key, think_used, closed, prompt)]
        for r in todo:
            p = int(a.drop * r["L"])
            base = r["prompt"] + r["gen"][:r["start"] + p]
            if final:                                         # natural thinking, then the final answer only
                ans_jobs[r["sid"]].append(("natural_final", r["L"], True,
                                           r["prompt"] + r["gen"][:r["start"] + r["L"] + 1] + nl2 + fp))
            ans_jobs[r["sid"]].append(("no_think", 0, None, r["prompt"] + r["gen"][:r["start"]] + suffix + fp))
            ans_jobs[r["sid"]].append(("cut_now", p, None, base + suffix + fp))
            ans_jobs[r["sid"]].append(("cut_now_hard", p, None, base + hard + fp))
            for x in a.allow:
                W = max(16, int(x * r["L"]))
                ans_jobs[r["sid"]].append((f"silent_{x}", p + W, None,
                                           r["prompt"] + r["gen"][:r["start"] + p + W] + suffix + fp))
        for (r, key, W, cprompt), (ids, why) in zip(cont_jobs, conts):
            p = int(a.drop * r["L"])
            warn_len = len(cprompt) - len(r["prompt"]) - r["start"] - p
            if ids and ids[-1] == t_close:
                ans_jobs[r["sid"]].append((key, p + warn_len + len(ids) - 1, True, cprompt + ids + nl2 + fp))
            elif why == "stop":                                # stopped on </think> but token not returned
                ans_jobs[r["sid"]].append((key, p + warn_len + len(ids), True,
                                           cprompt + ids + [t_close] + nl2 + fp))
            else:
                ans_jobs[r["sid"]].append((key, p + warn_len + len(ids), False, cprompt + ids + suffix + fp))
        flat = [(r, *j) for r in todo for j in ans_jobs[r["sid"]]]  # arms of one trace together: prefix cache
        caps = [min(a.final_max if final else a.answer_max, a.max_len - len(j[4]) - 8) for j in flat]
        outs = eng.generate([j[4] for j in flat], caps, [a.seed + 1] * len(flat))
        rows = []
        for (r, key, used, closed, _), cap, (ids, _) in zip(flat, caps, outs):
            text = tok.decode(ids, skip_special_tokens=True)
            graded = ("\\boxed{" + text) if final else text   # the prefix already opened \boxed{
            rows.append({"sid": r["sid"], "id": r["p"]["id"], "key": key, "L": r["L"], "think_used": used,
                         "closed_itself": closed, "ans_tokens": len(ids), "ans_hit_cap": len(ids) >= cap,
                         "correct": TB.grade("math500", graded, r["p"]["gold"]),
                         "nat_correct": r["nat"], "nat_ans": r["nat_ans"], "tail": text[-200:]})
        if final:                                              # natural = natural thinking + final answer only
            nat_final = {row["sid"]: row for row in rows if row["key"] == "natural_final"}
            for row in rows:
                row["nat_full_correct"] = row["nat_correct"]
                row["nat_correct"] = nat_final[row["sid"]]["correct"]
                row["nat_ans"] = nat_final[row["sid"]]["ans_tokens"]
        with jobs_path.open("a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        for r in todo:
            have[r["sid"]] = set(keys)
        print(f"round {ri}/{len(rounds)} done in {(time.time() - t0) / 60:.0f} min "
              f"({(time.time() - t_all) / 60:.0f} min total)", flush=True)
        if ri < len(rounds):
            print("results so far:")
            analyze(a.out)
    analyze(a.out)


if __name__ == "__main__":
    sys.exit(main())
