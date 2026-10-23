#!/usr/bin/env python3
"""Multi-sample budget-drop sweep for AIME 2026 and a uniform MATH-500 subset.

The legacy budget_drop_bench.py command remains available. Add --sweep to that
command to run this experiment, for example:

    python budget_drop_bench.py --sweep --dataset all --samples 8 --mock --aime-problems 2 --math-problems 2
    python gpu/budget_drop_bench.py --sweep --dataset all --samples 8
    python gpu/budget_drop_bench.py --sweep --dataset all --samples 8 --resume

The selected question set is fixed before generation and every selected question
is retained. Traces that reach their thinking cap are marked as censored rather
than silently dropped; cut fractions for those traces use the observed capped
thinking length. Behavioral metrics are explicit text heuristics, not semantic
judgments. Full answer text and matched marker examples are saved for auditing.

Execution flow:
  1. budget_drop_bench.py sees --sweep and delegates here to main().
  2. parse_args() validates the run options; load_population() and
      uniform_sample() select each dataset's questions before any model call.
  3. main() creates one Engine (vLLM) or Mock (pipeline-only), then calls
      run_dataset() once per selected dataset.
  4. run_dataset() writes the run config, generates/resumes natural traces
      (Phase A), prepares per-trace metadata, then starts the cut sweep (Phase B).
  5. run_drop_tests() runs continuation arms, forces final answers, and saves
      complete trace batches so an interrupted run can resume.
  6. summarize() reads the saved JSONL files and writes summary.json and CSV.

Important files in each dataset output directory:
  config.json      options that define the run; checked before resume
  manifest.json    selected question IDs and sampling seed
  phase_a.jsonl    raw natural generations, used to resume Phase A
  natural.jsonl    parsed natural accuracy and behavior metrics
  jobs.jsonl       one row per sample and experimental arm
  summary.json     aggregate metrics and paired comparisons
  accuracy_vs_cut.csv  spreadsheet-friendly summary of the cut grid

The --analyze mode skips generation and rebuilds the summary/CSV from the
existing config.json, natural.jsonl, and jobs.jsonl files.

How to run (commands below start from the repository root):
    Unit tests, no model or GPU:
        python -m unittest discover -s gpu -p test_budget_drop_sweep.py

    Small end-to-end mock, no vLLM or GPU (downloads the dataset and tokenizer):
        python gpu/budget_drop_bench.py --sweep --dataset aime2026 --aime-problems 2 --samples 1 --cut-points 0.5 --cut-sizes 0.1 --mock --out /tmp/lbd-mock

    Full run on a compatible GPU:
        python gpu/budget_drop_bench.py --sweep --dataset all --samples 8
        python gpu/budget_drop_bench.py --sweep --dataset all --samples 8 --resume

    Rebuild reports without generating more tokens:
        python gpu/budget_drop_bench.py --sweep --analyze results/budget_drop_sweep/aime2026

CLI parameter guide (fractions are relative to each natural thinking length L):
    --dataset all|aime2026|math500  Choose one or both datasets; default: all.
    --aime-problems N                AIME question count; default: all 30.
    --math-problems N                MATH-500 subset size; default: 100 of 500.
    --samples N                      Independent natural traces per question; default: 8.
    --cut-points P...                Where the budget changes, e.g. 0.25 0.5 0.75.
    --cut-sizes W...                 Additional thinking window, e.g. 0.1 0.25.
                                                                     Every P + W must be at most 1.0.
    --max-think N                    Natural thinking cap; defaults to 32768 for AIME
                                                                     and 8192 for MATH-500.
    --max-len N / --answer-max N     Model context and answer caps; defaults: 40960/4096.
    --model NAME                     Hugging Face model; default: Qwen/Qwen3-4B.
    --gpu-util F / --max-num-seqs N  vLLM GPU memory fraction/concurrency; defaults: 0.90/4.
    --batch-size N                   Generation prompts per vLLM batch; default: 16.
    --trace-batch-size N             Sample traces checkpointed together in Phase B; default: 4.
    --seed N                         Reproducible question selection and generation seed; default: 0.
    --out DIR                        Parent output directory; dataset subdirectories are added.
    --resume                         Reuse saved traces/results; config and problem IDs must match.
    --hard-cut                       Use a bare </think> for the immediate-cut arm.
    --mock                           Use synthetic generation; for pipeline checks only.
    --analyze DIR                    Rebuild reports from one dataset output directory.

The mock command needs Python packages datasets and transformers plus access to the
dataset and tokenizer files. A real run additionally needs vLLM and a compatible
CUDA GPU. The --analyze command needs only the saved result files and this code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import thinking_budget_bench as TB

from budget_drop_bench import HARD_CLOSE, WARNS

WARNING_NAMES = {"warn_soft": "soft", "warn_count": "countdown"}

# These markers measure explicit textual signals. They can miss implicit
# reasoning and can produce false positives, so evidence is retained in output.
LEAKAGE_RE = re.compile(
    r"\b(?:let me (?:think|work this out)|let's (?:think|work this out)|"
    r"we need to (?:calculate|compute|solve)|first,?\s+(?:we|I)\b|"
    r"next,?\s+(?:we|I)\b|double[- ]check(?:ing)?)\b",
    re.IGNORECASE,
)
RECHECK_RE = re.compile(
    r"\b(?:double[- ]check(?:ing)?|let's check|let me verify|"
    r"verify (?:the|our|my) (?:work|answer|result|calculation)|"
    r"check(?:ing)? (?:the|our|my) (?:work|answer|result|calculation)|"
    r"recalculat(?:e|ing)|substitut(?:e|ing) .{0,60} back)\b",
    re.IGNORECASE,
)
APPROACH_SWITCH_RE = re.compile(
    r"\b(?:another (?:approach|method|way)|alternative (?:approach|method)|"
    r"different (?:approach|method|way)|instead,|"
    r"let's try (?:another|a different)|switch(?:ing)? to)\b",
    re.IGNORECASE,
)


# 将浮点比例格式化成稳定的短字符串，用于实验组名称和结果字段。
def float_key(value: float) -> str:
    """Format a fraction for stable, readable names in arm keys and output rows."""
    return f"{value:.4f}".rstrip("0").rstrip(".")


# 从完整题库里随机抽 count 道、不重复；用 seed 保证以后可以抽到同一批题；如果要抽的数量超过题库容量就报错
def uniform_sample(population: list[dict], count: int, seed: int) -> list[dict]:
    """Choose a reproducible uniform subset without replacement.

    Sampling occurs before any generations are inspected, so success, trace
    length, and answer correctness cannot affect which questions are selected.

    Args:
        population: Complete list of normalized question records.
        count: Number of distinct questions to select.
        seed: Random seed used only for this subset selection.
    Returns:
        The selected question records, in reproducible random order.
    """
    if count < 1:
        raise ValueError("problem count must be positive")
    if count > len(population):
        raise ValueError(f"requested {count} problems, but the dataset has only {len(population)}")
    return random.Random(seed).sample(population, count)


# 按 cut point 和 cut size 组合arm, 生成每个实验组的配置，用for 循环遍历每个点和窗口大小
def make_arm_specs(cut_points: list[float], cut_sizes: list[float]) -> list[dict]:
    """Build one immediate-cut arm and four comparison arms per grid point.

    ``cut_point`` and ``cut_size`` are fractions of a trace's natural thinking
    length L. Window arms share the same point and size so their outcomes can
    be compared directly.

    Args:
        cut_points: Fractions of L at which the budget changes.
        cut_sizes: Fractions of L allowed after the change.
    Returns:
        Arm dictionaries containing a unique key, arm name, and grid values.
    """
    if not cut_points or not cut_sizes:
        raise ValueError("at least one cut point and cut size are required")
    if len(set(cut_points)) != len(cut_points) or len(set(cut_sizes)) != len(cut_sizes):
        raise ValueError("cut points and cut sizes must not contain duplicates")
    for point in cut_points:
        if not 0 < point < 1:
            raise ValueError("each cut point must be strictly between 0 and 1")
    for size in cut_sizes:
        if size <= 0:
            raise ValueError("each cut size must be positive")
        if any(point + size > 1.000001 for point in cut_points):
            raise ValueError("cut point + cut size must not exceed 1.0")


    # 在这里会生成每个 cut point 和 cut size arm，包括silent, resample、warn_soft、warn_count 和 cut_now
    # 按理说如果cut point count = a, cut size count = b，那么总共会生成 a * (1 + 4 * b) 个
    specs = []
    for point in cut_points:
        p_key = float_key(point)
        # A cut_now arm has no grace window, so it is defined once per cut point.
        specs.append({"key": f"cut_now|p={p_key}", "arm": "cut_now",
                      "cut_point": point, "cut_size": None})
        for size in cut_sizes:
            suffix = f"p={p_key}|w={float_key(size)}"
            # These four arms share the same cut point and post-cut allowance.
            for arm in ("silent", "resample", "warn_soft", "warn_count"):
                specs.append({"key": f"{arm}|{suffix}", "arm": arm,
                              "cut_point": point, "cut_size": size})
    return specs


# 在推理文本和最终答案中查找泄漏、复查、方法切换标记，并保留命中短语。
def behavior_metrics(reasoning: str, answer: str) -> dict:
    """Measure text-marker proxies and retain matching phrases as evidence.

    Leakage is searched in the answer after the thinking boundary. Re-checks
    and approach switches are searched in the thinking text. These are
    transparent lexical counts, not semantic labels; false positives and
    missed implicit behavior are possible.

    Args:
        reasoning: Text generated before the final-answer boundary.
        answer: Text generated after the final-answer boundary.
    Returns:
        Boolean leakage flag, marker counts, and matched evidence strings.
    """
    leakage = [match.group(0) for match in LEAKAGE_RE.finditer(answer)]
    rechecks = [match.group(0) for match in RECHECK_RE.finditer(reasoning)]
    switches = [match.group(0) for match in APPROACH_SWITCH_RE.finditer(reasoning)]
    return {
        "reasoning_leakage": bool(leakage),
        "recheck_count": len(rechecks),
        "approach_switch_count": len(switches),
        "behavior_evidence": {"leakage": leakage[:5], "rechecks": rechecks[:10],
                              "approach_switches": switches[:10]},
    }


# 将运行 seed 与样本/实验组标识组合成跨进程稳定的随机种子。
def stable_seed(seed: int, identity: str) -> int:
    """Derive a stable 32-bit seed from a run seed and a sample/arm identity.

    Python's built-in hash is randomized between processes. SHA-256 keeps a
    resumed run's generation seed stable across separate program invocations.

    Args:
        seed: User-selected base seed.
        identity: Stable sample or sample/arm identifier.
    Returns:
        A deterministic integer suitable for a generation seed.
    """
    digest = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


# 逐行读取 JSONL；遇到中断产生的不完整行时跳过，保留之前的结果。
def read_jsonl(path: Path) -> list[dict]:
    """Read JSON objects from a JSONL file, tolerating a truncated final row.

    Ignoring malformed lines allows recovery when a process was interrupted
    while appending a checkpoint. Valid earlier rows remain usable.
    """
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Ignore a final partial line left by an interrupted GPU run.
                continue
    return rows


# 将一批记录追加到 JSONL 并立即 flush，供长任务中断后恢复。
def append_jsonl(path: Path, rows: list[dict]) -> None:
    """Append records as UTF-8 JSONL and flush them as a resumable checkpoint."""
    if not rows:
        return
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


class Engine:
    """Batched vLLM adapter; all prompts are already tokenized.

    This is the real inference backend. Importing vLLM is deferred until this
    class is constructed, so helper tests and --mock do not require vLLM.
    """

    # 初始化真实 vLLM 推理后端和 tokenizer，并配置显存、上下文与并发数。
    def __init__(self, model: str, max_len: int, gpu_util: float, max_num_seqs: int,
                 batch_size: int, seed: int):
        """Load the model, tokenizer, and vLLM settings for repeated batched calls."""
        import os
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM
        from transformers import AutoTokenizer

        self.llm = LLM(model=model, enable_prefix_caching=True,
                       gpu_memory_utilization=gpu_util, max_model_len=max_len,
                       max_num_seqs=max_num_seqs, seed=seed)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.batch_size = batch_size

    # 批量生成模型输出；每条 prompt 使用对应的 token 上限、seed 和可选停止标记。
    def generate(self, prompts: list[list[int]], max_tokens: list[int], seeds: list[int],
                 stop_ids: list[int] | None = None) -> list[tuple[list[int], str]]:
        """Generate one result per prompt, honoring its own token cap and seed.

        Large task lists are split into ``batch_size`` chunks to limit memory
        pressure. ``stop_ids`` is used during thinking continuations so the
        model can close its own ``</think>`` section.

        Args:
            prompts: Tokenized model inputs, one list of token IDs per request.
            max_tokens: Maximum generated tokens for each corresponding prompt.
            seeds: Random seed for each corresponding request.
            stop_ids: Optional token IDs that end a continuation early.
        Returns:
            Pairs of generated token IDs and vLLM finish reason, in input order.
        """
        from vllm import SamplingParams

        results = []
        for offset in range(0, len(prompts), self.batch_size):
            end = offset + self.batch_size
            chunk_prompts = prompts[offset:end]
            # Each prompt gets its own cap and seed so different arms stay reproducible.
            params = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20,
                                     max_tokens=max(1, int(limit)), seed=int(item_seed),
                                     stop_token_ids=stop_ids)
                      for limit, item_seed in zip(max_tokens[offset:end], seeds[offset:end])]
            outputs = self.llm.generate(
                [{"prompt_token_ids": prompt} for prompt in chunk_prompts], params)
            results.extend((list(output.outputs[0].token_ids), output.outputs[0].finish_reason)
                           for output in outputs)
        return results


class Mock:
    """Deterministic fake model for pipeline checks; it does not measure quality.

    The real tokenizer is retained so prompt formatting and token accounting
    follow the production path, but generated reasoning and answers are fake.
    """

    # 初始化 mock 后端；保留真实 tokenizer，但不加载模型权重或调用 GPU。
    def __init__(self, model: str, problems: list[dict], batch_size: int):
        """Load only the tokenizer and remember questions for synthetic answers."""
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model)
        self.problems = problems
        self.batch_size = batch_size
        self.t_open = self.tok.convert_tokens_to_ids("<think>")
        self.t_close = self.tok.convert_tokens_to_ids("</think>")
        self.filler = self.tok.encode(" step", add_special_tokens=False)[0]

    # 按 Engine 的接口返回可复现的假推理或假答案，只用于检查流程，不代表模型效果。
    def generate(self, prompts: list[list[int]], max_tokens: list[int], seeds: list[int],
                 stop_ids: list[int] | None = None) -> list[tuple[list[int], str]]:
        """Return deterministic synthetic traces or answers matching Engine's API."""
        outputs = []
        for prompt, limit, seed in zip(prompts, max_tokens, seeds):
            rng = random.Random(seed)
            limit = max(1, int(limit))
            if stop_ids:
                count = rng.randint(1, limit)
                if count < limit and rng.random() < 0.6:
                    outputs.append(([self.filler] * count + [self.t_close], "stop"))
                else:
                    outputs.append(([self.filler] * limit, "length"))
                continue

            text = self.tok.decode(prompt, skip_special_tokens=False)
            problem = next((item for item in self.problems if item["q"] in text), None)
            gold = problem["gold"] if problem else "0"
            if self.t_open not in prompt:
                count = min(limit - 1, rng.randint(200, 900))
                answer = f"\n\nSo the answer is \\boxed{{{gold}}}."
                generated = ([self.t_open] + [self.filler] * count + [self.t_close] +
                             self.tok.encode(answer, add_special_tokens=False))
                outputs.append((generated[:limit], "stop" if len(generated) <= limit else "length"))
            else:
                answer = f"So the answer is \\boxed{{{gold}}}."
                outputs.append((self.tok.encode(answer, add_special_tokens=False), "stop"))
        return outputs


# 加载完整 AIME 或 MATH-500 题库；随机抽取子集由 main() 之后统一完成。
def load_population(dataset: str, aime_dataset: str) -> list[dict]:
    """Load the full AIME 2026 or MATH-500 question population.

    This function does not select the experiment subset; ``main()`` performs
    that separate, seeded selection after the complete population is loaded.

    Args:
        dataset: Internal name, either ``aime2026`` or ``math500``.
        aime_dataset: Hugging Face dataset ID used when loading AIME.
    Returns:
        Question records with ``id``, ``q``, and ``gold`` fields.
    """
    if dataset == "math500":
        # Load the full population first; selection is performed separately and uniformly.
        return TB.load_problems("math500", 500, 0)
    from datasets import load_dataset

    rows = load_dataset(aime_dataset, split="train")
    tag = aime_dataset.split("/")[-1]
    return [{"id": f"{tag}/{row['problem_idx']}", "q": row["problem"],
             "gold": str(row["answer"])} for row in rows]


# 将题目套入 Qwen 对话模板并编码为模型输入 token。
def make_prompt(tokenizer, question: str) -> list[int]:
    """Build the shared Qwen chat prompt and encode it without extra tokens."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question + TB.INSTR}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    return tokenizer.encode(text, add_special_tokens=False)


# 从 Phase A 输出中定位 think 区间，计算可用长度 L，并标记是否触及上限。
def trace_fields(generation: list[int], t_open: int, t_close: int, max_think: int) -> dict:
    """Locate the thinking span and return its usable length and censor status.

    A generation without ``</think>`` is right-censored: its true natural
    endpoint is unknown, so its observed length (up to ``max_think``) is used
    rather than dropping the selected problem.

    Args:
        generation: Raw model output token IDs from Phase A.
        t_open: Token ID for the opening ``<think>`` marker.
        t_close: Token ID for the closing ``</think>`` marker.
        max_think: Maximum reasoning length allowed by the run configuration.
    Returns:
        Start/end token positions, usable length ``L``, and censor flag.
    """
    # A capped trace has no known natural endpoint; retain it and use its observed cap as L.
    start = generation.index(t_open) + 1 if t_open in generation else 0
    end = generation.index(t_close) if t_close in generation else None
    observed = (end if end is not None else len(generation)) - start
    length = min(max(0, observed), max_think)
    return {"start": start, "end": end, "L": length,
            "capped": end is None or observed > max_think}


# 从续写 token 中去掉 think 边界标记，便于拼接文本和统计行为指标。
def clean_thinking_ids(ids: list[int], t_open: int, t_close: int) -> list[int]:
    """Remove boundary markers from a generated continuation for text metrics."""
    cleaned = list(ids)
    if cleaned and cleaned[0] == t_open:
        cleaned = cleaned[1:]
    if cleaned and cleaned[-1] == t_close:
        cleaned = cleaned[:-1]
    return cleaned


# 按题目为单位做 bootstrap，让同一道题的多次采样始终作为一个整体重抽。
def bootstrap_cluster_mean(rows: list[dict], field: str, seed: int,
                           iterations: int = 2000) -> tuple[float, float, float]:
    """Return a mean and percentile interval from a problem-cluster bootstrap.

    All samples belonging to a selected problem move together in each
    bootstrap draw, avoiding the false independence assumption that would
    result from resampling the eight traces separately.

    Args:
        rows: Records with a problem ``id`` and the numeric field to summarize.
        field: Name of the numeric value to average in each bootstrap draw.
        seed: Seed for reproducible bootstrap draws.
        iterations: Number of bootstrap replicates.
    Returns:
        Point estimate, lower 2.5th percentile, and upper 97.5th percentile.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["id"]].append(float(row[field]))
    problem_ids = list(grouped)
    if not problem_ids:
        return 0.0, 0.0, 0.0
    values = [value for group in grouped.values() for value in group]
    point = statistics.mean(values)
    rng = random.Random(seed)
    estimates = []
    # Resample problem IDs, not individual traces, because 8 samples of one problem are correlated.
    for _ in range(iterations):
        draw = [problem_ids[rng.randrange(len(problem_ids))] for _ in problem_ids]
        sampled = [value for problem_id in draw for value in grouped[problem_id]]
        estimates.append(statistics.mean(sampled))
    estimates.sort()
    return point, estimates[int(0.025 * iterations)], estimates[min(iterations - 1, int(0.975 * iterations))]


# 创建运行配置和题目清单；续跑时先确认配置及抽样题目与旧结果一致。
def ensure_run_files(out_dir: Path, config: dict, manifest: dict, resume: bool) -> None:
    """Create run metadata or verify it matches before reusing prior outputs.

    Refusing mismatched configuration or question IDs prevents accidentally
    combining partial results from different grids or random samples.

    Args:
        out_dir: Directory containing this dataset's run artifacts.
        config: Experiment options that must match when resuming.
        manifest: Selected problem IDs and selection metadata.
        resume: Whether existing metadata should be validated and reused.
    """
    config_path = out_dir / "config.json"
    manifest_path = out_dir / "manifest.json"
    if out_dir.exists() and any(out_dir.iterdir()) and not resume:
        raise ValueError(f"{out_dir} is not empty; pass --resume or choose another --out directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    if resume and config_path.exists():
        # Never mix runs with different grids, seeds, or selected question sets.
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(f"resume config differs from existing run in {out_dir}")
        if manifest_path.exists():
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing_manifest != manifest:
                raise ValueError(f"resume problem manifest differs from existing run in {out_dir}")
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# 把原始 Phase A 轨迹解析成实验记录，供同一条轨迹的所有 cut 实验配对使用。
def natural_records(problems: list[dict], samples: int, prompts: dict[str, list[int]],
                    phase_rows: dict[str, dict], tokenizer, t_open: int, t_close: int,
                    max_think: int, dataset: str) -> list[dict]:
    """Parse saved Phase A generations into records used by all cut arms.

    Each ``sid`` identifies one question/sample pair. Its prompt, natural
    token sequence, length, natural score, and baseline behavior metrics are
    reused to keep every comparison paired to the same trace.

    Args:
        problems: Selected question records.
        samples: Number of natural traces expected per question.
        prompts: Tokenized prompt indexed by question ID.
        phase_rows: Saved raw Phase A rows indexed by sample ID.
        tokenizer: Tokenizer used to decode thinking and answer text.
        t_open/t_close: Token IDs delimiting the model's thinking section.
        max_think: Maximum usable thinking length.
        dataset: Dataset label retained for the caller's run context.
    Returns:
        One parsed record per question/sample pair.
    """
    records = []
    for sample in range(samples):
        for problem in problems:
            # The sample ID ties every experimental arm back to the same natural trace.
            sid = f"{problem['id']}#sample={sample}"
            phase = phase_rows[sid]
            generation = phase["gen"]
            fields = trace_fields(generation, t_open, t_close, max_think)
            if fields["end"] is None:
                answer_text = ""
                thinking_ids = generation[fields["start"]:fields["start"] + fields["L"]]
            else:
                answer_text = tokenizer.decode(generation[fields["end"] + 1:], skip_special_tokens=True)
                thinking_ids = generation[fields["start"]:fields["start"] + fields["L"]]
            thinking_text = tokenizer.decode(thinking_ids, skip_special_tokens=True)
            metrics = behavior_metrics(thinking_text, answer_text)
            records.append({"sid": sid, "id": problem["id"], "sample": sample,
                            "p": problem, "prompt": prompts[problem["id"]],
                            "gen": generation, **fields,
                            "natural_correct": TB.grade("math500", answer_text, problem["gold"]),
                            "natural_answer": answer_text, "natural_metrics": metrics})
    return records


# 将相对 L 的 cut point/size 换算成整数 token 数，并限制在轨迹长度范围内。
def add_cut_and_size(record: dict, point: float, size: float | None) -> tuple[int, int]:
    """Convert fractional cut settings into integer prefix and grace lengths.

    Both values are bounded by the trace's usable natural length L. The return
    value is ``(tokens_before_cut, extra_tokens_after_cut)``.

    Args:
        record: Parsed natural trace containing ``L``.
        point: Fraction of L retained before the budget change.
        size: Fraction of L allowed after the change, or None for immediate cut.
    Returns:
        Integer pair ``(cut_tokens, extra_tokens)``.
    """
    # Both fractions are relative to this trace's usable natural thinking length L.
    cut = min(record["L"], max(0, int(round(point * record["L"]))))
    extra = 0 if size is None else max(1, int(round(size * record["L"])))
    extra = min(extra, max(0, record["L"] - cut))
    return cut, extra


# 根据已占用上下文和 answer_max 计算最终答案最多能生成多少 token。
def answer_cap(prompt_ids: list[int], max_len: int, answer_max: int) -> int:
    """Choose an answer-generation cap that fits inside the model context.

    Eight tokens are reserved for chat/model boundary overhead; the answer is
    then limited by whichever is smaller: remaining context or ``answer_max``.

    Args:
        prompt_ids: Tokenized context already assembled for the answer call.
        max_len: Model's total context-window limit.
        answer_max: Configured upper bound for generated answer tokens.
    Returns:
        Positive number of tokens available for answer generation.
    """
    available = max_len - len(prompt_ids) - 8
    if available < 1:
        raise ValueError("constructed answer prompt exceeds the configured context window")
    return min(answer_max, available)


# 执行 Phase B：生成续写与答案、评分并按批次保存各实验组结果。
def run_drop_tests(records: list[dict], specs: list[dict], engine, tokenizer,
                   t_open: int, t_close: int, enc, args, out_dir: Path) -> None:
    """Run Phase B for unfinished sample/arm pairs and checkpoint their results.

    For each trace batch the function first generates continuations needed by
    warning and resample arms. It then constructs prompts for every unfinished
    arm, generates final answers, scores them, calculates behavior markers,
    and appends one row per arm to ``jobs.jsonl``. Existing rows are skipped
    on resume.

    Args:
        records: Parsed Phase A traces for the current dataset.
        specs: Cut-point/cut-size arm definitions from make_arm_specs().
        engine: Engine or Mock implementing the batched generate() interface.
        tokenizer: Model tokenizer used to encode prompts and decode outputs.
        t_open/t_close: Thinking boundary token IDs.
        enc: Callable that converts a text suffix to token IDs.
        args: Parsed CLI options, including seeds and batch/context caps.
        out_dir: Dataset output directory containing the resumable jobs file.
    """
    jobs_path = out_dir / "jobs.jsonl"
    existing = read_jsonl(jobs_path)
    completed = defaultdict(dict)
    for row in existing:
        completed[row["sid"]][row["key"]] = row

    warning_suffix = enc(TB.EARLY_STOP)
    hard_suffix = enc(HARD_CLOSE)
    new_records = [record for record in records
                   if any(spec["key"] not in completed[record["sid"]] for spec in specs)]
    for offset in range(0, len(new_records), args.trace_batch_size):
        batch_records = new_records[offset:offset + args.trace_batch_size]
        continuation_tasks = []

        # Warnings resume the saved prefix; resample starts from the original question with the same total cap.
        for record in batch_records:
            for spec in specs:
                if spec["key"] in completed[record["sid"]]:
                    continue
                if spec["arm"] not in ("resample", "warn_soft", "warn_count"):
                    continue
                cut, extra = add_cut_and_size(record, spec["cut_point"], spec["cut_size"])
                if spec["arm"] == "resample":
                    cap = max(1, cut + extra)
                    prompt = record["prompt"]
                    reasoning_prefix = []
                else:
                    cap = max(1, extra)
                    prefix = record["gen"][:record["start"] + cut]
                    notice = WARNS[spec["arm"]]
                    prompt = record["prompt"] + prefix + enc(notice.format(W=extra))
                    reasoning_prefix = record["gen"][record["start"]:record["start"] + cut]
                continuation_tasks.append({"record": record, "spec": spec, "prompt": prompt,
                                           "cap": cap, "reasoning_prefix": reasoning_prefix,
                                           "seed": stable_seed(args.seed, record["sid"] + spec["key"])})

        continuation_outputs = engine.generate(
            [task["prompt"] for task in continuation_tasks],
            [task["cap"] for task in continuation_tasks],
            [task["seed"] for task in continuation_tasks], stop_ids=[t_close])
        continuations = {}
        for task, (ids, finish) in zip(continuation_tasks, continuation_outputs):
            continuations[(task["record"]["sid"], task["spec"]["key"])] = {
                "ids": ids, "finish": finish, "task": task}

        answer_tasks = []
        for record in batch_records:
            for spec in specs:
                if spec["key"] in completed[record["sid"]]:
                    continue
                arm = spec["arm"]
                cut, extra = add_cut_and_size(record, spec["cut_point"], spec["cut_size"])
                closed_itself = None
                if arm == "cut_now":
                    # Stop at p and ask for an answer immediately.
                    think_ids = record["gen"][record["start"]:record["start"] + cut]
                    model_prompt = record["prompt"] + record["gen"][:record["start"] + cut] + warning_suffix
                    think_used = len(think_ids)
                elif arm == "silent":
                    # Keep the original reasoning through p + W without announcing the cut.
                    kept = min(record["L"], cut + extra)
                    think_ids = record["gen"][record["start"]:record["start"] + kept]
                    model_prompt = record["prompt"] + record["gen"][:record["start"] + kept] + warning_suffix
                    think_used = len(think_ids)
                else:
                    # Reuse the continuation generated above, then honor its own close or force one.
                    continuation = continuations[(record["sid"], spec["key"])]
                    task = continuation["task"]
                    ids = continuation["ids"]
                    closed_itself = bool(ids and ids[-1] == t_close) or continuation["finish"] == "stop"
                    continuation_thinking = clean_thinking_ids(ids, t_open, t_close)
                    think_ids = task["reasoning_prefix"] + continuation_thinking
                    if arm == "resample":
                        generated = ids if t_open in ids else [t_open] + ids
                        base = record["prompt"] + generated
                    else:
                        base = task["prompt"] + ids
                    if closed_itself:
                        if not (ids and ids[-1] == t_close):
                            base += [t_close]
                        model_prompt = base + enc("\n\n")
                    else:
                        model_prompt = base + warning_suffix
                    think_used = len(think_ids)

                if arm == "cut_now" and args.hard_cut:
                    model_prompt = (record["prompt"] + record["gen"][:record["start"] + cut] +
                                    hard_suffix)
                thinking_text = tokenizer.decode(think_ids, skip_special_tokens=True)
                cap = answer_cap(model_prompt, args.max_len, args.answer_max)
                answer_tasks.append({"record": record, "spec": spec, "prompt": model_prompt,
                                     "cap": cap, "think_ids": think_ids,
                                     "think_used": think_used, "closed_itself": closed_itself,
                                     "thinking_text": thinking_text,
                                     "seed": stable_seed(args.seed + 1, record["sid"] + spec["key"])})

        answer_outputs = engine.generate(
            [task["prompt"] for task in answer_tasks],
            [task["cap"] for task in answer_tasks],
            [task["seed"] for task in answer_tasks])
        rows = []
        for task, (ids, _) in zip(answer_tasks, answer_outputs):
            record, spec = task["record"], task["spec"]
            answer_text = tokenizer.decode(ids, skip_special_tokens=True)
            metrics = behavior_metrics(task["thinking_text"], answer_text)
            rows.append({"sid": record["sid"], "id": record["id"], "sample": record["sample"],
                         "key": spec["key"], "arm": spec["arm"],
                         "cut_point": spec["cut_point"], "cut_size": spec["cut_size"],
                         "L": record["L"], "natural_capped": record["capped"],
                         "think_used": task["think_used"], "answer_tokens": len(ids),
                         "closed_itself": task["closed_itself"],
                         "correct": TB.grade("math500", answer_text, record["p"]["gold"]),
                         **metrics, "answer_text": answer_text})
                # Save complete trace batches so --resume only repeats unfinished traces.
        append_jsonl(jobs_path, rows)
        for row in rows:
            completed[row["sid"]][row["key"]] = row
        print(f"drop sweep checkpoint: {offset + len(batch_records)}/{len(new_records)} traces",
              flush=True)


# 从保存的自然轨迹和实验结果计算准确率、行为指标、置信区间并输出 JSON/CSV。
def summarize(out_dir: Path) -> dict:
    """Aggregate saved trace rows into accuracy, behavior, and paired metrics.

    Accuracy intervals and paired silent-versus-control differences use the
    problem-cluster bootstrap. This function also writes the JSON and CSV
    reports, and can be called independently with ``--analyze``.

    Args:
        out_dir: One dataset's output directory with config and JSONL artifacts.
    Returns:
        The summary object written to summary.json.
    """
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    natural = read_jsonl(out_dir / "natural.jsonl")
    jobs = read_jsonl(out_dir / "jobs.jsonl")
    specs = make_arm_specs(config["cut_points"], config["cut_sizes"])
    by_key = defaultdict(list)
    for row in jobs:
        by_key[row["key"]].append(row)

    # 汇总单个实验组的准确率区间、行为指标和平均 token 用量。
    def stats(rows: list[dict], identity: str) -> dict:
        """Summarize one experimental arm, including its clustered accuracy CI."""
        if not rows:
            return {"n_samples": 0, "n_problems": 0}
        accuracy, low, high = bootstrap_cluster_mean(rows, "correct", stable_seed(11, identity))
        return {
            "n_samples": len(rows), "n_problems": len({row["id"] for row in rows}),
            "accuracy_pct": round(100 * accuracy, 2),
            "accuracy_ci95_pct": [round(100 * low, 2), round(100 * high, 2)],
            "reasoning_leakage_pct": round(100 * statistics.mean(
                bool(row["reasoning_leakage"]) for row in rows), 2),
            "mean_rechecks": round(statistics.mean(row["recheck_count"] for row in rows), 3),
            "traces_with_recheck_pct": round(100 * statistics.mean(
                row["recheck_count"] > 0 for row in rows), 2),
            "mean_approach_switches": round(statistics.mean(
                row["approach_switch_count"] for row in rows), 3),
            "traces_with_approach_switch_pct": round(100 * statistics.mean(
                row["approach_switch_count"] > 0 for row in rows), 2),
            "mean_think_tokens": round(statistics.mean(row["think_used"] for row in rows), 1),
            "mean_answer_tokens": round(statistics.mean(row["answer_tokens"] for row in rows), 1),
            "capped_trace_pct": round(100 * statistics.mean(row["natural_capped"] for row in rows), 2),
        }

    accuracy_rows = []
    for spec in specs:
        row = {"arm": spec["arm"], "cut_point": spec["cut_point"],
               "cut_size": spec["cut_size"], "key": spec["key"]}
        row.update(stats(by_key[spec["key"]], spec["key"]))
        accuracy_rows.append(row)

    natural_correct = [row for row in natural]
    natural_accuracy = (100 * statistics.mean(row["correct"] for row in natural_correct)
                        if natural_correct else 0.0)
    natural_metrics = {
        "n_samples": len(natural),
        "n_problems": len({row["id"] for row in natural}),
        "reasoning_leakage_pct": round(100 * statistics.mean(
            bool(row.get("reasoning_leakage", False)) for row in natural), 2) if natural else 0.0,
        "mean_rechecks": round(statistics.mean(
            row.get("recheck_count", 0) for row in natural), 3) if natural else 0.0,
        "traces_with_recheck_pct": round(100 * statistics.mean(
            row.get("recheck_count", 0) > 0 for row in natural), 2) if natural else 0.0,
        "mean_approach_switches": round(statistics.mean(
            row.get("approach_switch_count", 0) for row in natural), 3) if natural else 0.0,
        "traces_with_approach_switch_pct": round(100 * statistics.mean(
            row.get("approach_switch_count", 0) > 0 for row in natural), 2) if natural else 0.0,
    }
    paired = []
    for spec in specs:
        if spec["arm"] != "silent":
            continue
        p, w = spec["cut_point"], spec["cut_size"]
        for other_arm in ("warn_soft", "warn_count", "resample"):
            other_key = next((item["key"] for item in specs
                              if item["arm"] == other_arm and item["cut_point"] == p
                              and item["cut_size"] == w), None)
            if not other_key:
                continue
            # Paired differences use only sample IDs present in both arms.
            silent_by_sid = {row["sid"]: row for row in by_key[spec["key"]]}
            other_by_sid = {row["sid"]: row for row in by_key[other_key]}
            diffs = [{"id": silent_by_sid[sid]["id"], "difference":
                      int(silent_by_sid[sid]["correct"]) - int(other_by_sid[sid]["correct"])}
                     for sid in silent_by_sid.keys() & other_by_sid.keys()]
            if diffs:
                point, low, high = bootstrap_cluster_mean(
                    diffs, "difference", stable_seed(13, spec["key"] + other_key))
                paired.append({"cut_point": p, "cut_size": w,
                               "comparison": f"silent - {other_arm}",
                               "n_samples": len(diffs), "difference_pp": round(100 * point, 2),
                               "ci95_pp": [round(100 * low, 2), round(100 * high, 2)]})

    summary = {"dataset": config["dataset"], "model": config["model"],
               "selected_problems": config["selected_problem_count"],
               "samples_per_problem": config["samples"],
               "selected_traces": len(natural), "natural_accuracy_pct": round(natural_accuracy, 2),
               "natural_behavior_metrics": natural_metrics,
               "behavior_metrics_note": config["behavior_metrics_note"],
               "resample_control_note": config["resample_control_note"],
               "accuracy_vs_cut": accuracy_rows, "paired_accuracy_differences": paired}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = out_dir / "accuracy_vs_cut.csv"
    columns = ["arm", "cut_point", "cut_size", "n_samples", "n_problems", "accuracy_pct",
               "accuracy_ci95_pct", "reasoning_leakage_pct", "mean_rechecks",
               "traces_with_recheck_pct", "mean_approach_switches",
               "traces_with_approach_switch_pct", "mean_think_tokens", "mean_answer_tokens",
               "capped_trace_pct"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in accuracy_rows:
            writer.writerow({key: row.get(key) for key in columns})
    return summary


# 运行一个数据集的 Phase A 与 Phase B；负责恢复轨迹、保存基线并生成汇总。
def run_dataset(dataset: str, problems: list[dict], engine, args, tokenizer,
                max_think: int, out_dir: Path) -> None:
    """Run or resume both phases for one already-selected dataset subset.

    Phase A generates missing natural traces and writes them immediately.
    The function parses all traces (including capped ones), saves baseline
    metrics, runs Phase B, then produces the final summary and CSV.

    Args:
        dataset: Output label, either ``aime2026`` or ``math500``.
        problems: Uniformly selected question records for this dataset.
        engine: Engine or Mock shared by the selected datasets.
        args: Validated CLI options.
        tokenizer: Tokenizer belonging to ``engine``.
        max_think: Natural reasoning cap for this dataset.
        out_dir: Dataset-specific output directory.
    """
    ## prepare arm specifications and prompts for each problem, 预备每个实验组的配置
    # 分别是make_arm_specs 和 make_prompt 的调用，生成实验组配置和每个问题的提示文本
    # 然后是ensure_run_files，确保输出目录和配置文件存在
    specs = make_arm_specs(args.cut_points, args.cut_sizes)
    prompts = {problem["id"]: make_prompt(tokenizer, problem["q"]) for problem in problems}
    manifest = {"dataset": dataset, "seed": args.seed,
                "selection": "uniform without replacement, selected before generation",
                "selected_problem_ids": [problem["id"] for problem in problems],
                "samples_per_problem": args.samples}
    config = {"dataset": dataset, "model": args.model, "samples": args.samples,
              "selected_problem_count": len(problems), "seed": args.seed,
              "cut_points": args.cut_points, "cut_sizes": args.cut_sizes,
              "max_think": max_think, "max_len": args.max_len,
              "answer_max": args.answer_max, "mock": args.mock,
              "hard_cut": args.hard_cut,
              "resample_control_note": "Fresh generation from the original question with total thinking cap cut_point*L + cut_size*L; compared with retaining the natural prefix.",
              "behavior_metrics_note": "Lexical heuristics only: leakage markers in answer text; re-check and approach-switch markers in reasoning text. Inspect evidence and answer_text before interpreting as semantic labels."}
    ensure_run_files(out_dir, config, manifest, args.resume)

    # --------------------------------------------------------------------------------------------------
    # Phase A makes the configured number of independent traces; no accuracy-based filtering follows.
    # --------------------------------------------------------------------------------------------------
    # 这里跑的是phase A, 生成每个问题的自然思考轨迹，不做准确性过滤
    phase_path = out_dir / "phase_a.jsonl"
    phase_rows = {row["sid"]: row for row in read_jsonl(phase_path)}
    # 这里的逻辑在 phase A, phase B 都是相同的：先检查文件里已有的样本，缺失的才调用模型生成。
    # 文件里已有的样本：直接复用，不调用模型，也不会再次 append。
    # 文件里缺失的样本：才会调用 generate()；成功后追加到文件。
    # TODO: run on better GPU?? 因为不同 GPU 架构、CUDA/cuDNN、vLLM 版本和计算精度可能不一样，所以相同的随机种子可能生成略有不同的结果？
    pending = [(problem, sample) for sample in range(args.samples) for problem in problems
               if f"{problem['id']}#sample={sample}" not in phase_rows]
    t_open = tokenizer.convert_tokens_to_ids("<think>")
    t_close = tokenizer.convert_tokens_to_ids("</think>")
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset + args.batch_size]
        batch_prompts = [prompts[problem["id"]] for problem, _ in batch]
        caps = [max(1, min(max_think + args.answer_max,
                           args.max_len - len(prompt) - 8)) for prompt in batch_prompts]
        seeds = [stable_seed(args.seed, f"{problem['id']}#sample={sample}")
                 for problem, sample in batch]
        outputs = engine.generate(batch_prompts, caps, seeds)
        new_rows = []
        for (problem, sample), (ids, finish) in zip(batch, outputs):
            row = {"sid": f"{problem['id']}#sample={sample}", "id": problem["id"],
                   "sample": sample, "finish": finish, "gen": ids}
            phase_rows[row["sid"]] = row
            new_rows.append(row)
        append_jsonl(phase_path, new_rows)
        print(f"{dataset} natural traces: {min(offset + len(batch), len(pending))}/{len(pending)} new",
              flush=True)
        
    # --------------------------------------------------------------------------------------------------
    # natural_records: 解析轨迹和基线指标
    # --------------------------------------------------------------------------------------------------
    records = natural_records(problems, args.samples, prompts, phase_rows, tokenizer,
                              t_open, t_close, max_think, dataset)
    natural_rows = []
    for record in records:
        natural_rows.append({"sid": record["sid"], "id": record["id"],
                             "sample": record["sample"], "L": record["L"],
                             "capped": record["capped"], "correct": record["natural_correct"],
                             "answer_tokens": len(tokenizer.encode(record["natural_answer"],
                                                                  add_special_tokens=False)),
                             **record["natural_metrics"]})
    (out_dir / "natural.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in natural_rows),
        encoding="utf-8")

    # --------------------------------------------------------------------------------------------------
    # Phase B compares all arms from each saved natural trace and writes a resumable checkpoint per batch.
    # --------------------------------------------------------------------------------------------------
    # → Phase A：每题生成 8 条自然轨迹
    # → natural_records：解析轨迹和基线指标
    # → Phase B：每条轨迹展开为 27 个实验 arm
    # → 生成答案、评分、写 jobs.jsonl
    # → summarize：汇总结果
    run_drop_tests(records, specs, engine, tokenizer, t_open, t_close,
                   lambda text: tokenizer.encode(text, add_special_tokens=False),
                   args, out_dir)
    summary = summarize(out_dir)
    print(f"{dataset}: {summary['selected_problems']} selected problems, "
          f"{summary['selected_traces']} traces, natural accuracy "
          f"{summary['natural_accuracy_pct']}%", flush=True)
    print(f"summary: {(out_dir / 'summary.json').resolve()}", flush=True)


# 解析命令行参数并检查 cut 网格和样本数等设置是否合法。
def parse_args(argv: list[str] | None = None):
    """Parse CLI parameters and reject invalid settings before loading data.

    Pass ``None`` to parse command-line arguments from the process. Tests or
    other Python callers may pass a list such as ``["--mock", "--samples", "1"]``.
    The full option list, defaults, and meanings are documented in this module's
    docstring and displayed by ``--help``.

    Args:
        argv: Optional list of command-line strings, excluding the program name.
    Returns:
        An argparse Namespace containing the validated run settings.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["all", "aime2026", "math500"], default="all")
    parser.add_argument("--aime-dataset", default="MathArena/aime_2026")
    parser.add_argument("--aime-problems", type=int, default=30)
    parser.add_argument("--math-problems", type=int, default=100)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--cut-points", type=float, nargs="+", default=[0.25, 0.50, 0.75])
    parser.add_argument("--cut-sizes", type=float, nargs="+", default=[0.10, 0.25])
    parser.add_argument("--max-think", type=int, default=None,
                        help="default: 32768 for AIME and 8192 for MATH-500")
    parser.add_argument("--max-len", type=int, default=40960)
    parser.add_argument("--answer-max", type=int, default=4096)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--gpu-util", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--trace-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/budget_drop_sweep"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hard-cut", action="store_true",
                        help="use bare </think> for cut_now instead of the early-stop phrase")
    parser.add_argument("--mock", action="store_true", help="fake generation; pipeline check only")
    parser.add_argument("--analyze", type=Path, default=None,
                        help="rebuild summary and CSV from one dataset output directory")
    args = parser.parse_args(argv)
    if args.analyze:
        return args
    if args.samples < 1 or args.batch_size < 1 or args.trace_batch_size < 1:
        parser.error("samples and batch sizes must be positive")
    try:
        make_arm_specs(args.cut_points, args.cut_sizes)
    except ValueError as error:
        parser.error(str(error))
    if args.dataset in ("all", "aime2026") and args.aime_problems < 1:
        parser.error("--aime-problems must be positive")
    if args.dataset in ("all", "math500") and args.math_problems < 1:
        parser.error("--math-problems must be positive")
    return args


# 程序总入口：加载题库、固定随机子集、选择 Engine/Mock，再逐数据集运行或分析。
def main(argv: list[str] | None = None) -> int:
    """Coordinate selection, backend setup, per-dataset runs, or analysis-only.

    Normal execution order is: parse options, load full populations, choose
    seeded subsets, construct one backend, then run each dataset separately.
    With ``--analyze``, generation is skipped and saved results are summarized.

    Args:
        argv: Optional CLI argument list; None means use sys.argv.
    Returns:
        Process exit code: zero for a completed run or analysis.
    """
    args = parse_args(argv)
    
    # If --analyze is specified, skip generation and only summarize the existing results.
    if args.analyze:
        summary = summarize(args.analyze)
        print(json.dumps(summary, indent=2))
        return 0

    # load populations and select seeded subsets
    dataset_names = ["aime2026", "math500"] if args.dataset == "all" else [args.dataset]
    populations = {name: load_population(name, args.aime_dataset) for name in dataset_names}
    counts = {"aime2026": args.aime_problems, "math500": args.math_problems}
    selected = {name: uniform_sample(population, counts[name], args.seed)
                for name, population in populations.items()}
    all_problems = [problem for items in selected.values() for problem in items]

    # select the engine: either a mock or the real Engine implementation
    if args.mock:
        engine = Mock(args.model, all_problems, args.batch_size)
    else:
        engine = Engine(args.model, args.max_len, args.gpu_util, args.max_num_seqs,
                        args.batch_size, args.seed)
    
    # retrieve the tokenizer from the engine for later use in dataset runs
    tokenizer = engine.tok
    for name in dataset_names:
        # 因为aime2026的思考轨迹可能更长，所以用了2^15作为默认最大思考长度，math500用2^13
        max_think = args.max_think or (32768 if name == "aime2026" else 8192)
        out_dir = args.out / name
        run_dataset(name, selected[name], engine, args, tokenizer, max_think, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
