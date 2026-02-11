#!/usr/bin/env python3
# main_generate_baseline.py
"""
- PromptAdapter 支持 4 个 Prompt 实验维度：
    * exp_family = quality / complexity / num_demos / diversity / none
    * condition / num_demos / diversity_mode 控制具体变体
"""

import os
import time
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional

from tqdm import tqdm
from openai import OpenAI

from baseline import read_problems, write_jsonl
from execution import check_correctness  # 目前 baseline 内不使用，但保留导入以兼容旧代码

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-8b")
DEFAULT_PROBLEM_FILE = "HumanEval.jsonl"

DEFAULT_MAX_SAMPLES = 80
DEFAULT_EXP_FAMILY = "none"          # 可选: none / quality / complexity / num_demos / diversity
DEFAULT_CONDITION = None             # 比如 "clean" / "wrong_demo" / "simple" / "detailed" 等
DEFAULT_NUM_DEMOS = 0                # num_demos 实验时的默认 K
DEFAULT_DIVERSITY_MODE = "low"       # diversity 实验默认: low / high
DEFAULT_OUTPUT_FILE = "baseline_A2_default.jsonl"

GLOBAL_THINKING_MODE = False
DEFAULT_WORKERS = 10
DEFAULT_TEMPERATURE = 1
CALL_TIMEOUT = 120

def call_with_retry(func, retries: int = 2, base_delay: float = 1.0):
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(base_delay * (2 ** attempt) + 0.1)


def extract_usage(resp) -> Dict[str, int]:
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage", {})

    def as_dict(o):
        if o is None:
            return {}
        if isinstance(o, dict):
            return o
        try:
            return o.__dict__
        except Exception:
            return {}

    u = as_dict(usage)
    prompt = int(u.get("prompt_tokens", u.get("input_tokens", 0) or 0))
    completion = int(u.get("completion_tokens", u.get("output_tokens", 0) or 0))
    total = int(u.get("total_tokens", prompt + completion))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _canonical_solution_or_placeholder(problem: Dict[str, Any]) -> str:
    sol = problem.get("canonical_solution")
    if isinstance(sol, str) and sol.strip():
        return sol.strip()
    entry = problem.get("entry_point", "solution")
    return (
        f"def {entry}(*args, **kwargs):\n"
        f"    # Reference implementation not available; placeholder used in prompt only.\n"
        f"    raise NotImplementedError\n"
    )


def _make_wrong_solution(problem: Dict[str, Any]) -> str:
    sol = problem.get("canonical_solution")
    if not isinstance(sol, str) or not sol.strip():
        return _canonical_solution_or_placeholder(problem)

    lines = sol.splitlines()
    header_line = None
    indent = ""
    for line in lines:
        if line.strip().startswith("def "):
            header_line = line.rstrip()
            indent = line[: len(line) - len(line.lstrip())]
            break
    if not header_line:
        return _canonical_solution_or_placeholder(problem)

    body = (
        f"{indent}    # NOTE: intentionally wrong demo implementation\n"
        f"{indent}    return None\n"
    )
    return header_line + "\n" + body


def _pick_irrelevant_demo(task_id: str, problems: Dict[str, Dict[str, Any]], task_order: List[str]) -> Tuple[str, Dict[str, Any]]:
    if not task_order:
        task_order = sorted(problems.keys())
    if task_id not in task_order:
        task_order = sorted(problems.keys())
    idx = task_order.index(task_id)
    other_idx = (idx + 1) % len(task_order)
    other_tid = task_order[other_idx]
    return other_tid, problems[other_tid]


def _pick_demo_tasks(
    problems: Dict[str, Dict[str, Any]],
    task_order: List[str],
    k: int,
    exclude_task: Optional[str] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    if k <= 0:
        return []
    if not task_order:
        task_ids = sorted(problems.keys())
    else:
        task_ids = list(task_order)

    demo_ids: List[str] = []
    for tid in task_ids:
        if exclude_task is not None and tid == exclude_task:
            continue
        demo_ids.append(tid)
        if len(demo_ids) >= k:
            break

    return [(tid, problems[tid]) for tid in demo_ids]

# diversity 用的多种模板
DIVERSITY_TEMPLATES: List[Dict[str, str]] = [
    {
        "system": "You are an expert Python developer.",
        "prompt_prefix": "请根据以下描述实现一个完整的 Python 函数：\n\n",
    },
    {
        "system": "You are a senior Python engineer who writes clean and efficient code.",
        "prompt_prefix": "Read the following problem description and implement the required function in Python:\n\n",
    },
    {
        "system": "You are a meticulous Python programmer who always passes all unit tests.",
        "prompt_prefix": "Implement the following Python function so that it satisfies all (possibly hidden) tests:\n\n",
    },
    {
        "system": "You are a Python tutor. You write simple but correct Python functions.",
        "prompt_prefix": "Please write a correct Python function according to the following instructions:\n\n",
    },
]


def _build_base_messages(system_text: str, user_prompt_text: str) -> List[Dict[str, str]]:
    user_content = (
        f"{user_prompt_text}\n\n"
        "⚠️请仅输出完整的函数定义（以 def 开头），不要解释或添加注释。\n"
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_content},
    ]


def build_messages(
    task_id: str,
    problem: Dict[str, Any],
    all_problems: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    exp_family: str = cfg.get("exp_family", "none") or "none"
    condition: Optional[str] = cfg.get("condition")
    num_demos: int = int(cfg.get("num_demos", 0) or 0)
    diversity_mode: Optional[str] = cfg.get("diversity_mode")
    task_order: List[str] = cfg.get("task_order") or sorted(all_problems.keys())

    system_default = "You are an expert Python developer."
    base_prompt_text = f"请根据以下描述实现一个完整的 Python 函数：\n\n{problem['prompt']}"
    prompt_meta: Dict[str, Any] = {
        "exp_family": exp_family,
        "condition": condition,
        "prompt_variant": None,
        "num_demos": 0,
        "diversity_mode": None,
        "prompt_template_id": 0,
    }

    # 1. Prompt Quality
    if exp_family == "quality":
        variant = condition or "clean"
        prompt_meta["prompt_variant"] = variant

        if variant == "clean":
            msgs = _build_base_messages(system_default, base_prompt_text)
            return msgs, prompt_meta

        elif variant == "wrong_demo":
            wrong_code = _make_wrong_solution(problem)
            user_text = (
                "下面是一个示例任务及其解答（注意：示例实现是故意写错的，只用于展示格式）：\n\n"
                "【示例题目】\n"
                f"{problem['prompt']}\n\n"
                "【错误示例代码】\n"
                f"{wrong_code}\n\n"
                "现在请你忽略上述错误实现，仅将其作为格式参考。\n"
                "请为下面的正式任务重新编写**正确**的 Python 函数实现：\n\n"
                f"{problem['prompt']}"
            )
            msgs = _build_base_messages(system_default, user_text)
            return msgs, prompt_meta

        elif variant == "irrelevant_demo":
            other_tid, other_prob = _pick_irrelevant_demo(task_id, all_problems, task_order)
            demo_code = _canonical_solution_or_placeholder(other_prob)
            user_text = (
                "下面是一个示例编程任务及其解答（与当前任务无关，仅作格式参考）：\n\n"
                "【示例题目】\n"
                f"{other_prob['prompt']}\n\n"
                "【示例 Python 代码】\n"
                f"{demo_code}\n\n"
                "现在请你完成下面真正的目标任务：\n\n"
                f"{problem['prompt']}"
            )
            msgs = _build_base_messages(system_default, user_text)
            return msgs, prompt_meta

        elif variant == "bad_instruction":
            system_text = (
                "You are a helpful assistant. You may output explanations in natural language "
                "before or after the code, and you do not need to strictly ensure the code can run."
            )
            user_text = (
                f"请根据以下描述实现一个 Python 函数，并可以在代码前后加入你的解释：\n\n{problem['prompt']}\n\n"
                "你可以在输出中加入自然语言说明。"
            )
            msgs = _build_base_messages(system_text, user_text)
            return msgs, prompt_meta

        else:
            prompt_meta["prompt_variant"] = "clean(fallback)"
            msgs = _build_base_messages(system_default, base_prompt_text)
            return msgs, prompt_meta

    # 2. Prompt Complexity
    if exp_family == "complexity":
        variant = condition or "original"
        prompt_meta["prompt_variant"] = variant

        base = problem["prompt"].strip()
        if variant == "simple":
            lines = [ln for ln in base.splitlines() if ln.strip()]
            if len(lines) > 2:
                truncated = "\n".join(lines[:2])
            else:
                truncated = base
            user_text = f"请实现一个 Python 函数，满足以下简要要求：\n\n{truncated}"
            msgs = _build_base_messages(system_default, user_text)
            return msgs, prompt_meta

        elif variant == "detailed":
            entry = problem.get("entry_point", "solution")
            extra = (
                "\n\n请特别注意：\n"
                f"- 函数名必须为 `{entry}`。\n"
                "- 尽可能覆盖边界条件与异常输入。\n"
                "- 避免使用全局变量，保持实现简洁清晰。"
            )
            user_text = (
                "请根据以下较为详细的描述，实现一个健壮的 Python 函数：\n\n"
                f"{base}{extra}"
            )
            msgs = _build_base_messages(system_default, user_text)
            return msgs, prompt_meta

        else:  # original
            user_text = base_prompt_text
            msgs = _build_base_messages(system_default, user_text)
            return msgs, prompt_meta

    # 3. Number of Demonstrations
    if exp_family == "num_demos":
        prompt_meta["prompt_variant"] = "num_demos"
        prompt_meta["num_demos"] = int(num_demos)

        if num_demos <= 0:
            msgs = _build_base_messages(system_default, base_prompt_text)
            return msgs, prompt_meta

        demos = _pick_demo_tasks(all_problems, task_order, num_demos, exclude_task=task_id)

        demo_blocks: List[str] = []
        for idx, (tid, demo_prob) in enumerate(demos, start=1):
            demo_code = _canonical_solution_or_placeholder(demo_prob)
            block = (
                f"### 示例 {idx}\n"
                "【示例题目】\n"
                f"{demo_prob['prompt']}\n\n"
                "【示例 Python 解答】\n"
                f"{demo_code}\n"
            )
            demo_blocks.append(block)

        demos_text = "\n\n".join(demo_blocks)
        user_text = (
            "下面给出若干示例编程任务及其 Python 解答，请先阅读这些示例，然后完成最后的目标任务。\n\n"
            f"{demos_text}\n\n"
            "=== 现在请完成下面的目标任务 ===\n\n"
            f"{problem['prompt']}"
        )
        msgs = _build_base_messages(system_default, user_text)
        return msgs, prompt_meta

    # 4. Prompt Diversity
    if exp_family == "diversity":
        mode = (diversity_mode or "low").lower()
        prompt_meta["prompt_variant"] = "diversity"
        prompt_meta["diversity_mode"] = mode

        if mode == "high":
            if not task_order:
                task_order = sorted(all_problems.keys())
            if task_id in task_order:
                pos = task_order.index(task_id)
            else:
                pos = 0
            template_id = pos % len(DIVERSITY_TEMPLATES)
        else:
            template_id = 0

        tmpl = DIVERSITY_TEMPLATES[template_id]
        prompt_meta["prompt_template_id"] = template_id

        user_text = f"{tmpl['prompt_prefix']}{problem['prompt']}"
        msgs = _build_base_messages(tmpl["system"], user_text)
        return msgs, prompt_meta
    prompt_meta["prompt_variant"] = "default"
    msgs = _build_base_messages(system_default, base_prompt_text)
    return msgs, prompt_meta
def make_process_one(
    model_name: str,
    temperature: float,
    thinking_mode: bool,
    save_raw_flag: Optional[Dict[str, Any]],
    problems: Dict[str, Dict[str, Any]],
    prompt_cfg: Dict[str, Any],
):
    def process_one(entry: Tuple[str, Dict[str, Any]]):
        task_id, problem = entry
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

        messages, prompt_meta = build_messages(task_id, problem, problems, prompt_cfg)

        try:
            t0 = time.time()
            resp = call_with_retry(
                lambda: client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    extra_body={"enable_thinking": bool(thinking_mode)},
                    timeout=CALL_TIMEOUT,
                )
            )
            gen_time = time.time() - t0

            content_raw = resp.choices[0].message.content.strip()
            usage = extract_usage(resp)

            result = {
                "task_id": task_id,
                "completion": content_raw,
                "tokens": {
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                },
                "thinking_mode_enabled": bool(thinking_mode),
                "generation_time": float(gen_time),
                "note": "ok",
                "prompt_config": prompt_meta,
            }

            if save_raw_flag is not None and not save_raw_flag.get("saved", False):
                try:
                    with open("debug_last_response.json", "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "task_id": task_id,
                                "messages": messages,
                                "response": content_raw,
                                "prompt_config": prompt_meta,
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    save_raw_flag["saved"] = True
                except Exception:
                    pass

            return result, gen_time

        except Exception as e:
            err_result = {
                "task_id": task_id,
                "completion": "",
                "tokens": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "thinking_mode_enabled": bool(thinking_mode),
                "generation_time": 0.0,
                "note": f"exception:{e}",
                "prompt_config": prompt_meta,
            }
            return err_result, 0.0

    return process_one

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"最大样本数（默认 {DEFAULT_MAX_SAMPLES}）",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--thinking", action="store_true", help="启用 thinking 模式（覆盖全局）")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--save_raw", action="store_true", help="保存一次 raw response（含 messages）")

    parser.add_argument(
        "--exp_family",
        type=str,
        default=DEFAULT_EXP_FAMILY,
        choices=["none", "quality", "complexity", "num_demos", "diversity"],
        help="Prompt 实验维度：none / quality / complexity / num_demos / diversity",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=DEFAULT_CONDITION,
        help="具体 condition 名称，例如：clean / wrong_demo / simple / detailed 等",
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        default=DEFAULT_NUM_DEMOS,
        help="示例数量（只在 exp_family=num_demos 时生效）",
    )
    parser.add_argument(
        "--diversity_mode",
        type=str,
        default=DEFAULT_DIVERSITY_MODE,
        choices=["low", "high"],
        help="Prompt 多样性模式（只在 exp_family=diversity 时有意义）",
    )
    parser.add_argument(
        "--problem_file",
        type=str,
        default=DEFAULT_PROBLEM_FILE,
        help=f"HumanEval 题目文件（默认 {DEFAULT_PROBLEM_FILE}）",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"生成结果输出文件（默认 {DEFAULT_OUTPUT_FILE}）",
    )

    args = parser.parse_args()

    thinking_mode = args.thinking or GLOBAL_THINKING_MODE

    problems = read_problems(args.problem_file)
    items = list(problems.items())
    if args.max_samples:
        items = items[: args.max_samples]

    task_order = [tid for (tid, _) in sorted(problems.items(), key=lambda x: x[0])]

    print(
        f"🚀 Baseline 启动： samples={len(items)} "
        f"model={MODEL_NAME} THINKING_MODE={thinking_mode} workers={args.workers}"
    )
    print(
        f"    exp_family={args.exp_family} condition={args.condition} "
        f"num_demos={args.num_demos} diversity_mode={args.diversity_mode}"
    )
    print(f"    problem_file={args.problem_file}  output_file={args.output_file}")

    save_raw_flag: Optional[Dict[str, Any]] = {"saved": False} if args.save_raw else None

    prompt_cfg: Dict[str, Any] = {
        "exp_family": args.exp_family,
        "condition": args.condition,
        "num_demos": args.num_demos,
        "diversity_mode": args.diversity_mode,
        "task_order": task_order,
    }

    process_one = make_process_one(
        MODEL_NAME,
        args.temperature,
        thinking_mode,
        save_raw_flag,
        problems,
        prompt_cfg,
    )

    results_map: Dict[str, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, entry): entry for entry in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            entry = futures[fut]
            task_id = entry[0]
            try:
                res, elapsed = fut.result()
            except Exception as e:
                res = {
                    "task_id": task_id,
                    "completion": "",
                    "tokens": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "thinking_mode_enabled": thinking_mode,
                    "generation_time": 0.0,
                    "note": f"exception_in_future:{e}",
                    "prompt_config": {
                        "exp_family": args.exp_family,
                        "condition": args.condition,
                        "prompt_variant": "future_exception",
                        "num_demos": args.num_demos,
                        "diversity_mode": args.diversity_mode,
                        "prompt_template_id": 0,
                    },
                }
                elapsed = 0.0
            results_map[res["task_id"]] = res

    ordered: List[Dict[str, Any]] = []
    for (task_id, _) in items:
        if task_id in results_map:
            ordered.append(results_map[task_id])
        else:
            ordered.append(
                {
                    "task_id": task_id,
                    "completion": "",
                    "tokens": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "thinking_mode_enabled": thinking_mode,
                    "generation_time": 0.0,
                    "note": "missing",
                    "prompt_config": {
                        "exp_family": args.exp_family,
                        "condition": args.condition,
                        "prompt_variant": "missing",
                        "num_demos": args.num_demos,
                        "diversity_mode": args.diversity_mode,
                        "prompt_template_id": 0,
                    },
                }
            )

    write_jsonl(args.output_file, ordered)

    valid = [r for r in ordered if r["tokens"].get("total_tokens", 0) > 0]
    n_valid = max(1, len(valid))
    prompt_tot = sum(r["tokens"].get("prompt_tokens", 0) for r in ordered)
    comp_tot = sum(r["tokens"].get("completion_tokens", 0) for r in ordered)
    total_tot = sum(r["tokens"].get("total_tokens", 0) for r in ordered)
    model_total_runtime = sum(float(r.get("generation_time", 0.0) or 0.0) for r in ordered)
    avg_gen_time = model_total_runtime / n_valid

    print("\n📊 Baseline 统计：")
    print(f"样本数: {len(ordered)}  有效样本: {n_valid}")
    print(f"平均输入 tokens: {prompt_tot / n_valid:.2f}")
    print(f"平均输出 tokens: {comp_tot / n_valid:.2f}")
    print(f"平均总 tokens: {total_tot / n_valid:.2f}")
    print(f"平均生成耗时(模型端): {avg_gen_time:.3f} s")
    print(f"模型总运行时长: {model_total_runtime:.3f} s")
    print(f"结果已保存到 {args.output_file}")
    print(
        f"评估命令：python evaluate_functional_correctness.py "
        f"--sample_file {args.output_file} --problem_file {args.problem_file}"
    )

if __name__ == "__main__":
    main()