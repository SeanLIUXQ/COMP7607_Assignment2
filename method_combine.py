#!/usr/bin/env python3
# method_combine.py

import os
import time
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional

from tqdm import tqdm
from openai import OpenAI

from baseline import read_problems, write_jsonl
from execution import check_correctness

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-8b")

DEFAULT_PROBLEM_FILE = "HumanEval.jsonl"
DEFAULT_OUTPUT_FILE = "combine_results.jsonl"

GLOBAL_THINKING_MODE = False
DEFAULT_WORKERS = 10

DEFAULT_K_SR = 2
DEFAULT_REPAIR_PER = 2

DEFAULT_TEMP_SR = 0.7
DEFAULT_TEMP_FUNC = 0.7

CALL_TIMEOUT = 120


DEFAULT_EXP_FAMILY = "none"       # none | quality | complexity | num_demos | diversity
DEFAULT_CONDITION = None
DEFAULT_NUM_DEMOS = 0             # 0,1,2,4
DEFAULT_DIVERSITY_MODE = "low"    # low | high

def call_with_retry(func, retries: int = 2, base_delay: float = 1.0):
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(base_delay * (2 ** attempt) + 0.1)


def extract_usage(resp) -> Dict[str, int]:
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage", {})

    def to_dict(o):
        if o is None:
            return {}
        if isinstance(o, dict):
            return o
        try:
            return o.__dict__
        except Exception:
            return {}

    u = to_dict(usage)
    prompt = int(u.get("prompt_tokens", u.get("input_tokens", 0) or 0))
    completion = int(u.get("completion_tokens", u.get("output_tokens", 0) or 0))
    total = int(u.get("total_tokens", prompt + completion))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def clean_first_function(src: str) -> str:
    if not src:
        return ""
    s = src.strip()
    if s.startswith("```"):
        i = s.find("\n")
        if i != -1:
            s = s[i + 1 :]
        if s.endswith("```"):
            s = s[:-3]
    j = s.find("def ")
    if j != -1:
        s = s[j:]
    return s.strip()

def _canonical_solution_or_placeholder(problem: Dict[str, Any]) -> str:
    sol = problem.get("canonical_solution")
    if isinstance(sol, str) and sol.strip():
        return sol.strip()
    entry = problem.get("entry_point", "solution")
    return (
        f"def {entry}(*args, **kwargs):\n"
        f"    # Reference implementation not available; placeholder body.\n"
        f"    raise NotImplementedError\n"
    )


def _make_wrong_solution(problem: Dict[str, Any]) -> str:
    sol = problem.get("canonical_solution")
    if not isinstance(sol, str) or not sol.strip():
        return _canonical_solution_or_placeholder(problem)

    lines = sol.splitlines()
    header = None
    indent = ""
    for line in lines:
        if line.strip().startswith("def "):
            header = line.rstrip()
            indent = line[: len(line) - len(line.lstrip())]
            break
    if not header:
        return _canonical_solution_or_placeholder(problem)

    body = (
        f"{indent}    # Intentionally wrong demo implementation.\n"
        f"{indent}    return None\n"
    )
    return header + "\n" + body


def _pick_irrelevant_demo(task_id: str,
                          problems: Dict[str, Dict[str, Any]],
                          order: List[str]) -> Tuple[str, Dict[str, Any]]:
    if not order:
        order = sorted(problems.keys())
    if task_id not in order:
        order = sorted(problems.keys())
    idx = order.index(task_id)
    other_idx = (idx + 1) % len(order)
    other_tid = order[other_idx]
    return other_tid, problems[other_tid]


def _pick_demo_tasks(problems: Dict[str, Dict[str, Any]],
                     order: List[str],
                     k: int,
                     exclude_task: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
    if k <= 0:
        return []
    if not order:
        order = sorted(problems.keys())
    demo_ids: List[str] = []
    for tid in order:
        if exclude_task is not None and tid == exclude_task:
            continue
        demo_ids.append(tid)
        if len(demo_ids) >= k:
            break
    return [(tid, problems[tid]) for tid in demo_ids]


DIVERSITY_TEMPLATES: List[Dict[str, str]] = [
    {
        "system": "You are an expert Python developer.",
        "prefix": "请根据以下描述实现一个完整的 Python 函数：\n\n",
    },
    {
        "system": "You are a senior Python engineer who writes clean and efficient code.",
        "prefix": "Read the following problem description and implement the required Python function:\n\n",
    },
    {
        "system": "You are a meticulous Python programmer who always passes all unit tests.",
        "prefix": "Implement the following Python function so that it satisfies all tests:\n\n",
    },
    {
        "system": "You are a Python tutor. You write simple but correct Python functions.",
        "prefix": "Please write a correct Python function according to the following instructions:\n\n",
    },
]


def _base_messages(system_text: str, user_text: str) -> List[Dict[str, str]]:
    user_content = (
        f"{user_text}\n\n"
        "⚠️请仅输出完整的函数定义（以 def 开头），不要解释或添加注释。\n"
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_content},
    ]


def build_prompt_messages(task_id: str,
                          problem: Dict[str, Any],
                          all_problems: Dict[str, Dict[str, Any]],
                          cfg: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    exp_family: str = cfg.get("exp_family", DEFAULT_EXP_FAMILY) or "none"
    condition: Optional[str] = cfg.get("condition", DEFAULT_CONDITION)
    num_demos: int = int(cfg.get("num_demos", DEFAULT_NUM_DEMOS) or 0)
    diversity_mode: str = (cfg.get("diversity_mode", DEFAULT_DIVERSITY_MODE) or "low").lower()
    order: List[str] = cfg.get("task_order") or sorted(all_problems.keys())

    system_default = "You are an expert Python developer."
    base_text = f"请根据以下描述实现一个完整的 Python 函数：\n\n{problem['prompt']}"

    meta: Dict[str, Any] = {
        "exp_family": exp_family,
        "condition": condition,
        "num_demos": num_demos,
        "diversity_mode": diversity_mode,
        "prompt_variant": None,
        "prompt_template_id": 0,
    }

    # ----- 1) Prompt quality -----
    if exp_family == "quality":
        variant = condition or "clean"
        meta["prompt_variant"] = variant

        if variant == "clean":
            msgs = _base_messages(system_default, base_text)
            return msgs, meta

        if variant == "wrong_demo":
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
            msgs = _base_messages(system_default, user_text)
            return msgs, meta

        if variant == "irrelevant_demo":
            other_tid, other_prob = _pick_irrelevant_demo(task_id, all_problems, order)
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
            msgs = _base_messages(system_default, user_text)
            return msgs, meta

        if variant == "bad_instruction":
            system_text = (
                "You are a helpful assistant. You may output explanations and you do not "
                "need to make code executable."
            )
            user_text = (
                f"请根据以下描述实现一个 Python 函数，并可以在代码前后加入你的解释：\n\n{problem['prompt']}\n\n"
                "你可以输出自然语言说明，不用保证代码一定可以运行。\n"
            )
            msgs = _base_messages(system_text, user_text)
            return msgs, meta

        # fallback
        meta["prompt_variant"] = "clean(fallback)"
        msgs = _base_messages(system_default, base_text)
        return msgs, meta

    # ----- 2) Prompt complexity -----
    if exp_family == "complexity":
        variant = condition or "original"
        meta["prompt_variant"] = variant
        base_prompt = problem["prompt"].strip()

        if variant == "simple":
            lines = [ln for ln in base_prompt.splitlines() if ln.strip()]
            if len(lines) > 2:
                truncated = "\n".join(lines[:2])
            else:
                truncated = base_prompt
            user_text = f"请实现一个 Python 函数，满足以下简要要求：\n\n{truncated}"
            msgs = _base_messages(system_default, user_text)
            return msgs, meta

        if variant == "detailed":
            entry = problem.get("entry_point", "solution")
            extra = (
                "\n\n请特别注意：\n"
                f"- 函数名必须为 `{entry}`。\n"
                "- 覆盖正常输入和边界输入（空、0、负数等）。\n"
                "- 避免使用全局变量，保证实现简洁清晰。\n"
            )
            user_text = (
                "请根据以下较为详细的描述，实现一个健壮的 Python 函数：\n\n"
                f"{base_prompt}{extra}"
            )
            msgs = _base_messages(system_default, user_text)
            return msgs, meta

        # original
        user_text = base_text
        msgs = _base_messages(system_default, user_text)
        return msgs, meta

    # ----- 3) Number of demonstrations (few-shot) -----
    if exp_family == "num_demos":
        meta["prompt_variant"] = "num_demos"
        meta["num_demos"] = num_demos

        if num_demos <= 0:
            msgs = _base_messages(system_default, base_text)
            return msgs, meta

        demos = _pick_demo_tasks(all_problems, order, num_demos, exclude_task=task_id)
        blocks: List[str] = []
        for idx, (dtid, demo_prob) in enumerate(demos, start=1):
            demo_code = _canonical_solution_or_placeholder(demo_prob)
            block = (
                f"### 示例 {idx}\n"
                "【示例题目】\n"
                f"{demo_prob['prompt']}\n\n"
                "【示例 Python 解答】\n"
                f"{demo_code}\n"
            )
            blocks.append(block)
        demos_text = "\n\n".join(blocks)
        user_text = (
            "下面给出若干示例编程任务及其 Python 解答，请先阅读这些示例，然后完成最后的目标任务。\n\n"
            f"{demos_text}\n\n"
            "=== 现在请完成下面的目标任务 ===\n\n"
            f"{problem['prompt']}"
        )
        msgs = _base_messages(system_default, user_text)
        return msgs, meta

    # ----- 4) Prompt diversity -----
    if exp_family == "diversity":
        meta["prompt_variant"] = "diversity"
        meta["diversity_mode"] = diversity_mode

        if diversity_mode == "high":
            if not order:
                order = sorted(all_problems.keys())
            if task_id in order:
                pos = order.index(task_id)
            else:
                pos = 0
            tid = pos % len(DIVERSITY_TEMPLATES)
        else:
            tid = 0

        tmpl = DIVERSITY_TEMPLATES[tid]
        meta["prompt_template_id"] = tid

        user_text = f"{tmpl['prefix']}{problem['prompt']}"
        msgs = _base_messages(tmpl["system"], user_text)
        return msgs, meta

    meta["prompt_variant"] = "default"
    msgs = _base_messages(system_default, base_text)
    return msgs, meta

def run_self_refine_quick(
    problem: Dict[str, Any],
    client: OpenAI,
    base_messages: List[Dict[str, str]],
    k: int = DEFAULT_K_SR,
    temperature: float = DEFAULT_TEMP_SR,
    thinking_first: bool = False,
    save_raw_flag: Optional[Dict[str, Any]] = None,
    max_time: float = 30.0,
) -> Tuple[str, Dict[str, int], str]:
    tokens_acc = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    note_parts: List[str] = []
    code = ""
    result_str = None

    start = time.time()
    for step in range(k):
        if time.time() - start > max_time:
            note_parts.append("sr_timeout")
            break

        if step == 0:
            messages = base_messages
        else:
            fail_info = result_str or "Unknown error"
            messages = [
                base_messages[0],
                {
                    "role": "user",
                    "content": (
                        "下面是上一轮生成的函数实现以及失败信息，请你在保持函数签名不变的前提下修复它：\n\n"
                        "【上一版代码】\n"
                        f"{code}\n\n"
                        "【失败信息】\n"
                        f"{fail_info}\n\n"
                        "请输出修复后的完整函数定义（以 def 开头），不要解释或添加注释。\n"
                    ),
                },
            ]

        try:
            resp = call_with_retry(
                lambda: client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=temperature,
                    extra_body={"enable_thinking": bool(thinking_first)},
                    timeout=CALL_TIMEOUT,
                )
            )
        except Exception as e:
            note_parts.append(f"sr_exception_step{step+1}:{e}")
            break

        raw = resp.choices[0].message.content.strip()
        code = clean_first_function(raw)
        if save_raw_flag is not None and not save_raw_flag.get("saved", False):
            try:
                with open("debug_last_response_combine_sr.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "task_id": problem.get("task_id", "<unknown>"),
                            "step": step + 1,
                            "messages": messages,
                            "raw": raw,
                            "clean_code": code,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                save_raw_flag["saved"] = True
            except Exception:
                pass

        u = extract_usage(resp)
        for kx, vx in u.items():
            tokens_acc[kx] += vx

        if not code:
            note_parts.append(f"sr_empty_code_step{step+1}")
            continue

        try:
            r = check_correctness(problem, code, timeout=3.0, completion_id=None)
            result_str = r.get("result")
            if r.get("passed"):
                note_parts.append(f"sr_pass_step{step+1}")
                return code, tokens_acc, "|".join(note_parts)
            else:
                note_parts.append(f"sr_fail_step{step+1}:{result_str}")
        except Exception as e:
            result_str = f"exception:{e}"
            note_parts.append(f"sr_exec_error_step{step+1}:{e}")

    if not code:
        code = ""
    return code, tokens_acc, "|".join(note_parts) if note_parts else "sr_finished"


def run_codet_fast(
    problem: Dict[str, Any],
    client: OpenAI,
    init_func: str = "",
    base_messages: Optional[List[Dict[str, str]]] = None,
    repair_per: int = DEFAULT_REPAIR_PER,
    temperature_func: float = DEFAULT_TEMP_FUNC,
    save_raw_flag: Optional[Dict[str, Any]] = None,
    max_time: float = 40.0,
) -> Tuple[str, Dict[str, int], str]:
    tokens_acc = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    note_parts: List[str] = []
    code = init_func or ""
    result_str = None

    start = time.time()
    if not code:
        if base_messages is None:
            base_messages = [
                {
                    "role": "system",
                    "content": "You are a pragmatic Python developer.",
                },
                {
                    "role": "user",
                    "content": (
                        f"请根据下列描述实现完整的 Python 函数：\n\n{problem['prompt']}\n\n"
                        "⚠️请仅输出完整的函数定义（以 def 开头），不要解释或添加注释。\n"
                    ),
                },
            ]
        try:
            resp0 = call_with_retry(
                lambda: client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=base_messages,
                    temperature=temperature_func,
                    extra_body={"enable_thinking": False},
                    timeout=CALL_TIMEOUT,
                )
            )
            raw0 = resp0.choices[0].message.content.strip()
            code = clean_first_function(raw0)
            u0 = extract_usage(resp0)
            for kx, vx in u0.items():
                tokens_acc[kx] += vx
            note_parts.append("codet_init")
        except Exception as e:
            note_parts.append(f"codet_init_exception:{e}")
            return "", tokens_acc, "|".join(note_parts)

    # Repair loop
    for step in range(repair_per):
        if time.time() - start > max_time:
            note_parts.append("codet_timeout")
            break

        if not code:
            note_parts.append(f"codet_empty_code_step{step+1}")
            break

        # Execute current code for error message
        try:
            r = check_correctness(problem, code, timeout=3.0, completion_id=None)
            result_str = r.get("result")
            if r.get("passed"):
                note_parts.append(f"codet_pass_before_fix_step{step+1}")
                return code, tokens_acc, "|".join(note_parts)
        except Exception as e:
            result_str = f"exception:{e}"
            note_parts.append(f"codet_exec_error_step{step+1}:{e}")

        # Ask model to fix using error info
        messages = [
            {
                "role": "system",
                "content": "You are a developer who fixes Python functions.",
            },
            {
                "role": "user",
                "content": (
                    "下面是当前的函数实现以及测试失败信息，请在保持函数签名不变的前提下修复它：\n\n"
                    "【当前代码】\n"
                    f"{code}\n\n"
                    "【失败信息】\n"
                    f"{result_str}\n\n"
                    "请输出修复后的完整函数定义（以 def 开头），不要解释或添加注释。\n"
                ),
            },
        ]

        try:
            resp_fix = call_with_retry(
                lambda: client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=temperature_func,
                    extra_body={"enable_thinking": False},
                    timeout=CALL_TIMEOUT,
                )
            )
        except Exception as e:
            note_parts.append(f"codet_fix_exception_step{step+1}:{e}")
            break

        raw_fix = resp_fix.choices[0].message.content.strip()
        new_code = clean_first_function(raw_fix)
        u_fix = extract_usage(resp_fix)
        for kx, vx in u_fix.items():
            tokens_acc[kx] += vx

        if save_raw_flag is not None and not save_raw_flag.get("saved", False):
            try:
                with open("debug_last_response_combine_codet.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "task_id": problem.get("task_id", "<unknown>"),
                            "step": step + 1,
                            "messages": messages,
                            "raw": raw_fix,
                            "clean_code": new_code,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                save_raw_flag["saved"] = True
            except Exception:
                pass

        if not new_code:
            note_parts.append(f"codet_empty_new_code_step{step+1}")
            continue

        code = new_code

        # Immediately check repaired code
        try:
            r2 = check_correctness(problem, code, timeout=3.0, completion_id=None)
            result_str = r2.get("result")
            if r2.get("passed"):
                note_parts.append(f"codet_pass_after_fix_step{step+1}")
                return code, tokens_acc, "|".join(note_parts)
            else:
                note_parts.append(f"codet_fail_step{step+1}:{result_str}")
        except Exception as e:
            result_str = f"exception:{e}"
            note_parts.append(f"codet_exec_error_after_fix_step{step+1}:{e}")

    return code, tokens_acc, "|".join(note_parts) if note_parts else "codet_finished"

def make_process_one(
    k_sr: int,
    repair_per: int,
    temp_sr: float,
    temp_func: float,
    thinking_first: bool,
    save_raw_flag: Optional[Dict[str, Any]],
    problems: Dict[str, Dict[str, Any]],
    prompt_cfg: Dict[str, Any],
    max_sample_time: float = 60.0,
):
    def process_one(entry: Tuple[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
        task_id, problem = entry
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

        messages, meta = build_prompt_messages(task_id, problem, problems, prompt_cfg)

        start_all = time.time()

        # Stage 1: Self-Refine
        sr_code, sr_tokens, sr_note = run_self_refine_quick(
            problem,
            client,
            messages,
            k=k_sr,
            temperature=temp_sr,
            thinking_first=thinking_first,
            save_raw_flag=save_raw_flag,
            max_time=max_sample_time * 0.5,
        )

        # Check SR result
        sr_pass = False
        if sr_code:
            try:
                r_sr = check_correctness(problem, sr_code, timeout=3.0, completion_id=None)
                sr_pass = bool(r_sr.get("passed"))
            except Exception:
                sr_pass = False

        if sr_pass:
            final_code = sr_code
            final_note = f"sr_only|{sr_note}"
            ct_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        else:
            # Stage 2: CodeT repair
            ct_code, ct_tokens, ct_note = run_codet_fast(
                problem,
                client,
                init_func=sr_code,
                base_messages=messages,
                repair_per=repair_per,
                temperature_func=temp_func,
                save_raw_flag=save_raw_flag,
                max_time=max_sample_time * 0.5,
            )
            final_code = ct_code or sr_code or ""
            final_note = f"sr_then_codet|SR:{sr_note}|CT:{ct_note}"

        total_time = time.time() - start_all

        tokens = {
            "prompt_tokens": sr_tokens.get("prompt_tokens", 0) + ct_tokens.get("prompt_tokens", 0),
            "completion_tokens": sr_tokens.get("completion_tokens", 0) + ct_tokens.get("completion_tokens", 0),
            "total_tokens": sr_tokens.get("total_tokens", 0) + ct_tokens.get("total_tokens", 0),
        }

        format_ok = bool(final_code)

        result = {
            "task_id": task_id,
            "completion": final_code,
            "tokens": tokens,
            "thinking_mode_enabled": bool(thinking_first),
            "generation_time": float(total_time),
            "note": final_note,
            "format_ok": format_ok,
            "prompt_config": meta,
        }
        return result, total_time

    return process_one

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--k_sr", type=int, default=DEFAULT_K_SR)
    parser.add_argument("--repair_per", type=int, default=DEFAULT_REPAIR_PER)
    parser.add_argument("--temperature_sr", type=float, default=DEFAULT_TEMP_SR)
    parser.add_argument("--temperature_func", type=float, default=DEFAULT_TEMP_FUNC)
    parser.add_argument("--thinking_first", action="store_true")
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--max_time_per_sample", type=float, default=60.0)

    parser.add_argument(
        "--exp_family",
        type=str,
        default=DEFAULT_EXP_FAMILY,
        choices=["none", "quality", "complexity", "num_demos", "diversity"],
    )
    parser.add_argument("--condition", type=str, default=DEFAULT_CONDITION)
    parser.add_argument("--num_demos", type=int, default=DEFAULT_NUM_DEMOS)
    parser.add_argument(
        "--diversity_mode",
        type=str,
        default=DEFAULT_DIVERSITY_MODE,
        choices=["low", "high"],
    )

    parser.add_argument("--problem_file", type=str, default=DEFAULT_PROBLEM_FILE)
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE)

    args = parser.parse_args()

    thinking_first = args.thinking_first or GLOBAL_THINKING_MODE

    problems = read_problems(args.problem_file)
    items = list(problems.items())
    if args.max_samples:
        items = items[: args.max_samples]

    task_order = [tid for (tid, _) in sorted(problems.items(), key=lambda x: x[0])]

    prompt_cfg: Dict[str, Any] = {
        "exp_family": args.exp_family,
        "condition": args.condition,
        "num_demos": args.num_demos,
        "diversity_mode": args.diversity_mode,
        "task_order": task_order,
    }

    save_raw_flag: Optional[Dict[str, Any]] = {"saved": False} if args.save_raw else None

    print(
        f"🚀 Combine 启动： samples={len(items)} model={MODEL_NAME} workers={args.workers} "
        f"k_sr={args.k_sr} repair_per={args.repair_per} THINKING_FIRST={thinking_first}"
    )
    print(
        f"    exp_family={args.exp_family} condition={args.condition} "
        f"num_demos={args.num_demos} diversity_mode={args.diversity_mode}"
    )
    print(f"    problem_file={args.problem_file} output_file={args.output_file}")

    process_one = make_process_one(
        k_sr=args.k_sr,
        repair_per=args.repair_per,
        temp_sr=args.temperature_sr,
        temp_func=args.temperature_func,
        thinking_first=thinking_first,
        save_raw_flag=save_raw_flag,
        problems=problems,
        prompt_cfg=prompt_cfg,
        max_sample_time=args.max_time_per_sample,
    )

    results_map: Dict[str, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, entry): entry for entry in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Combining"):
            entry = futures[fut]
            task_id = entry[0]
            try:
                res, _ = fut.result()
            except Exception as e:
                res = {
                    "task_id": task_id,
                    "completion": "",
                    "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "thinking_mode_enabled": thinking_first,
                    "generation_time": 0.0,
                    "note": f"exception_in_future:{e}",
                    "format_ok": False,
                    "prompt_config": {
                        "exp_family": args.exp_family,
                        "condition": args.condition,
                        "num_demos": args.num_demos,
                        "diversity_mode": args.diversity_mode,
                        "prompt_variant": "future_exception",
                        "prompt_template_id": 0,
                    },
                }
            results_map[res["task_id"]] = res

    ordered: List[Dict[str, Any]] = []
    for (tid, _) in items:
        if tid in results_map:
            ordered.append(results_map[tid])
        else:
            ordered.append(
                {
                    "task_id": tid,
                    "completion": "",
                    "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "thinking_mode_enabled": thinking_first,
                    "generation_time": 0.0,
                    "note": "missing",
                    "format_ok": False,
                    "prompt_config": {
                        "exp_family": args.exp_family,
                        "condition": args.condition,
                        "num_demos": args.num_demos,
                        "diversity_mode": args.diversity_mode,
                        "prompt_variant": "missing",
                        "prompt_template_id": 0,
                    },
                }
            )

    write_jsonl(args.output_file, ordered)

    valid = [r for r in ordered if r.get("tokens", {}).get("total_tokens", 0) > 0]
    n_valid = max(1, len(valid))
    prompt_tot = sum(r.get("tokens", {}).get("prompt_tokens", 0) for r in ordered)
    comp_tot = sum(r.get("tokens", {}).get("completion_tokens", 0) for r in ordered)
    total_tot = sum(r.get("tokens", {}).get("total_tokens", 0) for r in ordered)
    model_total_runtime = sum(float(r.get("generation_time", 0.0) or 0.0) for r in ordered)
    avg_gen_time = model_total_runtime / n_valid

    print("\n📊 Combine 统计：")
    print(f"样本数: {len(ordered)}  有效样本: {n_valid}")
    print(f"平均输入 tokens: {prompt_tot / n_valid:.2f}")
    print(f"平均输出 tokens: {comp_tot / n_valid:.2f}")
    print(f"平均总 tokens: {total_tot / n_valid:.2f}")
    print(f"平均生成耗时(模型端): {avg_gen_time:.3f} s")
    print(f"模型总运行时长: {model_total_runtime:.3f} s")
    print(f"结果保存到 {args.output_file}")
    print(
        "评估命令：python evaluate_combine_functional_correctness.py "
        f"--sample_file {args.output_file} --problem_file {args.problem_file}"
    )

if __name__ == "__main__":
    main()