# evaluate_combine_functional_correctness.py
import argparse
import json
import os
import shutil
import time

from evaluation import evaluate_functional_correctness


def entry_point(
    sample_file: str,
    problem_file: str,
    k: str = "1",
    n_workers: int = 12,
    timeout: float = 3.0,
):
    ks = [int(x.strip()) for x in k.split(",") if x.strip()]
    print("🔍 开始评估 Combine ...")
    t0 = time.time()

    samples = []
    with open(sample_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    total_samples = len(samples)
    total_prompt_tokens = sum(
        s.get("tokens", {}).get("prompt_tokens", 0) for s in samples
    )
    total_completion_tokens = sum(
        s.get("tokens", {}).get("completion_tokens", 0) for s in samples
    )
    total_tokens = total_prompt_tokens + total_completion_tokens

    pass_at_k = evaluate_functional_correctness(
        sample_file=sample_file,
        problem_file=problem_file,
        k=ks,
        n_workers=n_workers,
        timeout=timeout,
    )

    base_name = os.path.splitext(os.path.basename(sample_file))[0]
    raw_results_file = sample_file + "_results.jsonl"

    results_dir = "combine_eval_results"
    summaries_dir = "combine_eval_summaries"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    target_results_file = os.path.join(
        results_dir, f"{base_name}_results.jsonl"
    )

    if os.path.exists(raw_results_file):
        try:
            try:
                shutil.move(raw_results_file, target_results_file)
            except Exception:
                shutil.copy(raw_results_file, target_results_file)
        except Exception:
            target_results_file = raw_results_file
    else:
        target_results_file = raw_results_file

    passed = 0
    exec_times = []

    if os.path.exists(target_results_file):
        with open(target_results_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("passed"):
                    passed += 1
                et = r.get("exec_time")
                if isinstance(et, (int, float)):
                    exec_times.append(float(et))

    avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0.0
    total_runtime = time.time() - t0
    acc_pct = (passed / total_samples * 100.0) if total_samples > 0 else 0.0

    formatted_passk_items = []
    for kk, vv in pass_at_k.items():
        if isinstance(kk, str):
            key_str = kk
        else:
            key_str = f"pass@{kk}"
        formatted_passk_items.append(f"{key_str}: {vv:.5f}")
    formatted_passk = "{ " + ", ".join(formatted_passk_items) + " }"

    print(f"  Total Samples         : {total_samples}")
    print(f"  Passed                : {passed} / {total_samples} ({acc_pct:.5f}%)")
    print(f"  Avg Exec Time         : {avg_exec_time:.5f} s")
    print(f"  Avg Prompt Tokens     : {total_prompt_tokens / total_samples if total_samples else 0:.5f}")
    print(f"  Avg Completion Tokens : {total_completion_tokens / total_samples if total_samples else 0:.5f}")
    print(f"  Total Tokens Used     : {total_tokens}")
    print(f"  Total Runtime         : {total_runtime:.5f} s")
    print(f"  pass@k                : {formatted_passk}")
    print(f"  Results File          : {target_results_file}")

    summary = {
        "total_samples": total_samples,
        "passed": passed,
        "accuracy": acc_pct / 100.0,
        "avg_exec_time": avg_exec_time,
        "avg_prompt_tokens": total_prompt_tokens / total_samples if total_samples else 0.0,
        "avg_completion_tokens": total_completion_tokens / total_samples if total_samples else 0.0,
        "total_tokens_used": total_tokens,
        "total_runtime": total_runtime,
        "pass@k": pass_at_k,
        "results_file": target_results_file,
    }

    summary_path = os.path.join(summaries_dir, f"{base_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_file", type=str, required=True)
    parser.add_argument("--problem_file", type=str, required=True)
    parser.add_argument("--k", type=str, default="1")
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()

    entry_point(
        sample_file=args.sample_file,
        problem_file=args.problem_file,
        k=args.k,
        n_workers=args.n_workers,
        timeout=args.timeout,
    )