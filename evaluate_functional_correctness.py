# evaluate_functional_correctness.py
import argparse
import json
import time
from evaluation import evaluate_functional_correctness


def entry_point(sample_file: str, problem_file: str, k: str = "1", n_workers: int = 12, timeout: float = 3.0):
    ks = [int(x.strip()) for x in k.split(",") if x.strip()]
    print("🔍 开始评估 ...")
    start = time.time()
    samples = []
    with open(sample_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                samples.append(json.loads(line))
            except Exception:
                continue

    total_samples = len(samples)
    print(f"Reading samples...\n{total_samples} 件样本读取完毕")

    pass_at_k = evaluate_functional_correctness(
        sample_file=sample_file,
        k=ks,
        n_workers=n_workers,
        timeout=timeout,
        problem_file=problem_file,
    )

    result_file = sample_file + "_results.jsonl"
    results = []
    try:
        with open(result_file, "r", encoding="utf-8") as rf:
            for line in rf:
                if not line.strip():
                    continue
                results.append(json.loads(line))
    except FileNotFoundError:
        print(f"⚠️ 结果文件 {result_file} 未找到")

    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    exec_times = [r.get("exec_time") for r in results if r.get("exec_time") is not None]
    avg_exec = sum(exec_times) / len(exec_times) if exec_times else 0.0

    prompt_tokens = [s.get("tokens", {}).get("prompt_tokens", 0) for s in samples]
    completion_tokens = [s.get("tokens", {}).get("completion_tokens", 0) for s in samples]
    total_tokens = [s.get("tokens", {}).get("total_tokens", 0) for s in samples]

    avg_prompt = (sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else 0.0
    avg_completion = (sum(completion_tokens) / len(completion_tokens)) if completion_tokens else 0.0
    total_used = sum(total_tokens)

    summary = {
        "total_samples": total,
        "passed": passed,
        "accuracy(%)": round((passed / total * 100) if total else 0, 5),
        "avg_exec_time_s": round(avg_exec, 5),
        "avg_prompt_tokens": round(avg_prompt, 5),
        "avg_completion_tokens": round(avg_completion, 5),
        "total_prompt_tokens": int(sum(prompt_tokens)),
        "total_completion_tokens": int(sum(completion_tokens)),
        "total_tokens_used": int(total_used),
        "pass@k": {k: float(f"{v:.5f}") for k, v in pass_at_k.items()},
        "results_file": result_file,
        "total_runtime_s": round(time.time() - start, 5),
    }

    print("\n📊 === 评估结果汇总 ===")
    print(f"  Total Samples         : {summary['total_samples']}")
    print(f"  Passed                : {summary['passed']} / {summary['total_samples']} ({summary['accuracy(%)']:.5f}%)")
    print(f"  Avg Exec Time         : {summary['avg_exec_time_s']:.5f} s")
    print(f"  Avg Prompt Tokens     : {summary['avg_prompt_tokens']:.5f}")
    print(f"  Avg Completion Tokens : {summary['avg_completion_tokens']:.5f}")
    print(f"  Total Tokens Used     : {summary['total_tokens_used']}")
    print(f"  Total Runtime         : {summary['total_runtime_s']:.5f} s")

    formatted_passk = ", ".join([f"{k}: {v:.5f}" for k, v in summary['pass@k'].items()])
    print(f"  pass@k                : {{{formatted_passk}}}")
    print(f"  Results File          : {summary['results_file']}")

    summary_path = sample_file.replace(".jsonl", "") + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2, ensure_ascii=False)
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
