# evaluation.py
import time
import numpy as np
import tqdm
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from baseline import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl
from execution import check_correctness

def estimate_pass_at_k(num_samples, num_correct, k):
    import numpy as _np
    num_samples = _np.array(num_samples, dtype=int)
    num_correct = _np.array(num_correct, dtype=int)
    def estimator(n, c, k):
        n = int(n); c = int(c)
        if n == 0:
            return 0.0
        if n - c < k:
            return 1.0
        prob = 1.0 - _np.prod(1.0 - (_np.arange(c, c + k) / _np.arange(n - k + 1, n + 1)))
        return float(max(0.0, min(1.0, prob)))
    res = [_np.array([estimator(n, c, k) for n, c in zip(num_samples, num_correct)])]
    return res[0]

def evaluate_functional_correctness(sample_file, k=[1], n_workers=12, timeout=3.0, problem_file=HUMAN_EVAL):
    problems = read_problems(problem_file)
    samples = list(stream_jsonl(sample_file))
    n_samples = len(samples)
    if n_samples == 0:
        return {f"pass@{kk}": 0.0 for kk in (k if isinstance(k, (list, tuple)) else [k])}
    completion_counter = Counter()
    futures = []
    future_to_meta = {}

    start_all = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for s in samples:
            task_id = s["task_id"]
            completion = s.get("completion", "")
            cid = completion_counter[task_id]
            fut = ex.submit(check_correctness, problems[task_id], completion, timeout, cid)
            future_to_meta[fut] = (task_id, cid, s)
            completion_counter[task_id] += 1
        results_map = defaultdict(list)
        for fut in tqdm.tqdm(as_completed(list(future_to_meta.keys())), total=len(future_to_meta), desc="Running test suites"):
            task_id, cid, orig_sample = future_to_meta[fut]
            try:
                res = fut.result(timeout=timeout + 1)
                results_map[res["task_id"]].append((res["completion_id"], res))
            except Exception as e:
                results_map[task_id].append((cid, {
                    "task_id": task_id,
                    "completion_id": cid,
                    "passed": False,
                    "result": f"Error: {e}"
                }))
    totals, corrects = [], []
    for task_id, lst in results_map.items():
        lst.sort(key=lambda x: x[0])
        passed_list = [item[1].get("passed", False) for item in lst]
        totals.append(len(passed_list))
        corrects.append(sum(1 for p in passed_list if p))

    totals = np.array(totals, dtype=int)
    corrects = np.array(corrects, dtype=int)

    pass_at_k = {}
    k_list = k if isinstance(k, (list, tuple, np.ndarray)) else [k]
    for kk in k_list:
        try:
            val = estimate_pass_at_k(totals, corrects, kk)
            if hasattr(val, "__len__"):
                vf = float(val.mean()) if len(val) > 1 else float(val[0])
            else:
                vf = float(val)
        except Exception:
            vf = 0.0
        pass_at_k[f"pass@{kk}"] = round(vf, 5)
    def combine_generator():
        counters = {tid: 0 for tid in results_map.keys()}
        for s in stream_jsonl(sample_file):
            tid = s["task_id"]
            if tid in results_map and counters.get(tid, 0) < len(results_map[tid]):
                cid, item = results_map[tid][counters[tid]]
                counters[tid] += 1
                s["passed"] = item.get("passed", False)
                s["result"] = item.get("result", None)
                s["exec_time"] = item.get("exec_time", None)
            else:
                s["passed"] = False
                s["result"] = None
                s["exec_time"] = None
            yield s
    out_file = sample_file + "_results.jsonl"
    write_jsonl(out_file, combine_generator())
    elapsed = time.time() - start_all
    print(f"✅ Evaluation completed in {elapsed:.2f}s for {n_samples} samples.")
    return pass_at_k