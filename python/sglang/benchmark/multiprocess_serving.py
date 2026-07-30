# SPDX-License-Identifier: Apache-2.0

"""Run sglang.benchmark.serving with multiple load-generator processes."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

VALUE_OPTIONS_OVERRIDDEN_BY_LAUNCHER = {
    "--max-concurrency",
    "--num-prompts",
    "--output-file",
    "--request-rate",
    "--warmup-requests",
}
FLAG_OPTIONS_OVERRIDDEN_BY_LAUNCHER = {
    "--disable-tqdm",
    "--flush-cache",
    "--output-details",
    "--plot-throughput",
}


@dataclass
class Worker:
    index: int
    num_prompts: int
    max_concurrency: Optional[int]
    request_rate: float
    process: subprocess.Popen
    log_file: Path
    result_file: Path
    ready_file: Path
    log_handle: Any


def split_integer(total: int, parts: int) -> List[int]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    if total < parts:
        raise ValueError(f"cannot split {total} into {parts} positive parts")
    quotient, remainder = divmod(total, parts)
    return [quotient + (index < remainder) for index in range(parts)]


def split_request_rate(total_rate: float, prompt_counts: Sequence[int]) -> List[float]:
    if math.isinf(total_rate):
        return [float("inf")] * len(prompt_counts)
    if not math.isfinite(total_rate) or total_rate <= 0:
        raise ValueError("request rate must be positive or inf")
    total_prompts = sum(prompt_counts)
    return [total_rate * count / total_prompts for count in prompt_counts]


def _get_option(
    command_args: Sequence[str], option: str, default: Optional[str] = None
) -> Optional[str]:
    for index, token in enumerate(command_args):
        if token == option:
            if index + 1 >= len(command_args):
                raise ValueError(f"{option} requires a value")
            return command_args[index + 1]
        prefix = option + "="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return default


def _has_flag(command_args: Sequence[str], option: str) -> bool:
    return option in command_args


def _strip_overridden_options(command_args: Sequence[str]) -> List[str]:
    cleaned = []
    index = 0
    while index < len(command_args):
        token = command_args[index]
        if token in VALUE_OPTIONS_OVERRIDDEN_BY_LAUNCHER:
            if index + 1 >= len(command_args):
                raise ValueError(f"{token} requires a value")
            index += 2
            continue
        if any(
            token.startswith(option + "=")
            for option in VALUE_OPTIONS_OVERRIDDEN_BY_LAUNCHER
        ):
            index += 1
            continue
        if token in FLAG_OPTIONS_OVERRIDDEN_BY_LAUNCHER:
            index += 1
            continue
        cleaned.append(token)
        index += 1
    return cleaned


def _metric_stats(values: Sequence[float], name: str) -> Dict[str, float]:
    samples: Any = np.asarray(values, dtype=np.float64) if values else 0
    return {
        f"mean_{name}_ms": float(np.mean(samples) * 1000),
        f"median_{name}_ms": float(np.median(samples) * 1000),
        f"std_{name}_ms": float(np.std(samples) * 1000),
        f"p90_{name}_ms": float(np.percentile(samples, 90) * 1000),
        f"p95_{name}_ms": float(np.percentile(samples, 95) * 1000),
        f"p99_{name}_ms": float(np.percentile(samples, 99) * 1000),
    }


def _peak_metrics(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {
            "max_output_tokens_per_s": 0.0,
            "max_concurrent_requests": 0,
        }

    min_start_time = min(sample["start_time"] for sample in samples)
    max_end_time = max(sample["start_time"] + sample["latency"] for sample in samples)
    duration_seconds = int(np.ceil(max_end_time - min_start_time)) + 1
    tokens_per_second = np.zeros(duration_seconds)
    concurrent_requests_per_second = np.zeros(duration_seconds)

    for sample in samples:
        token_times = [sample["start_time"] + sample["ttft"]]
        current_time = token_times[0]
        for itl_value in sample["itls"]:
            current_time += itl_value
            token_times.append(current_time)

        for token_time in token_times:
            second_bucket = int(token_time - min_start_time)
            if 0 <= second_bucket < duration_seconds:
                tokens_per_second[second_bucket] += 1

        request_start_second = int(sample["start_time"] - min_start_time)
        request_end_second = int(
            sample["start_time"] + sample["latency"] - min_start_time
        )
        for second in range(
            request_start_second, min(request_end_second + 1, duration_seconds)
        ):
            concurrent_requests_per_second[second] += 1

    return {
        "max_output_tokens_per_s": float(np.max(tokens_per_second)),
        "max_concurrent_requests": int(np.max(concurrent_requests_per_second)),
    }


def _validate_detail_lengths(result: Dict[str, Any], worker_index: int) -> int:
    required = (
        "successes",
        "latencies",
        "start_times",
        "ttfts",
        "itls",
        "output_lens",
    )
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(
            f"worker {worker_index} result lacks exact-aggregation fields: {missing}. "
            "Run it with the matching serving.py version."
        )
    lengths = {key: len(result[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"worker {worker_index} detail arrays have different lengths: {lengths}"
        )
    count = next(iter(lengths.values()))
    if "metric_itls" in result and len(result["metric_itls"]) != count:
        raise ValueError(
            f"worker {worker_index} metric_itls has length "
            f"{len(result['metric_itls'])}, expected {count}"
        )
    reported_completed = int(result.get("completed", sum(result["successes"])))
    observed_completed = sum(bool(success) for success in result["successes"])
    if reported_completed != observed_completed:
        raise ValueError(
            f"worker {worker_index} reports completed={reported_completed}, "
            f"but detail records contain {observed_completed} successes"
        )
    return count


def aggregate_results(
    child_results: Sequence[Dict[str, Any]],
    total_num_prompts: int,
    total_concurrency: Optional[int],
    total_request_rate: float,
) -> Dict[str, Any]:
    if not child_results:
        raise ValueError("no child benchmark results")

    benchmark_start_time = min(
        float(result["benchmark_start_time"]) for result in child_results
    )
    benchmark_end_time = max(
        float(result["benchmark_end_time"]) for result in child_results
    )
    duration = benchmark_end_time - benchmark_start_time
    if duration <= 0:
        raise ValueError(f"invalid aggregate benchmark duration: {duration}")

    successful_samples = []
    ttfts = []
    tpots = []
    metric_itls = []
    e2e_latencies = []
    observed_records = 0

    for worker_index, result in enumerate(child_results):
        count = _validate_detail_lengths(result, worker_index)
        observed_records += count
        metric_itl_rows = result.get("metric_itls", result["itls"])
        for index in range(count):
            if not result["successes"][index]:
                continue
            output_len = int(result["output_lens"][index])
            latency = float(result["latencies"][index])
            ttft = float(result["ttfts"][index])
            raw_itls = result["itls"][index]
            request_metric_itls = metric_itl_rows[index]
            start_time = float(result["start_times"][index])

            if output_len > 1:
                tpots.append((latency - ttft) / (output_len - 1))
            ttfts.append(ttft)
            metric_itls.extend(request_metric_itls)
            e2e_latencies.append(latency)
            successful_samples.append(
                {
                    "start_time": start_time,
                    "latency": latency,
                    "ttft": ttft,
                    "itls": raw_itls,
                }
            )

    completed = len(successful_samples)
    if completed == 0:
        raise ValueError("all requests failed; aggregate latency metrics are undefined")

    total_input = sum(int(result["total_input_tokens"]) for result in child_results)
    total_input_text = sum(
        int(result["total_input_text_tokens"]) for result in child_results
    )
    total_input_vision = sum(
        int(result["total_input_vision_tokens"]) for result in child_results
    )
    total_output = sum(int(result["total_output_tokens"]) for result in child_results)
    total_output_retokenized = sum(
        int(result["total_output_tokens_retokenized"]) for result in child_results
    )

    last_result = max(
        child_results, key=lambda result: float(result["benchmark_end_time"])
    )
    aggregate = {
        "tag": last_result.get("tag"),
        "backend": last_result.get("backend"),
        "dataset_name": last_result.get("dataset_name"),
        "request_rate": (
            "inf" if math.isinf(total_request_rate) else total_request_rate
        ),
        "max_concurrency": total_concurrency,
        "num_prompts": total_num_prompts,
        "num_processes": len(child_results),
        "benchmark_start_time": benchmark_start_time,
        "benchmark_end_time": benchmark_end_time,
        "duration": duration,
        "completed": completed,
        "failed": observed_records - completed,
        "total_input_tokens": total_input,
        "total_input_text_tokens": total_input_text,
        "total_input_vision_tokens": total_input_vision,
        "total_output_tokens": total_output,
        "total_output_tokens_retokenized": total_output_retokenized,
        "request_throughput": completed / duration,
        "input_throughput": total_input / duration,
        "output_throughput": total_output / duration,
        "output_throughput_retokenized": total_output_retokenized / duration,
        "total_throughput": (total_input + total_output) / duration,
        "total_throughput_retokenized": (total_input + total_output_retokenized)
        / duration,
        "concurrency": float(np.sum(e2e_latencies) / duration),
        "accept_length": last_result.get("accept_length"),
        "server_info": last_result.get("server_info"),
    }
    aggregate.update(_metric_stats(e2e_latencies, "e2e_latency"))
    aggregate.update(_metric_stats(ttfts, "ttft"))
    aggregate.update(_metric_stats(tpots, "tpot"))
    aggregate.update(_metric_stats(metric_itls, "itl"))
    aggregate["max_itl_ms"] = float(np.max(metric_itls or 0) * 1000)
    aggregate.update(_peak_metrics(successful_samples))

    cache_reports = [
        result.get("cache_report")
        for result in child_results
        if result.get("cache_report")
    ]
    if cache_reports:
        total_prompt_tokens = sum(
            int(report["total_prompt_tokens"]) for report in cache_reports
        )
        total_cached_tokens = sum(
            int(report["total_cached_tokens"]) for report in cache_reports
        )
        aggregate["cache_report"] = {
            "total_prompt_tokens": total_prompt_tokens,
            "total_cached_tokens": total_cached_tokens,
            "cache_hit_rate_pct": (
                round(100 * total_cached_tokens / total_prompt_tokens, 2)
                if total_prompt_tokens
                else 0.0
            ),
            "device_cached_tokens": _sum_optional(
                cache_reports, "device_cached_tokens"
            ),
            "host_cached_tokens": _sum_optional(cache_reports, "host_cached_tokens"),
            "storage_cached_tokens": _sum_optional(
                cache_reports, "storage_cached_tokens"
            ),
            "storage_backend": next(
                (
                    report.get("storage_backend")
                    for report in cache_reports
                    if report.get("storage_backend")
                ),
                None,
            ),
        }

    return aggregate


def _sum_optional(reports: Sequence[Dict[str, Any]], key: str) -> Optional[int]:
    values = [report.get(key) for report in reports]
    if not any(value is not None for value in values):
        return None
    return sum(int(value or 0) for value in values)


def _format_rate(rate: float) -> str:
    return "inf" if math.isinf(rate) else repr(rate)


def _format_concurrency(max_concurrency: Optional[int]) -> str:
    return "not set" if max_concurrency is None else str(max_concurrency)


def _build_worker_command(
    child_base_args: Sequence[str],
    total_num_prompts: int,
    max_concurrency: Optional[int],
    request_rate: float,
    warmup_requests: int,
    result_file: Path,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        *child_base_args,
        "--num-prompts",
        str(total_num_prompts),
        "--request-rate",
        _format_rate(request_rate),
        "--warmup-requests",
        str(warmup_requests),
        "--output-file",
        str(result_file),
        "--output-details",
        "--disable-tqdm",
    ]
    if max_concurrency is not None:
        command.extend(["--max-concurrency", str(max_concurrency)])
    return command


def print_aggregate_result(result: Dict[str, Any]) -> None:
    print("\n{s:{c}^{n}}".format(s=" Aggregate Serving Benchmark Result ", n=58, c="="))
    rows = (
        ("Backend:", result["backend"]),
        ("Processes:", result["num_processes"]),
        ("Traffic request rate:", result["request_rate"]),
        (
            "Max request concurrency:",
            _format_concurrency(result["max_concurrency"]),
        ),
        ("Successful requests:", result["completed"]),
        ("Failed requests:", result["failed"]),
        ("Benchmark duration (s):", f"{result['duration']:.2f}"),
        ("Total input tokens:", result["total_input_tokens"]),
        ("Total generated tokens:", result["total_output_tokens"]),
        ("Request throughput (req/s):", f"{result['request_throughput']:.2f}"),
        ("Input token throughput (tok/s):", f"{result['input_throughput']:.2f}"),
        ("Output token throughput (tok/s):", f"{result['output_throughput']:.2f}"),
        (
            "Peak output token throughput (tok/s):",
            f"{result['max_output_tokens_per_s']:.2f}",
        ),
        ("Peak concurrent requests:", result["max_concurrent_requests"]),
        ("Total token throughput (tok/s):", f"{result['total_throughput']:.2f}"),
        ("Concurrency:", f"{result['concurrency']:.2f}"),
    )
    for label, value in rows:
        print(f"{label:<42} {value}")

    for title, key in (
        ("End-to-End Latency", "e2e_latency"),
        ("Time to First Token", "ttft"),
        ("Time per Output Token", "tpot"),
        ("Inter-Token Latency", "itl"),
    ):
        print(f"{title:-^58}")
        for percentile, label in (
            ("mean", "Mean"),
            ("median", "Median"),
            ("p90", "P90"),
            ("p95", "P95"),
            ("p99", "P99"),
        ):
            value = result[f"{percentile}_{key}_ms"]
            print(f"{label + ' ' + title + ' (ms):':<42} {value:.2f}")
    print("=" * 58)


def _parse_request_rate(value: str) -> float:
    try:
        rate = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "request rate must be a number or inf"
        ) from exc
    if rate <= 0 or math.isnan(rate):
        raise argparse.ArgumentTypeError("request rate must be positive or inf")
    return rate


def _validate_passthrough_args(
    parser: argparse.ArgumentParser,
    benchmark_args: Sequence[str],
    num_prompts: int,
) -> None:
    if not benchmark_args:
        parser.error(
            "missing bench_serving arguments; put them after '--', for example "
            "-- --backend sglang --dataset-name generated-shared-prefix ..."
        )
    dataset_name = _get_option(benchmark_args, "--dataset-name", "sharegpt")
    if dataset_name == "mooncake" or _has_flag(
        benchmark_args, "--use-trace-timestamps"
    ):
        parser.error("mooncake/trace-timestamp scheduling cannot be split by RR")
    if _has_flag(benchmark_args, "--profile"):
        parser.error("multi-process benchmark does not support --profile")
    if _has_flag(benchmark_args, "--plot-throughput"):
        parser.error(
            "multi-process benchmark does not support --plot-throughput; "
            "use the aggregate JSON result for plotting"
        )

    if dataset_name == "generated-shared-prefix":
        num_groups = int(_get_option(benchmark_args, "--gsp-num-groups", "64"))
        prompts_per_group = int(
            _get_option(benchmark_args, "--gsp-prompts-per-group", "16")
        )
        generated_count = num_groups * prompts_per_group
        if generated_count != num_prompts:
            parser.error(
                "generated-shared-prefix ignores --num-prompts: "
                f"--gsp-num-groups ({num_groups}) * --gsp-prompts-per-group "
                f"({prompts_per_group}) = {generated_count}, but --num-prompts "
                f"is {num_prompts}. Make the product equal the requested total."
            )


def _resolve_base_url(command_args: Sequence[str]) -> str:
    base_url = _get_option(command_args, "--base-url")
    if base_url:
        return base_url.rstrip("/")

    backend = _get_option(command_args, "--backend", "sglang")
    host = _get_option(command_args, "--host", "0.0.0.0")
    default_ports = {
        "sglang": 30000,
        "sglang-native": 30000,
        "sglang-oai": 30000,
        "sglang-oai-chat": 30000,
        "sglang-embedding": 30000,
        "lmdeploy": 23333,
        "lmdeploy-chat": 23333,
        "vllm": 8000,
        "vllm-chat": 8000,
        "vllm-embedding": 8000,
        "trt": 8000,
        "gserver": 9988,
        "truss": 8080,
    }
    port = int(
        _get_option(command_args, "--port", str(default_ports.get(backend, 30000)))
    )
    formatted_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{formatted_host}:{port}"


def _flush_cache(base_url: str, backend: str) -> None:
    headers = {}
    if api_key := os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    elif api_key := os.environ.get("API_KEY"):
        headers["Authorization"] = api_key

    cache_endpoint = (
        "/reset_prefix_cache" if backend.startswith("vllm") else "/flush_cache"
    )
    request = urllib.request.Request(
        base_url.rstrip("/") + cache_endpoint,
        data=b"",
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"flush_cache returned HTTP {response.status}")


def _tail(path: Path, line_count: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-line_count:])
    except OSError:
        return "<unable to read worker log>"


def _wait_until_ready(workers: Sequence[Worker], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        failed = [worker for worker in workers if worker.process.poll() is not None]
        if failed:
            worker = failed[0]
            raise RuntimeError(
                f"worker {worker.index} exited before the start barrier "
                f"(code {worker.process.returncode}):\n{_tail(worker.log_file)}"
            )
        if all(worker.ready_file.exists() for worker in workers):
            return
        if time.monotonic() >= deadline:
            missing = [
                str(worker.index)
                for worker in workers
                if not worker.ready_file.exists()
            ]
            raise TimeoutError(
                f"workers {', '.join(missing)} did not reach the start barrier"
            )
        time.sleep(0.05)


def _read_result(path: Path) -> Dict[str, Any]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected one JSON result in {path}, found {len(lines)}")
    return json.loads(lines[0])


def _terminate_workers(workers: Sequence[Worker]) -> None:
    for worker in workers:
        if worker.process.poll() is None:
            worker.process.terminate()
    for worker in workers:
        try:
            worker.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.process.kill()
            worker.process.wait()


def _wait_for_workers(workers: Sequence[Worker]) -> None:
    unfinished = set(range(len(workers)))
    while unfinished:
        for worker_index in list(unfinished):
            worker = workers[worker_index]
            return_code = worker.process.poll()
            if return_code is None:
                continue

            unfinished.remove(worker_index)
            worker.log_handle.close()
            if return_code != 0:
                raise RuntimeError(
                    f"worker {worker.index} failed with code {return_code}:\n"
                    f"{_tail(worker.log_file)}"
                )
        if unfinished:
            time.sleep(0.05)


def _worker_summary(result: Dict[str, Any], worker: Worker) -> Dict[str, Any]:
    summary = {
        "worker": worker.index,
        "assigned_num_prompts": worker.num_prompts,
        "assigned_max_concurrency": worker.max_concurrency,
        "assigned_request_rate": (
            "inf" if math.isinf(worker.request_rate) else worker.request_rate
        ),
        "result_file": str(worker.result_file),
        "log_file": str(worker.log_file),
    }
    metric_keys = (
        "duration",
        "completed",
        "total_input_tokens",
        "total_output_tokens",
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "total_throughput",
        "concurrency",
        "max_output_tokens_per_s",
        "max_concurrent_requests",
        "mean_e2e_latency_ms",
        "median_e2e_latency_ms",
        "std_e2e_latency_ms",
        "p90_e2e_latency_ms",
        "p95_e2e_latency_ms",
        "p99_e2e_latency_ms",
        "mean_ttft_ms",
        "median_ttft_ms",
        "std_ttft_ms",
        "p90_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "std_tpot_ms",
        "p90_tpot_ms",
        "p95_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "std_itl_ms",
        "p90_itl_ms",
        "p95_itl_ms",
        "p99_itl_ms",
    )
    summary.update({key: result.get(key) for key in metric_keys})
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch multiple sglang bench_serving processes and recompute global "
            "metrics from all raw request samples."
        )
    )
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help=(
            "Total maximum concurrency across all processes. If omitted, child "
            "benchmarks run without a concurrency semaphore."
        ),
    )
    parser.add_argument("--request-rate", type=_parse_request_rate, required=True)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("multiprocess_benchmark_result.jsonl"),
    )
    parser.add_argument(
        "--child-output-dir",
        type=Path,
        help="Directory for per-process JSONL and stdout logs.",
    )
    parser.add_argument("--barrier-timeout-sec", type=float, default=1800)
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to sglang.benchmark.serving after '--'.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    launcher_args = parser.parse_args(argv)
    benchmark_args = list(launcher_args.benchmark_args)
    if benchmark_args and benchmark_args[0] == "--":
        benchmark_args = benchmark_args[1:]

    if launcher_args.num_processes <= 0:
        parser.error("--num-processes must be positive")
    if launcher_args.num_prompts < launcher_args.num_processes:
        parser.error("--num-prompts must be at least --num-processes")
    if (
        launcher_args.max_concurrency is not None
        and launcher_args.max_concurrency < launcher_args.num_processes
    ):
        parser.error("--max-concurrency must be at least --num-processes")
    if launcher_args.barrier_timeout_sec <= 0:
        parser.error("--barrier-timeout-sec must be positive")
    _validate_passthrough_args(parser, benchmark_args, launcher_args.num_prompts)

    prompt_counts = split_integer(
        launcher_args.num_prompts, launcher_args.num_processes
    )
    concurrency_counts: List[Optional[int]]
    if launcher_args.max_concurrency is None:
        concurrency_counts = [None] * launcher_args.num_processes
    else:
        concurrency_counts = split_integer(
            launcher_args.max_concurrency, launcher_args.num_processes
        )
    request_rates = split_request_rate(launcher_args.request_rate, prompt_counts)

    warmup_requests = int(_get_option(benchmark_args, "--warmup-requests", "1"))
    backend = _get_option(benchmark_args, "--backend", "sglang")
    flush_cache = _has_flag(benchmark_args, "--flush-cache") or (
        "sglang" in backend
        and os.getenv("SGLANG_IS_IN_CI", "").lower() in ("1", "true")
    )
    base_seed = int(_get_option(benchmark_args, "--seed", "42"))
    child_base_args = _strip_overridden_options(benchmark_args)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    child_output_dir = launcher_args.child_output_dir or Path(
        f"multiprocess_benchmark_{timestamp}_{os.getpid()}"
    )
    child_output_dir.mkdir(parents=True, exist_ok=False)
    start_file = child_output_dir / "start"
    gsp_run_id = uuid.uuid4().hex[:8]
    gsp_run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    workers = []
    try:
        print("Worker assignment:")
        for index in range(launcher_args.num_processes):
            result_file = child_output_dir / f"worker_{index:02d}.jsonl"
            log_file = child_output_dir / f"worker_{index:02d}.log"
            ready_file = child_output_dir / f"worker_{index:02d}.ready"
            child_warmup_requests = warmup_requests if index == 0 else 0
            command = _build_worker_command(
                child_base_args=child_base_args,
                total_num_prompts=launcher_args.num_prompts,
                max_concurrency=concurrency_counts[index],
                request_rate=request_rates[index],
                warmup_requests=child_warmup_requests,
                result_file=result_file,
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "SGLANG_BENCH_NUM_SHARDS": str(launcher_args.num_processes),
                    "SGLANG_BENCH_SHARD_INDEX": str(index),
                    "SGLANG_BENCH_EXPECTED_NUM_REQUESTS": str(
                        launcher_args.num_prompts
                    ),
                    "SGLANG_BENCH_READY_FILE": str(ready_file.resolve()),
                    "SGLANG_BENCH_START_FILE": str(start_file.resolve()),
                    "SGLANG_BENCH_BARRIER_TIMEOUT_SEC": str(
                        launcher_args.barrier_timeout_sec
                    ),
                    "SGLANG_BENCH_REQUEST_SEED": str(base_seed + 1000003 + index),
                    "SGLANG_BENCH_GSP_RUN_ID": gsp_run_id,
                    "SGLANG_BENCH_GSP_RUN_TIMESTAMP": gsp_run_timestamp,
                    "SGLANG_BENCH_OMIT_GENERATED_TEXT": "1",
                }
            )
            log_handle = log_file.open("w")
            try:
                process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
            except Exception:
                log_handle.close()
                raise
            worker = Worker(
                index=index,
                num_prompts=prompt_counts[index],
                max_concurrency=concurrency_counts[index],
                request_rate=request_rates[index],
                process=process,
                log_file=log_file,
                result_file=result_file,
                ready_file=ready_file,
                log_handle=log_handle,
            )
            workers.append(worker)
            print(
                f"  worker {index}: prompts={prompt_counts[index]}, "
                f"concurrency={_format_concurrency(concurrency_counts[index])}, "
                f"request_rate={_format_rate(request_rates[index])}"
            )

        _wait_until_ready(workers, launcher_args.barrier_timeout_sec)
        if flush_cache:
            _flush_cache(_resolve_base_url(benchmark_args), backend)
        time.sleep(1.0)
        start_file.touch()

        _wait_for_workers(workers)

        child_results = [_read_result(worker.result_file) for worker in workers]
        aggregate = aggregate_results(
            child_results=child_results,
            total_num_prompts=launcher_args.num_prompts,
            total_concurrency=launcher_args.max_concurrency,
            total_request_rate=launcher_args.request_rate,
        )
        aggregate["workers"] = [
            _worker_summary(result, worker)
            for result, worker in zip(child_results, workers)
        ]
        aggregate["child_output_dir"] = str(child_output_dir.resolve())

        print("\nPer-process result:")
        for worker_result in aggregate["workers"]:
            print(
                f"  worker {worker_result['worker']}: "
                f"completed={worker_result['completed']}, "
                f"req/s={worker_result['request_throughput']:.2f}, "
                f"output tok/s={worker_result['output_throughput']:.2f}, "
                f"TTFT={worker_result['mean_ttft_ms']:.2f} ms, "
                f"TPOT={worker_result['mean_tpot_ms']:.2f} ms"
            )
        print_aggregate_result(aggregate)

        launcher_args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with launcher_args.output_file.open("a") as output_file:
            output_file.write(json.dumps(aggregate) + "\n")
        print(f"Aggregate result: {launcher_args.output_file.resolve()}")
        print(f"Child results and logs: {child_output_dir.resolve()}")
        return 0
    except (KeyboardInterrupt, Exception) as exc:
        _terminate_workers(workers)
        for worker in workers:
            if not worker.log_handle.closed:
                worker.log_handle.close()
        if isinstance(exc, KeyboardInterrupt):
            print("Interrupted", file=sys.stderr)
            return 130
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
