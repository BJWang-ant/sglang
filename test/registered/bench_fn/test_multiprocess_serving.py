import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sglang.benchmark.multiprocess_serving import (
    Worker,
    _build_worker_command,
    _flush_cache,
    _resolve_base_url,
    _wait_for_workers,
    aggregate_results,
    build_parser,
    split_integer,
    split_request_rate,
)
from sglang.benchmark.serving import _shard_input_requests
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _child_result(
    *,
    benchmark_start_time,
    benchmark_end_time,
    successes,
    latencies,
    start_times,
    ttfts,
    itls,
    metric_itls,
    output_lens,
    total_input,
    total_output,
):
    duration = benchmark_end_time - benchmark_start_time
    completed = sum(successes)
    return {
        "tag": None,
        "backend": "sglang",
        "dataset_name": "generated-shared-prefix",
        "benchmark_start_time": benchmark_start_time,
        "benchmark_end_time": benchmark_end_time,
        "duration": duration,
        "successes": successes,
        "latencies": latencies,
        "start_times": start_times,
        "ttfts": ttfts,
        "itls": itls,
        "metric_itls": metric_itls,
        "output_lens": output_lens,
        "completed": completed,
        "total_input_tokens": total_input,
        "total_input_text_tokens": total_input,
        "total_input_vision_tokens": 0,
        "total_output_tokens": total_output,
        "total_output_tokens_retokenized": total_output,
        "request_throughput": completed / duration,
        "output_throughput": total_output / duration,
        "mean_ttft_ms": 0,
        "mean_tpot_ms": 0,
        "mean_e2e_latency_ms": 0,
        "accept_length": None,
        "server_info": None,
    }


class TestMultiprocessServing(CustomTestCase):
    def test_max_concurrency_is_optional(self):
        parsed = build_parser().parse_args(
            [
                "--num-processes",
                "4",
                "--num-prompts",
                "16",
                "--request-rate",
                "20",
                "--",
                "--backend",
                "sglang",
            ]
        )
        self.assertIsNone(parsed.max_concurrency)

        unlimited_command = _build_worker_command(
            child_base_args=["--backend", "sglang"],
            total_num_prompts=16,
            max_concurrency=None,
            request_rate=5,
            warmup_requests=0,
            result_file=Path("worker.jsonl"),
        )
        self.assertNotIn("--max-concurrency", unlimited_command)

        limited_command = _build_worker_command(
            child_base_args=["--backend", "sglang"],
            total_num_prompts=16,
            max_concurrency=8,
            request_rate=5,
            warmup_requests=0,
            result_file=Path("worker.jsonl"),
        )
        concurrency_index = limited_command.index("--max-concurrency")
        self.assertEqual(limited_command[concurrency_index + 1], "8")

    def test_split_work(self):
        self.assertEqual(split_integer(10, 3), [4, 3, 3])
        self.assertEqual(split_integer(448, 8), [56] * 8)
        self.assertEqual(split_request_rate(20, [4, 3, 3]), [8, 6, 6])
        self.assertTrue(
            all(
                value == float("inf")
                for value in split_request_rate(float("inf"), [2, 2])
            )
        )

    def test_backend_specific_default_ports_and_cache_flush(self):
        self.assertEqual(
            _resolve_base_url(["--backend", "vllm-chat"]),
            "http://0.0.0.0:8000",
        )
        self.assertEqual(
            _resolve_base_url(["--backend", "lmdeploy-chat"]),
            "http://0.0.0.0:23333",
        )

        response = MagicMock()
        response.__enter__.return_value.status = 200
        with patch(
            "sglang.benchmark.multiprocess_serving.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            _flush_cache("http://server:8000", "vllm-chat")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://server:8000/reset_prefix_cache")

    def test_worker_failure_is_detected_without_waiting_in_index_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "worker.log"
            log_path.write_text("worker failed")

            still_running = MagicMock()
            still_running.poll.return_value = None
            failed = MagicMock()
            failed.poll.return_value = 7
            workers = [
                Worker(
                    index=0,
                    num_prompts=1,
                    max_concurrency=1,
                    request_rate=1,
                    process=still_running,
                    log_file=log_path,
                    result_file=Path(temp_dir) / "result-0.jsonl",
                    ready_file=Path(temp_dir) / "ready-0",
                    log_handle=io.StringIO(),
                ),
                Worker(
                    index=1,
                    num_prompts=1,
                    max_concurrency=1,
                    request_rate=1,
                    process=failed,
                    log_file=log_path,
                    result_file=Path(temp_dir) / "result-1.jsonl",
                    ready_file=Path(temp_dir) / "ready-1",
                    log_handle=io.StringIO(),
                ),
            ]

            with self.assertRaisesRegex(RuntimeError, "worker 1 failed with code 7"):
                _wait_for_workers(workers)

    def test_request_shards_are_disjoint_and_complete(self):
        requests = list(range(10))
        shards = []
        for index in range(3):
            with patch.dict(
                os.environ,
                {
                    "SGLANG_BENCH_NUM_SHARDS": "3",
                    "SGLANG_BENCH_SHARD_INDEX": str(index),
                },
            ):
                shards.append(_shard_input_requests(requests))

        self.assertEqual(shards, [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]])
        self.assertEqual(sorted(value for shard in shards for value in shard), requests)

    def test_aggregate_recomputes_request_level_metrics(self):
        child_results = [
            _child_result(
                benchmark_start_time=100.0,
                benchmark_end_time=104.0,
                successes=[True],
                latencies=[2.0],
                start_times=[100.0],
                ttfts=[0.5],
                itls=[[0.75, 0.75]],
                metric_itls=[[0.75, 0.75]],
                output_lens=[3],
                total_input=10,
                total_output=3,
            ),
            _child_result(
                benchmark_start_time=100.1,
                benchmark_end_time=104.5,
                successes=[True, False],
                latencies=[1.0, 0.0],
                start_times=[100.1, 0.0],
                ttfts=[0.2, 0.0],
                itls=[[0.8], []],
                metric_itls=[[0.8], []],
                output_lens=[2, 0],
                total_input=20,
                total_output=2,
            ),
        ]

        result = aggregate_results(
            child_results,
            total_num_prompts=3,
            total_concurrency=2,
            total_request_rate=20,
        )

        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["failed"], 1)
        self.assertAlmostEqual(result["duration"], 4.5)
        self.assertAlmostEqual(result["request_throughput"], 2 / 4.5)
        self.assertAlmostEqual(result["output_throughput"], 5 / 4.5)
        self.assertAlmostEqual(result["mean_e2e_latency_ms"], 1500)
        self.assertAlmostEqual(result["mean_ttft_ms"], 350)
        self.assertAlmostEqual(result["mean_tpot_ms"], 775)
        self.assertAlmostEqual(result["concurrency"], 3 / 4.5)
        self.assertEqual(result["max_output_tokens_per_s"], 2)
        self.assertEqual(result["max_concurrent_requests"], 2)

        unlimited_result = aggregate_results(
            child_results,
            total_num_prompts=3,
            total_concurrency=None,
            total_request_rate=20,
        )
        self.assertIsNone(unlimited_result["max_concurrency"])

    def test_aggregate_rejects_mismatched_metric_details(self):
        child = _child_result(
            benchmark_start_time=100.0,
            benchmark_end_time=101.0,
            successes=[True],
            latencies=[1.0],
            start_times=[100.0],
            ttfts=[0.2],
            itls=[[0.8]],
            metric_itls=[],
            output_lens=[2],
            total_input=10,
            total_output=2,
        )
        with self.assertRaisesRegex(ValueError, "metric_itls has length"):
            aggregate_results(
                [child],
                total_num_prompts=1,
                total_concurrency=1,
                total_request_rate=1,
            )


if __name__ == "__main__":
    unittest.main()
