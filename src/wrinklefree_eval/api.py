"""Clean Python API for WrinkleFree evaluation.

Simple one-liner interface:
    from wrinklefree_eval import evaluate
    results = evaluate("path/to/model", benchmark="bitdistill")
"""

from pathlib import Path
from typing import Any
import json
import logging

import lm_eval
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# Task name mapping from our configs to lm-eval task names
TASK_MAPPING = {
    # GLUE tasks (built into lm-eval)
    "mnli": "glue_mnli",
    "qnli": "glue_qnli",
    "sst2": "glue_sst2",
    # Summarization (custom task)
    "cnn_dailymail_summarization": "cnn_dailymail_summarization",
}

# Benchmark presets
BENCHMARK_PRESETS = {
    "bitdistill": ["mnli", "qnli", "sst2", "cnn_dailymail_summarization"],
    "glue": ["mnli", "qnli", "sst2"],
    "summarization": ["cnn_dailymail_summarization"],
    "smoke_test": ["sst2", "cnn_dailymail_summarization"],
}


def list_benchmarks() -> dict[str, list[str]]:
    """List available benchmark presets and their tasks.

    Returns:
        Dict mapping benchmark name to list of task names
    """
    return BENCHMARK_PRESETS.copy()


def evaluate(
    model_path: str,
    benchmark: str = "bitdistill",
    tasks: list[str] | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    batch_size: int | str = "auto",
    num_fewshot: int | dict[str, int] | None = None,
    limit: int | None = None,
    smoke_test: bool = False,
    output_dir: str | None = None,
    use_bitnet: bool = False,
    trust_remote_code: bool = True,
    verbosity: str = "INFO",
    **kwargs,
) -> dict[str, Any]:
    """Evaluate a model on BitDistill benchmarks.

    Simple one-liner API for evaluation:
        results = evaluate("path/to/model", benchmark="bitdistill")

    Args:
        model_path: HuggingFace model ID or local path to model
        benchmark: Benchmark preset ("bitdistill", "glue", "summarization", "smoke_test")
        tasks: Override tasks list (if None, uses benchmark preset)
        device: Device to run on ("cuda", "cpu")
        dtype: Model dtype ("float16", "bfloat16", "float32")
        batch_size: Batch size for evaluation ("auto" for automatic)
        num_fewshot: Number of few-shot examples (int for all tasks, or dict per task)
        limit: Limit number of samples per task (None = full dataset)
        smoke_test: Enable smoke test mode (limit=10 per task)
        output_dir: Directory to save results (None = don't save)
        use_bitnet: Use BitNet kernels if available
        trust_remote_code: Trust remote code in model config
        verbosity: Logging verbosity ("DEBUG", "INFO", "WARNING")
        **kwargs: Additional arguments passed to lm_eval.simple_evaluate

    Returns:
        Dict with results for each task:
        {
            "glue_sst2": {"accuracy": 0.92, ...},
            "cnn_dailymail_summarization": {"rouge1": 0.45, "rouge2": 0.21, ...},
            ...
        }
    """
    # Configure logging
    logging.basicConfig(level=getattr(logging, verbosity))

    # Resolve tasks from benchmark preset
    if tasks is None:
        if benchmark not in BENCHMARK_PRESETS:
            available = ", ".join(BENCHMARK_PRESETS.keys())
            raise ValueError(f"Unknown benchmark: {benchmark}. Available: {available}")
        tasks = BENCHMARK_PRESETS[benchmark]

    # Apply smoke test limits
    if smoke_test:
        limit = limit or 10
        logger.info(f"Smoke test mode: limiting to {limit} samples per task")

    # Map task names to lm-eval task names
    lm_eval_tasks = []
    for task in tasks:
        if task in TASK_MAPPING:
            lm_eval_tasks.append(TASK_MAPPING[task])
        else:
            # Assume it's already an lm-eval task name
            lm_eval_tasks.append(task)

    # Register custom tasks
    custom_tasks_dir = Path(__file__).parent / "tasks"
    if custom_tasks_dir.exists():
        lm_eval.tasks.include_path(str(custom_tasks_dir))
        logger.debug(f"Registered custom tasks from {custom_tasks_dir}")

    # Build model arguments
    model_args = f"pretrained={model_path}"
    model_args += f",dtype={dtype}"
    model_args += f",trust_remote_code={trust_remote_code}"

    # Handle few-shot configuration
    if isinstance(num_fewshot, dict):
        # Per-task few-shot (not directly supported, use default)
        num_fewshot_arg = 0
    else:
        num_fewshot_arg = num_fewshot if num_fewshot is not None else 0

    logger.info(f"Evaluating {model_path} on tasks: {lm_eval_tasks}")
    logger.info(f"Device: {device}, Dtype: {dtype}, Batch size: {batch_size}")

    # Run evaluation using lm_eval
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=lm_eval_tasks,
        num_fewshot=num_fewshot_arg,
        batch_size=batch_size,
        device=device,
        limit=limit,
        **kwargs,
    )

    # Extract and format results
    formatted_results = _format_results(results)

    # Save results if output_dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / "results.json"
        with open(results_file, "w") as f:
            json.dump(formatted_results, f, indent=2)
        logger.info(f"Results saved to {results_file}")

        # Save full results with samples
        full_results_file = output_path / "results_full.json"
        with open(full_results_file, "w") as f:
            # Convert to serializable format
            serializable = _make_serializable(results)
            json.dump(serializable, f, indent=2)

    return formatted_results


def _format_results(results: dict) -> dict[str, Any]:
    """Format lm_eval results into a cleaner structure.

    Args:
        results: Raw results from lm_eval.simple_evaluate

    Returns:
        Formatted results dict
    """
    formatted = {}

    if "results" not in results:
        return formatted

    for task_name, task_results in results["results"].items():
        formatted[task_name] = {}

        for metric_name, value in task_results.items():
            # Skip internal metrics
            if metric_name.startswith("_"):
                continue

            # Clean up metric names
            clean_name = metric_name
            if "," in metric_name:
                # Handle metrics like "acc,none"
                clean_name = metric_name.split(",")[0]

            # Handle stderr separately
            if "_stderr" in metric_name:
                continue

            formatted[task_name][clean_name] = value

            # Add stderr if available
            stderr_key = f"{metric_name}_stderr"
            if stderr_key in task_results:
                formatted[task_name][f"{clean_name}_stderr"] = task_results[stderr_key]

    return formatted


def _make_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, "item"):  # numpy/torch scalar
        return obj.item()
    else:
        return str(obj)


def evaluate_from_config(config_path: str) -> dict[str, Any]:
    """Evaluate using a Hydra config file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Evaluation results
    """
    cfg = OmegaConf.load(config_path)

    return evaluate(
        model_path=cfg.model_path,
        benchmark=cfg.get("benchmark", {}).get("name", "bitdistill"),
        device=cfg.get("device", "cuda"),
        dtype=cfg.get("dtype", "bfloat16"),
        batch_size=cfg.get("batch_size", "auto"),
        limit=cfg.get("benchmark", {}).get("limits", {}).get("default"),
        smoke_test=cfg.get("smoke_test", False),
        output_dir=cfg.get("output_dir"),
    )
