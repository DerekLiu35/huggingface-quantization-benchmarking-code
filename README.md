# huggingface-quantization-benchmarking-code
# LLM Quantization Benchmarking Suite

This repository provides scripts and configurations to benchmark the performance (latency, memory usage) and evaluate the quality (task-based evaluation) of various quantization methods applied to Large Language Models (LLMs), specifically focusing on the Llama family.

It utilizes two main tools:

1.  **`optimum-benchmark`**: For performance benchmarking (latency, memory) under controlled scenarios.
2.  **`lighteval` (via `light_eval_scripts.py`)**: For evaluating model accuracy on standard NLP tasks.

## Prerequisites

1.  **Python Environment**: Ensure you have a Python environment (e.g., conda, venv) set up.
2.  **GPU**: A CUDA-compatible GPU is required, especially for `optimum-benchmark` and quantized models. The necessary VRAM depends on the model size and quantization method.
3.  **Dependencies**: Install the necessary libraries. You'll likely need:
    *   `optimum-benchmark`
    *   `lighteval`
    *   `torch`, `transformers`, `accelerate`
    *   Specific quantization libraries based on the methods you want to test (e.g., `bitsandbytes`, `auto-gptq`, `autoawq`, `torchao`).
    
## 1. Performance Benchmarking (`optimum-benchmark`)

This tool measures latency and memory usage for model inference under specific configurations.

### How to Run

Use the `optimum-benchmark` command, specifying the configuration directory (`optimum_configs`) and the desired configuration file name (without the `.yaml` extension).

```bash
optimum-benchmark --config-dir /path/to/benchmark_quantization/optimum_configs --config-name <config_file_name>
```

### Examples

*   **Run AWQ benchmark:**
    ```bash
    optimum-benchmark --config-dir ./optimum_configs --config-name llama_awq
    ```

*   **Run baseline torchao_int4wo benchmark (with torch.compile, remove `cache_implementation: static` to run without compile):**
    ```bash
    optimum-benchmark --config-dir ./optimum_configs --config-name llama_torchao_int4wo
    ```
    
## Quality Evaluation (`lighteval`)

This script evaluates the model's performance on a set of NLP tasks defined in `no_mmlu.txt`.

### How to Run

Execute the Python script directly:

```bash
python light_eval_scripts.py
```

Unlike `optimum-benchmark`, configuration for `lighteval` is primarily done by **modifying the `light_eval_scripts.py` file itself.**

Key variables to modify:

1.  **`model_name`**: Set this to the Hugging Face identifier of the model you want to evaluate (e.g., `"meta-llama/Llama-3.1-8B-Instruct"`, `"hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"`).
2.  **`quantization_config`**:
    *   The script currently has several **commented-out examples** for different quantization methods using `transformers` integration (BitsAndBytes, HQQ, Higgs, GPTQ, TorchAO, Quanto, EETQ).
3.  **`tasks`**: Specifies the tasks to run. Defaults to `no_mmlu.txt`. You can change this to a different file or use the `lighteval` task string format (e.g., `"leaderboard|truthfulqa:mc|0|0"`).
4.  **`override_batch_size`**: Set the batch size for evaluation. Note the comment about potential issues with `-1` for some backends.