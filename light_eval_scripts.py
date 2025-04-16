import os
from datetime import timedelta

import torch
import yaml
from accelerate import Accelerator, InitProcessGroupKwargs

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_input import GenerationParameters
from lighteval.models.transformers.adapter_model import AdapterModelConfig
from lighteval.models.transformers.delta_model import DeltaModelConfig
from lighteval.models.transformers.transformers_model import BitsAndBytesConfig, TransformersModelConfig
from lighteval.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters

model_args = ""
tasks = "no_mmlu.txt"
# tasks = "leaderboard|truthfulqa:mc|0|0"
# === Common parameters ===
use_chat_template = False
system_prompt = None
dataset_loading_processes = 1
custom_tasks = None
cache_dir = None
num_fewshot_seeds = 1
# === saving ===
output_dir = "results"

push_to_hub = False
push_to_tensorboard = False
public_run = False
results_org = None
save_details = True
# === debug ===
max_samples = None
override_batch_size = 32 # don't use -1 (e.g. use 32) for torchao and quanto for some reason
job_id = 0

accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])

cache_dir = "/raid/derek_liu"
# cache_dir = os.getenv("HF_HOME")
env_config = EnvConfig(token=os.getenv("HF_TOKEN"), cache_dir=cache_dir)

evaluation_tracker = EvaluationTracker(
    output_dir=output_dir,
    save_details=save_details,
    push_to_hub=push_to_hub,
    push_to_tensorboard=push_to_tensorboard,
    public=public_run,
    hub_results_org=results_org,
)
pipeline_params = PipelineParameters(
    launcher_type=ParallelismManager.ACCELERATE,
    env_config=env_config,
    job_id=job_id,
    dataset_loading_processes=dataset_loading_processes,
    custom_tasks_directory=custom_tasks,
    override_batch_size=override_batch_size,
    num_fewshot_seeds=num_fewshot_seeds,
    max_samples=max_samples,
    use_chat_template=use_chat_template,
    system_prompt=system_prompt,
)


# model_name = "ISTA-DASLab/Llama-3.1-8B-Instruct-HIGGS-4bit"
# model_name = "mobiuslabsgmbh/Llama-3.1-8B-Instruct_4bitgs64_hqq_hf"
# model_name = "VPTQ-community/Meta-Llama-3.1-8B-Instruct-v12-k65536-4096-woft"
# model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "ISTA-DASLab/Meta-Llama-3.1-8B-Instruct-AQLM-PV-2Bit-1x16-hf"
model_name = "meta-llama/Llama-3.1-70B-Instruct"

config = {
    "base_params": {
        "model_args": f"pretrained={model_name},model_parallel=True",  # Example model
        "dtype": "bfloat16",  # Example dtype
        "compile": False,
    },
    "merged_weights": {
        "delta_weights": False,
        "adapter_weights": False,
        "base_model": None,
    }
}

# Creating optional quantization configuration
if config["base_params"]["dtype"] == "4bit":
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
elif config["base_params"]["dtype"] == "8bit":
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
else:
    quantization_config = None

# torchao
# from transformers import TorchAoConfig
# quantization_config = TorchAoConfig("int8_weight_only", group_size=128)
# config["base_params"]["dtype"] = "auto"

# quanto
# from transformers import QuantoConfig
# quantization_config = QuantoConfig(weights="int4")

# HQQ
# from transformers import HqqConfig
# quantization_config = HqqConfig(nbits=8, group_size=64)

# HIGGS
from transformers import HiggsConfig
quantization_config = HiggsConfig(bits=4)

# bnb 8bit
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# eetq
# from transformers import EetqConfig
# quantization_config = EetqConfig("int8")

# gptq
# from transformers import AutoTokenizer, GPTQConfig
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# quantization_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)

# We extract the model args
args_dict = {k.split("=")[0]: k.split("=")[1] for k in config["base_params"]["model_args"].split(",")}

args_dict["generation_parameters"] = GenerationParameters.from_dict(config)

# We store the relevant other args
args_dict["base_model"] = config["merged_weights"]["base_model"]
args_dict["compile"] = bool(config["base_params"]["compile"])
args_dict["dtype"] = config["base_params"]["dtype"]
args_dict["accelerator"] = accelerator
args_dict["quantization_config"] = quantization_config # comment out if model is already quantized
args_dict["batch_size"] = override_batch_size
# args_dict["multichoice_continuations_start_space"] = config["generation"][
#     "multichoice_continuations_start_space"
# ]
args_dict["multichoice_continuations_start_space"] = None
args_dict["use_chat_template"] = use_chat_template

args_dict["device"] = "cuda"

# Keeping only non null params
args_dict = {k: v for k, v in args_dict.items() if v is not None}

if config["merged_weights"].get("delta_weights", False):
    if config["merged_weights"]["base_model"] is None:
        raise ValueError("You need to specify a base model when using delta weights")
    model_config = DeltaModelConfig(**args_dict)
elif config["merged_weights"].get("adapter_weights", False):
    if config["merged_weights"]["base_model"] is None:
        raise ValueError("You need to specify a base model when using adapter weights")
    model_config = AdapterModelConfig(**args_dict)
elif config["merged_weights"]["base_model"] not in ["", None]:
    raise ValueError("You can't specify a base model if you are not using delta/adapter weights")
else:
    model_config = TransformersModelConfig(**args_dict)



pipeline = Pipeline(
    tasks=tasks,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    model_config=model_config,
)

pipeline.evaluate()

pipeline.show_results()

results = pipeline.get_results()

pipeline.save_and_push_results()

# results
