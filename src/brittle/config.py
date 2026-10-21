"""Fixed settings for every run. Changing any of these changes the cache keys or the item sample."""

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
ITEMS = os.path.join(DATA, "items.jsonl")
RESULTS = os.path.join(ROOT, "results")
REPORTS = os.path.join(ROOT, "reports")
FIGURES = os.path.join(REPORTS, "figures")
HANDCHECK = os.path.join(ROOT, "handcheck")
CACHE_DB = os.path.join(ROOT, "cache", "llm.sqlite")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODELS = ["qwen3:8b", "llama3.1:8b", "gemma3:12b", "qwen3:30b-a3b"]

SEED = 7
BASE_OPTIONS = {"temperature": 0, "seed": SEED, "num_ctx": 4096}
NUM_PREDICT = {"gsm8k": 512, "mmlu": 128, "extract": 160}

# The installed qwen3:30b-a3b is the thinking-only release; think: false does not stop it reasoning
# (memory-rot found this). It runs with thinking on and a larger cap; only the final answer is parsed.
THINKING_ONLY = {"qwen3:30b-a3b"}
THINKING_NUM_PREDICT = 4096

# Pinned Hugging Face sources: (repo, revision, file path inside the repo, sha256 of the file).
SOURCES = {
    "gsm8k_test": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
                   "main/test-00000-of-00001.parquet",
                   "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"),
    "e2e_test": ("tuetschek/e2e_nlg_cleaned", "6f77a74f6ce3095770924452f3f1b9631f837535",
                 "default/test/0000.parquet",
                 "2ffd18a7c46d32dd34cebadf38413cd45dd614e81c25a6e53084a55363f64053"),
}
MMLU_REPO = "cais/mmlu"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
MMLU_SHA256 = {
    "high_school_biology": "73600cc5c8032ffd20115887ec4f2c868d75c403c70ac1d115295e72cb8d80e3",
    "high_school_geography": "d083f0a37e9802ff70af8c0ccee95dc6836d80db74f45327519756327076a2c1",
    "high_school_macroeconomics": "a0f2d6a33352452442a13ffb947ab04dcf16513e87bee7c52e8957c2bc761a4c",
    "high_school_psychology": "9df4a744eb6faa12bb359df1f808c0461c055f240b998a9ec9d97b1c4e49a958",
}
MMLU_SUBJECTS = ["high_school_biology", "high_school_geography", "high_school_macroeconomics",
                 "high_school_psychology"]
for _s in MMLU_SUBJECTS:
    SOURCES[f"mmlu_{_s}"] = (MMLU_REPO, MMLU_REVISION, f"{_s}/test-00000-of-00001.parquet", MMLU_SHA256[_s])

N_PER_TASK = 100      # main items per task
N_PILOT = 10          # pilot items per task, disjoint from the main items, excluded from results
TASKS = ["gsm8k", "mmlu", "extract"]

BOOT = 2000
SPLITS = 2000         # random half splits for the best-prompt analysis


def options(task: str, model: str = "") -> dict:
    n = THINKING_NUM_PREDICT if model in THINKING_ONLY else NUM_PREDICT[task]
    return {**BASE_OPTIONS, "num_predict": n}
