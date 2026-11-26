# PROPS: Progressively Private Self-alignment of Large Language Models

This repository contains code implementing the methods described in the paper:

> **PROPS: Progressively Private Self-alignment of Large Language Models**  
> Noel Teku, Fengwei Tian, Payel Bhattacharjee, Souradip Chakraborty, Amrit Singh Bedi, Ravi Tandon  
> Transactions on Machine Learning Research (TMLR), 2025 · [arXiv](https://arxiv.org/abs/2508.06783)

## Overview

PROPS aligns LLMs with human preferences while keeping the preference labels private. It uses DPO and compares three approaches:

- **Randomized Response (RR)**: flips each preference label at random, which gives label-level differential privacy.
- **DP-SGD**: adds noise to the gradients, which protects the whole training set. Used as a baseline.
- **PROPS (Progressively Private Self-Alignment)**: trains in stages. Each stage uses the model from the previous stage to clean up the noisy RR labels for the next stage.

## Status of this code

This is research code: Colab notebooks, not a packaged library. Some scripts from the paper were lost.
Below we list what is here, what is missing, and where the trained models are.

## What is in this repository

| Folder | File | What it does |
| --- | --- | --- |
| `SFT_Train/` | `FineTuning.ipynb` | Supervised fine-tuning (SFT) before alignment, using the DPO reference code |
| | `pythia1.yaml` | Example model config for Pythia-1B |
| `Label-DP/` | `Jon_DPO_RR.ipynb` | RR baseline (DPO on randomized-response labels), truthy-dpo-v0.1 |
| | `Alpaca_DPO_RR.ipynb` | RR baseline on AlpacaEval |
| `PROPS/` | `Jon_DPO_PROPS.ipynb` | 2-stage PROPS on truthy-dpo-v0.1 |
| | `Jon_DPO_3PROPS.ipynb` | 3-stage PROPS on truthy-dpo-v0.1 |
| | `Alpaca_DPO_2PROPS.ipynb` | 2-stage PROPS on AlpacaEval |
| `DP-SGD/` | `DP_SGD_DPO.ipynb`, `trainers.py`, `preference_datasets.py`, `config.yaml` | DP-SGD baseline for DPO |
| `Eval/` | `Response_Gen_Jon.ipynb` | Generates responses from trained models |
| | `gpt4-eval.py`, `Win_Rate.ipynb` | GPT-4 judge and win-tie-loss rates |
| | `measure_reward_new.py`, `preference_datasets.py`, `utils.py` | Helper scripts |

The notebooks are written for GPT-2 Large. To run GPT-2 Medium or Pythia-1B, change the model name.

## Datasets

| Name in files | Dataset | How it is loaded |
| --- | --- | --- |
| `Jon` | truthy-dpo-v0.1 | `load_dataset("jondurbin/truthy-dpo-v0.1")` |
| `Alpaca` | AlpacaEval | `load_dataset("reciprocate/alpaca-eval")` |
| `HH` | Anthropic HH-RLHF | `get_dataset("hh", ...)` in `preference_datasets.py` |

## Trained models

The trained checkpoints are on Hugging Face: [huggingface.co/Setpember](https://huggingface.co/Setpember).
Each model card says how the model was trained. Model names follow this pattern:

`{dataset}_{model}_{method}_epi_{ε}`

- **dataset**: `Jon` = truthy-dpo-v0.1, `Alpaca` = AlpacaEval, `HH` = Anthropic HH-RLHF
- **model**: `GPT2` = GPT-2 (124M), `GPT2M` = GPT-2 Medium, `GPT2L` = GPT-2 Large, `pythia` / `pythia1b` / `Pythia` = Pythia-1B
- **method**: `DPO_props` = 2-stage PROPS, `DPO_3props` = 3-stage PROPS, `DPO_RR` or `DPO` = RR baseline, `DPSGD` = DP-SGD baseline
- **ε**: `point1` or `01` = 0.1, `point5` or `05` = 0.5, `1`, `2`, and `inf` = no privacy

`sft_*`, `gpt2-hh`, and `pythia-1b-helpful-sft` are the SFT starting points.
The PPO and reward models (`*_PPO_*`, `Jon_ppo_*`, `Jon_reward_*`) are an exploratory PPO version of PROPS. They are not reported in the paper.

Some checkpoints (SFT and DP-SGD) are stored as `policy.pt`, the format used by the DPO reference code. Load them like this:

```python
import torch, transformers
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="Setpember/Jon_GPT2M_DPSGD_epi_1", filename="policy.pt")
model = transformers.AutoModelForCausalLM.from_pretrained("gpt2-medium")
model.load_state_dict(torch.load(path, map_location="cpu")["state"])
```

## Settings

The paper (Appendix) gives the settings used for the reported results: learning rate 5e-5 (3e-5 in some cases), batch size 4, and 3 epochs for PROPS and RR.
Some notebooks still hold values from later tests. For example, `Label-DP/Jon_DPO_RR.ipynb` sets 7 epochs and learning rate 5e-3.
To reproduce the paper, use the values in the paper.

## Not included

- Training notebooks for HH-RLHF
- Notebook outputs (they were cleared), so the exact numbers in the tables are not stored here

## Third-party code

`DP-SGD/trainers.py`, `DP-SGD/preference_datasets.py`, `Eval/preference_datasets.py`, and `Eval/utils.py` come from
[direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization) by Eric Mitchell et al.,
licensed under the Apache License 2.0 (see `LICENSE-direct-preference-optimization`).
We changed the two files in `DP-SGD/`: `trainers.py` adds DP-SGD (per-example clipping and noise), and `preference_datasets.py` adds more datasets, including truthy-dpo-v0.1. The two files in `Eval/` are unchanged.

---

## Citation

```bibtex
@article{teku2025props,
  title   = {{PROPS}: Progressively Private Self-alignment of Large Language Models},
  author  = {Teku, Noel and Tian, Fengwei and Bhattacharjee, Payel and Chakraborty, Souradip and Bedi, Amrit Singh and Tandon, Ravi},
  journal = {Transactions on Machine Learning Research},
  year    = {2025}
}
```
