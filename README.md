# Aligning Large Language Models with Preference Privacy

This repository contains code implementing the methods described in the paper:

> **[Aligning Large Language Models with Preference Privacy]**

---

## 📘 Overview

This project explores **preference-level privacy** in aligning large language models (LLMs) using human feedback. It implements:

- **Randomized Response (RR)**: Adds label-level noise to ensure differential privacy for human preferences.
- **DP-Stochastic Gradient Descent (DP-SGD)**: Provides full-data differential privacy, used for comparison.
- **PROPS (Progressively Private Self-Alignment)**: A multi-stage self-alignment strategy that refines noisy labels using intermediate models trained under privacy constraints.

---

## 🔧 Features

- ✅ Privacy-preserving training using **label-only differential privacy** (via RR).
- ✅ Multi-stage alignment with **PROPS**, improving performance under tight privacy budgets.
- ✅ Tools for comparing **privacy–utility trade-offs** across alignment strategies.
- ✅ Compatible with Hugging Face Transformers.

---

## 📊 Datasets Used

PROPS has been evaluated using the following **three preference datasets**:

1. **[AlpacaEval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval)** – Benchmark for evaluating LLMs with preference comparisons.
2. **[Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)** – Helpful and harmless human feedback dataset.
3. **[Truthy-DPO-v0.1](https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1)** – Preference dataset tailored for Direct Preference Optimization (DPO).

---

## 🧠 Models Evaluated

We tested PROPS on the following pretrained language models:

- **[GPT2-Medium](https://huggingface.co/gpt2-medium)** (355M parameters)
- **[GPT2-Large](https://huggingface.co/gpt2-large)** (774M parameters)
- **[Pythia-1B](https://huggingface.co/EleutherAI/pythia-1b)** – Open LLM from EleutherAI

---
