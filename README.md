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

