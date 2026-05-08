# Do LLMs Learn Reasoning Ability in Reinforcement Post-Training?

This project investigates whether reinforcement learning (RL) post-training genuinely improves reasoning ability in large language models, or mainly reshapes the output distribution over existing reasoning paths.

We compare supervised fine-tuned (SFT) and RL post-trained variants of OLMo2 and DeepSeek-Math models across mathematical reasoning, coding, commonsense reasoning, and instruction-following benchmarks.

The project includes:
- RL post-training using GRPO
- pass@k evaluation pipelines
- in-domain and out-of-domain reasoning benchmarks

---

# Repository Structure

```text
├── Training/
│   └── OLMo_7B_GRPO.ipynb
│
├── Math_Evaluation/
│   ├── Base_OLMo_Pass_k_Test.ipynb
│   ├── LoRA_OLMo_Pass_k_Test.ipynb
│   ├── Deepseek_Pass_k_Test_Math.ipynb
│   └── olmo2_math_results/
│
├── Out_of_Domain_Evaluation/
│   ├── Deepseek_BooIQ_Evaluation.ipynb
│   └── HumanEval_PassK.ipynb
│
├── notebooks/
│   └── OLMo_Baseline.ipynb
│
├── CommonsenseQA_IFEval_Evaluation/
│   ├── OLMo2_Baseline_CommonsenseQA_IFEval.ipynb
│   ├── OLMo2_GRPO_CommonsenseQA_IFEval.ipynb
│   ├── Deepseek_SFT_CommonsenseQA_Pass8.ipynb
│   ├── Deepseek_RL_CommonsenseQA_Pass8.ipynb
│   ├── Deepseek_IFEval.ipynb
│   └── results/
│       ├── olmo2_baseline_results/
│       ├── olmo2_grpo_results/
│       ├── deepseek_sft_results/
│       └── deepseek_rl_results/
│
└── README.md
```
---

# Code Structure

## Training/

Contains reinforcement learning post-training fine tuning code for OLMo2 7B using GRPO and LoRA.

### File
- `OLMo_7B_GRPO.ipynb`
  - store checkpoint in Google Drive
---
## Math_Evaluation/
Contains pass@k evaluation pipelines for mathematical reasoning datasets. The results will be saved in JSON format.
### Files
- `Base_OLMo_Pass_k_Test.ipynb` — pass@k evaluation for OLMo2 7B base model
- `LoRA_OLMo_Pass_k_Test.ipynb` — pass@k evaluation for OLMo2 7B + LoRA adapter
- `Deepseek_Pass_k_Test_Math.ipynb` — pass@k evaluation for DeepSeek-Math models

Supported datasets: GSM8K, MATH500, AIME
Supported models: DeepSeek-Math-7B-SFT, DeepSeek-Math-7B-RL
Supported metrics: pass@1, pass@4, pass@8, pass@16, pass@32, pass@64, pass@128

---

## Out_of_Domain_Evaluation/

Contains out-of-domain reasoning evaluations.

### Files

#### `Deepseek_BooIQ_Evaluation.ipynb`

Evaluates: BoolQ
Supported models: DeepSeek-Math-7B-SFT, DeepSeek-Math-7B-RL
Supported metrics: pass@1, pass@8

---

#### `HumanEval_PassK.ipynb`

Evaluates: HumanEval coding benchmark
Supported models: DeepSeek-Math-7B-SFT, DeepSeek-Math-7B-RL
Supported metrics: pass@1, pass@8

---

## notebooks/

Contains exploratory baseline notebooks.

### File
- `OLMo_Baseline.ipynb` — OLMo baseline evaluation on GSM8K

---

## CommonsenseQA_IFEval_Evaluation/

Contains commonsense reasoning (CommonsenseQA) and instruction-following (IFEval) evaluations for both OLMo2 and DeepSeek-Math models. Pre-computed results for all notebooks are saved in `results/`.

### Files

#### `OLMo2_Baseline_CommonsenseQA_IFEval.ipynb`

Evaluates the SFT baseline model (`allenai/OLMo-2-1124-7B-SFT`) on CommonsenseQA and IFEval.

Supported datasets: CommonsenseQA (full validation), IFEval (541 prompts)
Supported model: OLMo-2-1124-7B-SFT (baseline)
Supported metrics: pass@1 (CommonsenseQA accuracy), rule-based IFEval strict/loose

Results saved to: `results/olmo2_baseline_results/`

---

#### `OLMo2_GRPO_CommonsenseQA_IFEval.ipynb`

Evaluates the GRPO LoRA post-trained model (`allenai/OLMo-2-1124-7B-SFT` + `checkpoint-450`) on CommonsenseQA and IFEval.

Supported datasets: CommonsenseQA (full validation), IFEval (541 prompts)
Supported model: OLMo-2-1124-7B-SFT + GRPO LoRA (checkpoint-450)
Supported metrics: pass@1 (CommonsenseQA accuracy), rule-based IFEval strict/loose

Results saved to: `results/olmo2_grpo_results/`

---

#### `Deepseek_SFT_CommonsenseQA_Pass8.ipynb`

Evaluates DeepSeek-Math-7B-Instruct (SFT) on CommonsenseQA using pass@8.

Supported datasets: CommonsenseQA (first 200 validation examples)
Supported model: DeepSeek-Math-7B-Instruct (SFT)
Supported metrics: pass@8

Results saved to: `results/deepseek_sft_results/`

---

#### `Deepseek_RL_CommonsenseQA_Pass8.ipynb`

Evaluates DeepSeek-Math-7B-RL on CommonsenseQA using pass@8 with 4-bit quantization.

Supported datasets: CommonsenseQA (first 200 validation examples)
Supported model: DeepSeek-Math-7B-RL
Supported metrics: pass@8
Quantization: 4-bit NF4

Results saved to: `results/deepseek_rl_results/`

---

#### `Deepseek_IFEval.ipynb`

Evaluates both DeepSeek-Math-7B-SFT and DeepSeek-Math-7B-RL on IFEval using the official evaluator. Automatically saves generation checkpoints.

Supported datasets: IFEval (300 prompts)
Supported models: DeepSeek-Math-7B-SFT, DeepSeek-Math-7B-RL
Supported metrics: official IFEval strict/loose prompt-level and instruction-level accuracy

Results saved to: `results/deepseek_sft_results/` and `results/deepseek_rl_results/`

---

# How to Run Each Notebook in Colab

## 1. RL Post-Training

Open:

```text
Training/OLMo_7B_GRPO.ipynb
```

Then click:

```text
Runtime -> Select A100 GPU -> Run All
```
---

## 2. Math pass@k Evaluation

Open:

```text
Math_Evaluation/Deepseek_Pass_k_Test_Math.ipynb
```

Before running, choose: dataset, model version

Then click:

```text
Runtime -> Select A100 GPU -> Run All
```

---

## 3. BoolQ Evaluation

Open:

```text
Out_of_Domain_Evaluation/Deepseek_BooIQ_Evaluation.ipynb
```

Choose model version

Then click:

```text
Runtime -> Select A100 GPU/T4 GPU -> Run All
```

---

## 4. HumanEval pass@k Evaluation

Open:

```text
Out_of_Domain_Evaluation/HumanEval_PassK.ipynb
```

Choose model version

Then click:

```text
Runtime -> Select A100 GPU/T4 GPU -> Run All
```

---

## 5. OLMo2 Baseline CommonsenseQA + IFEval

Open:

```text
CommonsenseQA_IFEval_Evaluation/OLMo2_Baseline_CommonsenseQA_IFEval.ipynb
```

Then click:

```text
Runtime -> Select A100 GPU -> Run All
```

Pre-computed results are in `CommonsenseQA_IFEval_Evaluation/results/olmo2_baseline_results/`.

---

## 6. OLMo2 GRPO CommonsenseQA + IFEval

Open:

```text
CommonsenseQA_IFEval_Evaluation/OLMo2_GRPO_CommonsenseQA_IFEval.ipynb
```

Mount Google Drive when prompted (the LoRA checkpoint is loaded from Drive).

Then click:

```text
Runtime -> Select A100 GPU -> Run All
```

Pre-computed results are in `CommonsenseQA_IFEval_Evaluation/results/olmo2_grpo_results/`.

---

## 7. DeepSeek SFT CommonsenseQA pass@8

Open:

```text
CommonsenseQA_IFEval_Evaluation/Deepseek_SFT_CommonsenseQA_Pass8.ipynb
```

Then click:

```text
Runtime -> Select A100 GPU/T4 GPU -> Run All
```

Pre-computed results are in `CommonsenseQA_IFEval_Evaluation/results/deepseek_sft_results/`.

---

## 8. DeepSeek RL CommonsenseQA pass@8

Open:

```text
CommonsenseQA_IFEval_Evaluation/Deepseek_RL_CommonsenseQA_Pass8.ipynb
```

Then click:

```text
Runtime -> Select A100 GPU/T4 GPU -> Run All
```

Pre-computed results are in `CommonsenseQA_IFEval_Evaluation/results/deepseek_rl_results/`.

---

## 9. DeepSeek IFEval

Open:

```text
CommonsenseQA_IFEval_Evaluation/Deepseek_IFEval.ipynb
```

Choose model version (SFT or RL) before running.

Then click:

```text
Runtime -> Select A100 GPU/T4 GPU -> Run All
```

Pre-computed results are in `CommonsenseQA_IFEval_Evaluation/results/deepseek_sft_results/` and `CommonsenseQA_IFEval_Evaluation/results/deepseek_rl_results/`.

---
