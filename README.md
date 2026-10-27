# Anytime Width: Progressive Neuron Activation in Dense Language Models

**Course project — Fall 2026**

## Problem Statement

A dense transformer uses every neuron in every layer for every token, regardless of how hard that token is to predict. Prior work on contextual sparsity (Deja Vu, PowerInfer) shows that many neurons contribute little for a given input, but those methods fix a compute budget in advance. We ask a different question: **if a model is allowed to activate neurons progressively, starting with the most important ones and adding more until the layer's output stabilizes, how many does it actually need, and does that number tell us anything about the token?**

Concretely, this project investigates:

1. **How much width does a token need?** The distribution of neurons used per token and per layer before the output converges, on current open models.
2. **Does difficulty show up as width?** Whether hard tokens (reasoning steps, rare entities, first tokens of an answer) require more neurons than easy ones, across task types.
3. **Is width a quality signal?** Whether tokens that fail to converge, or converge late, are the ones where the model is wrong, and how this compares to entropy and trained probes as an error/hallucination predictor.
4. **Does adaptive beat fixed?** Whether per-token adaptive width matches dense-model quality better than fixed-budget sparsity at the same average compute.

## Proposed Method

**Model.** A small open dense LLM (primary: Llama 3.2 1B or 3B; secondary: Gemma 2 2B). If time permits, Llama 3.1 8B to check whether findings hold across scale.

**Neuron ordering.** For each MLP layer, rank neurons by importance using activation statistics on a calibration set (mean absolute activation, with learned-score and random orderings as comparisons).

**Progressive activation.** An MLP block's output is a sum over neurons, so adding neurons is additive: we compute the layer output with the top-k neurons, then add the next chunk without recomputing the first. Each layer widens in chunks until the change in its output (relative norm of the delta) falls below a threshold, then passes its result upward. This is a *local* stopping rule: no global recompute of later layers.

**Measurements.** For every generated token we record per-layer neurons used, convergence step, and the final output distribution. We compare the adaptive model's outputs to the dense model's on standard benchmarks.

**Analysis (the "shape" study).** Per-token width is treated as an activation-shape statistic. We test it as a predictor of (a) token/task difficulty and (b) model errors, against baselines (output entropy, a linear probe on hidden states).

**Baselines.** Dense model; fixed-budget sparsity at matched average neuron count; random ordering; standard early-exit as a depth-wise comparison.

**Demo.** A live dashboard: type a prompt, watch per-layer width bars fill as each token is generated, with the predicted-uncertainty signal, the answer, and a throughput counter alongside. Optional: sparse-autoencoder feature view via Neuronpedia.

## Data Sources

- **Calibration:** a sample of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) or WikiText-103 for neuron ranking.
- **Evaluation benchmarks:** [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (math reasoning), [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa) (factual recall), [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) (code), and a general-language perplexity set (WikiText). Difficulty labels come from benchmark metadata and from model correctness.
- **Models:** Llama 3.2 1B/3B, Gemma 2 2B (Hugging Face). All open-weights.
- **Tooling:** PyTorch, Hugging Face Transformers, TransformerLens / nnsight for activation access, Neuronpedia for SAE features (demo only).

## Hardware

Single NVIDIA RTX 5090 (32 GB). All models and experiments are sized to run locally.

## Evaluation Plan

- Width distributions per model, layer, and task.
- Correlation between width and difficulty/correctness; AUROC for error prediction vs. entropy and probe baselines.
- Quality (accuracy / perplexity) vs. average width, adaptive vs. fixed-budget, with multiple seeds where randomness applies.
- Measured wall-clock throughput in PyTorch vs. theoretical FLOP savings, reported honestly.

## Timeline (5 weeks)

| Week | Milestone |
|------|-----------|
| 1 | Baseline model running; activation hooks; neuron ordering; first width measurements |
| 2 | Progressive activation with local stopping; dense-vs-adaptive quality comparison |
| 3 | Full benchmark runs; difficulty and error-prediction analysis |
| 4 | Second model / scale check; dashboard demo; begin report |
| 5 | Final report, recorded demo video, repo cleanup |

## Repository Structure (planned)

```
src/          progressive activation, ordering, hooks
experiments/  sweep configs and scripts
analysis/     notebooks and plots
demo/         live dashboard
report/       final report (PDF)
```

## How to Run

_To be filled in as the project develops._

## Citations & Acknowledgements

This project builds on prior work in contextual sparsity, adaptive computation, and interpretability, including Deja Vu (Liu et al., 2023), PowerInfer (Song et al., 2023), MatFormer (Devvrit et al., 2023), Mixture-of-Depths (Raposo et al., 2024), LayerSkip (Elhoushi et al., 2024), and sparse-autoencoder work from Anthropic and others. Open-source libraries and models (PyTorch, Hugging Face Transformers, TransformerLens, nnsight, Llama 3.2, Gemma 2) are used under their respective licenses. Full citations will be maintained in `report/` and updated here as the project progresses.
