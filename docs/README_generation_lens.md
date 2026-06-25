# Generation Lens

![LogitDiff Overview](../assests/docs_figures/logit_diff_framework_overview_1.png)

## Overview

This guide covers the generation workflow inside `LogitDiff Lens`. It is the generation-time logit lens path for following model behavior during actual continuation instead of only on the fixed prompt. This is useful when you want to see how divergence unfolds step by step as text is generated.

## Wrappers and normalization

The generation path depends on wrappers for consistent generation behavior, token handling, and readout choices across models.

The main readout choices users should care about are:

- `raw`
- `ModelNorm`
- `Tuned Lens`

If you compare generation results across readout modes, treat that as part of the analysis rather than a minor detail.

## When to use it

Use the generation lens when you want to:

- compare two models during continuation rather than only on the prompt
- compare the same model under different decoding conditions
- compare the same model under different prompt-format or template conditions
- track how divergence changes over generated steps
- study generation-level Jaccard overlap or other token-distribution comparisons
- compare prompt behavior with realized continuation behavior

Typical same-model comparisons include:

- different chat templates
- chat template versus no template
- plain prompt versus prefix-style prompting
- different temperatures
- different sampling strategies
- other decoding or prompting conditions

## Templates and prompt formatting

The existing generation-lens config path already supports template and prompt-format control.

The main user-facing settings are:

- `template_name`
- `prompt_format`
- `use_chat_template`
- `system_prompt`

Common setups include:

- `prompt_format: "plain"` with `use_chat_template: false` for no-template generation
- `prompt_format: "chat_template"` with `use_chat_template: true` for chat-formatted generation
- prefix-style prompting when the prompt source or wrapper setup uses a prefixed format

This means a generation-lens comparison does not need to be only model A versus model B. It can also be:

- the same model with one template versus another template
- the same model with chat formatting versus no template
- the same model with different system prompts
- the same model with different decoding settings

## What you give it

- a generation-lens config file
- a base model
- a comparison model, adapter, or alternate condition
- a prompt source such as a dataset or prompt list
- generation settings such as token budget, top-k, prompt formatting, tokenization behavior, and masking-related settings

## What it gives back

It produces saved generation-lens outputs that can be summarized across steps, layers, prompts, and comparison metrics, while preserving the prompt formatting and tokenization conditions used for the run.

## Example command

```bash
PYTHONPATH=src python pipelines/<pipeline-group>/run_gen_lens.py \
  --config configs/<group>/gen_lens/<config-name>.json
```

Example config families already in the repository include:

- `configs/em_qwen/gen_lens/chat_template/...`
- `configs/em_qwen/gen_lens/no_template/...`
- `configs/quant_llama/gen_lens/...`

The same generation-lens workflow can be followed up with generation heatmaps, generation patchscopes, and generation patch sweeps from the main project pipelines.

## How to interpret the result

Read generation-lens output as a trajectory view. Instead of only asking what the models predict before generation starts, ask where their continuations begin to separate, how stable that difference is over time, and whether top-token overlap stays high or falls apart during continuation. This applies both to two-model comparisons and to same-model comparisons under different generation conditions.

Generation-lens follow-up analyses also include generation-focused patching, patch sweeps, and heatmap workflows. For explicit prompt-heatmap and generation-heatmap plotting paths, see [README_heatmaps.md](/media/am/AM/logit-diff-lens/docs/README_heatmaps.md). For wrapper details, see [README_wrappers.md](/media/am/AM/logit-diff-lens/docs/README_wrappers.md). For intervention workflows, see [README_patchscopes.md](/media/am/AM/logit-diff-lens/docs/README_patchscopes.md).
