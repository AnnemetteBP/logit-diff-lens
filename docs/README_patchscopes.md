# Patchscopes

![Patchscope Workflow](../assests/docs_figures/logit_diff_patchscope_5.png)

## Overview

Patchscopes let you take a representation from one prompt or system and insert it into another run. In `LogitDiff`, this is useful for following up on an interesting divergence and testing whether a specific layer-position representation changes the target readout.

## When to use it

Use patchscopes when you want to:

- test whether one token representation drives a change downstream
- follow up on a strong difference from a heatmap
- compare how two systems react to the same inserted representation
- compare one model under two generation conditions
- explore causal intervention rather than just observation
- follow up on batched or dataset-level findings with targeted single examples
- run generation-oriented intervention analysis, not only prompt-only patching

## What you give it

- a saved source artifact
- a target prompt or target run
- a source layer and token position
- a target layer and token position
- an output path

For the command placeholders in this guide:

- `<base-model-name>`, `<comparison-model-name>`, `<model-name>` are strings
- `<prompt-text>`, `<target-prompt>`, `<system-prompt>` are strings
- `<source-layer>` and `<target-layer>` are integer layer indices
- `<source-position>` and `<target-position>` are integer token positions

The source side can come from either:

- a prompt-side saved artifact
- a generation-side saved artifact or generation run output

## What it gives back

It saves a patched run showing how the target prompt or target continuation behaves after the chosen representation is inserted.

## Prompt lens and generation lens

Patchscopes are not limited to prompt-only analysis.

They are useful in both:

- prompt-lens analysis, where you patch within or across fixed prompt runs
- generation-lens analysis, where you patch into continuations and study how later generated behavior changes

That includes both:

- single-model intervention studies
- pairwise studies across two models, two lenses, or two generation conditions

This includes the kind of generation-focused patchscope analysis used in paper-style case studies, where a chosen high-difference token or layer is patched across positions to see how the continuation changes.

Generation-focused patching is also useful when you want to compare the same model under different generation conditions such as template choice, prefix formatting, or system-prompt changes.

## Example command

```bash
PYTHONPATH=src python pipelines/run_patchscope_prompt.py \
  --model-name <model-name> \
  --source-artifact tmp/artifacts/<source-run>.pt \
  --target-prompt "<target-prompt>" \
  --source-layer-index <source-layer> \
  --source-position <source-position> \
  --target-layer-index <target-layer> \
  --target-position <target-position> \
  --readout-mode model_norm \
  --output-path tmp/artifacts/<patchscope-run>.pt
```

## Generation patchscope example

```bash
PYTHONPATH=src python pipelines/run_patchscope_generation.py \
  --base-model-id <base-model-name> \
  --comparison-model-id <comparison-model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<generation-patchscope-run>.json \
  --use-chat-template \
  --system-prompt "<system-prompt>" \
  --num-generated-positions 4
```

## Generation patch sweep example

```bash
PYTHONPATH=src python pipelines/run_patchscope_generation_sweep.py \
  --base-model-id <base-model-name> \
  --comparison-model-id <comparison-model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<generation-patch-sweep>.json
```

## Shared interpretation

Prompt patchscope and generation patchscope are the same broader LogitDiff intervention family:

- prompt patchscope asks what changes in a fixed prompt run
- generation patchscope asks what changes in the continuation trajectory

Both are meant to follow up on saved prompt or generation analyses rather than standing apart from the lens workflows.

That means the useful upstream controls are the same wrapper-controlled settings used elsewhere in the toolkit:

- `prompt` or `dataset` source
- prompt formatting and optional system prompt
- truncation, max length, and padding
- `force_include_input`
- `force_include_output`
- chosen readout or norm mode

## How to interpret the result

Read the patched result as an intervention test. If the target prediction or continuation shifts in a meaningful way, the inserted representation is likely carrying information that matters at that location. In generation-focused analysis, the main question is often not only whether the next token changes, but whether the later continuation pattern changes as well.
