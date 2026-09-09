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

Generation-focused patching is also useful when you want to compare the same model under different generation conditions such as no-template prompting, chat-template prompting, custom chat-template files, or system-prompt changes.

## Prompt patchscope example

```bash
PYTHONPATH=src python pipelines/run_patchscope_prompt.py \
  --model-name <model-name> \
  --tokenizer-name <tokenizer-name> \
  --adapter-path <adapter-path> \
  --source-artifact tmp/artifacts/<source-run>.pt \
  --target-prompt "<target-prompt>" \
  --source-layer-index <source-layer> \
  --source-position <source-position> \
  --target-layer-index <target-layer> \
  --target-position <target-position> \
  --readout-mode model_norm \
  --top-k 10 \
  --dtype bfloat16 \
  --output-path tmp/artifacts/<patchscope-run>.pt
```

This prompt patchscope command uses a saved prompt-side artifact as the source representation and a target prompt string as the patched target run.

## Generation patchscope example

The generation patchscope and patch-sweep commands are exposed through the main project path, while currently bridging to the existing generation intervention implementation already present in the repo. This surface is not yet the same package-owned saved/live contract as the prompt patchscope path.

```bash
PYTHONPATH=src python pipelines/run_patchscope_generation.py \
  --base-model-id <base-model-name> \
  --comparison-model-id <comparison-model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<generation-patchscope-run>.json \
  --tokenizer-id <tokenizer-name> \
  --adapter-path <adapter-path> \
  --use-chat-template \
  --chat-template-path <template-file.jinja> \
  --system-prompt "<system-prompt>" \
  --dtype bfloat16 \
  --comparison-force-single-gpu \
  --num-generated-positions 4
```

Generation patchscope formatting behavior:

- no-template prompting is the default when `--use-chat-template` is omitted
- `--use-chat-template` enables chat-template formatting
- `--chat-template-path` lets you provide a custom template file
- `--system-prompt` adds a system message to the formatted prompt

## Generation patch sweep example

```bash
PYTHONPATH=src python pipelines/run_patchscope_generation_sweep.py \
  --base-model-id <base-model-name> \
  --comparison-model-id <comparison-model-name> \
  --prompt "<prompt-text>" \
  --tokenizer-id <tokenizer-name> \
  --adapter-path <adapter-path> \
  --use-chat-template \
  --chat-template-path <template-file.jinja> \
  --system-prompt "<system-prompt>" \
  --dtype bfloat16 \
  --output-path tmp/artifacts/<generation-patch-sweep>.json
```

## Shared interpretation

Prompt patchscope and generation patchscope are the same broader LogitDiff intervention family:

- prompt patchscope asks what changes in a fixed prompt run
- generation patchscope asks what changes in the continuation trajectory

Both are meant to follow up on saved prompt or generation analyses rather than standing apart from the lens workflows.

## Direct parameters vs. upstream capture requirements

The public patchscope entrypoints are not identical to the capture CLIs.

Direct parameters on the patchscope commands include:

- prompt patchscope currently uses a saved source artifact plus a target prompt
- generation patchscope and generation patch sweep currently expose `--base-model-id`, `--comparison-model-id` or `--adapter-path`, no-template prompting, `--use-chat-template`, `--chat-template-path`, `--system-prompt`, and generated-position controls through the legacy-backed generation intervention path
- prompt patchscope directly exposes `--readout-mode`

Upstream capture requirements matter whenever the patched readout should include input-side or output-side logits:

- prompt-side `--force-include-input` matters when the source artifact must include the embedding readout
- prompt-side `--force-include-output` matters when the source artifact must include the output-side L+1 readout
- prompt-side `--norm-modes raw model_norm` matters when downstream patchscope interpretation depends on both raw and final-norm projections
- generation-side runs should likewise include the needed input or output rows before patchscope-style follow-up is interpreted as a logit-level result

## How to interpret the result

Read the patched result as an intervention test. If the target prediction or continuation shifts in a meaningful way, the inserted representation is likely carrying information that matters at that location. In generation-focused analysis, the main question is often not only whether the next token changes, but whether the later continuation pattern changes as well.
