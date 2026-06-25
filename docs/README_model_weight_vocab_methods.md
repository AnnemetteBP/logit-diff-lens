# Weight and Vocabulary Methods

![Weight / Vocab Methods](../assests/docs_figures/logit_diff_weight_vocab_7.png)

## Overview

This part of `LogitDiff` is for vocabulary-space and weight-space interpretation. It is the right place for analyses such as projecting hidden-state differences into the vocabulary, studying weight directions, and following low-rank structure such as SVD.

## When to use it

Use these methods when you want to:

- project a hidden-state direction into token space
- inspect weight directions or LM-head directions
- study low-rank structure in a comparison result
- summarize large hidden-state or logit differences with token-level clues
- inspect a single model or lens readout on its own
- compare two models, two lenses, or two generation conditions

## Typical inputs

- saved prompt comparison artifacts
- saved prompt captures
- saved generation-lens outputs
- weight matrices or learned directions

So this method family should also be read in two parallel ways:

- prompt-lens follow-up over saved prompt artifacts or comparisons
- generation-lens follow-up over saved generation outputs

and in two result styles:

- single-run interpretation
- pairwise difference interpretation

## Prompt-side starting point

Prompt-side direct capture:

```bash
PYTHONPATH=src python pipelines/capture_prompt_artifacts.py \
  --model-name <model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<prompt-run>.pt \
  --dtype bfloat16 \
  --truncate \
  --max-length 512 \
  --padding longest \
  --force-include-input \
  --force-include-output \
  --norm-modes raw model_norm
```

or prompt-side comparison:

```bash
PYTHONPATH=src python pipelines/compare_prompt_artifacts.py \
  --ft-artifact tmp/artifacts/<ft-run>.pt \
  --base-artifact tmp/artifacts/<base-run>.pt \
  --comparison-output tmp/artifacts/<comparison-run>.pt \
  --readout-mode model_norm \
  --metric topk_jaccard_ft_base \
  --plot-output tmp/artifacts/<comparison-run>.html
```

## Generation-side starting point

Generation-side direct capture:

```bash
PYTHONPATH=src python pipelines/capture_generation_artifacts.py \
  --model-name <model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<generation-run>.pt \
  --dtype bfloat16 \
  --prompt-format plain \
  --truncate \
  --max-length 512 \
  --padding longest \
  --max-new-tokens 32 \
  --batch-size 8 \
  --force-include-input \
  --force-include-output \
  --norm-modes raw unit_norm eps_norm model_norm
```

or the config-driven generation route:

```bash
PYTHONPATH=src python pipelines/em_qwen/run_gen_lens.py \
  --config configs/em_qwen/gen_lens/chat_template/risky_14.json
```

The generation config is where the generation-side prompt source, prompt formatting, template choice, decoding settings, and saved outputs are defined before any vocabulary-space follow-up analysis is applied.

## Shared capture guidance

For prompt-side reuse, the usual reusable capture surface is:

- `--force-include-input` when the embedding-side view matters
- `--force-include-output` when the output-side L+1 view matters
- `--norm-modes raw model_norm` when you want both direct and final-norm LM-head projections

For generation-side reuse, the parallel controls are:

- `--prompt` or `--dataset-path`
- `--truncate`
- `--max-length`
- `--padding`
- `--max-new-tokens`
- `--batch-size`
- `--force-include-input`
- `--force-include-output`
- `--norm-modes`

## Shared interpretation

Weight-space and vocabulary-space analysis are not prompt-only methods. They should be applied consistently to both prompt and generation artifacts wherever the saved outputs contain the needed hidden states, logits, or derived differences.

## How to interpret the result

These methods are best used as semantic follow-up tools. A projected token list or a dominant singular direction is a clue about what a difference may mean, not a substitute for the rest of the prompt, generation, patching, and prism analyses.
