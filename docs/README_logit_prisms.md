# Logit Prisms

![Logit Prisms](../assests/docs_figures/logit_diff_logit_prisms_4.png)

## Overview

Logit Prisms are the component-localization view in `LogitDiff`. They help separate what is coming from embeddings, attention, MLP updates, or the full stream, and they are useful for both prompt-side and generation-side analysis.

## When to use it

Use Logit Prisms when you want to:

- localize where a difference seems to appear
- compare attention-heavy vs. MLP-heavy effects
- decide where to intervene with patching or patchscopes
- compare prompt-time and generation-time divergence
- inspect one model on its own
- compare two models, two lenses, or two generation conditions

## Inputs

Typical inputs are:

- saved prompt capture artifacts
- saved comparison artifacts
- saved generation-lens outputs
- a chosen component view such as embedding, attention, MLP, or full residual stream

That means prism-style analysis should be read as one family with two common entry routes:

- prompt lens route
- generation lens route

and two common comparison modes:

- single-model component inspection
- pairwise component comparison

## Readout and wrapper consistency

Prism-style views depend on the same wrapper and readout surface as the rest of the toolkit. If you compare `raw`, `model_norm`, or an external tuned-lens readout, that difference belongs to the result and should be tracked explicitly.

## Prompt-side workflow

A common prompt-side path is:

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
  --norm-modes raw model_norm \
  --collect-components \
  --project-component-logits
```

and then:

```bash
PYTHONPATH=src python pipelines/compare_prompt_artifacts.py \
  --ft-artifact tmp/artifacts/<ft-run>.pt \
  --base-artifact tmp/artifacts/<base-run>.pt \
  --comparison-output tmp/artifacts/<comparison-run>.pt \
  --readout-mode model_norm \
  --metric topk_jaccard_ft_base \
  --plot-output tmp/artifacts/<comparison-run>.html
```

## Generation-side workflow

For generation-side prism-style analysis, the saved generation run is the base input:

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
  --norm-modes raw unit_norm eps_norm model_norm \
  --collect-components \
  --project-component-logits
```

or the config-driven generation route:

```bash
PYTHONPATH=src python pipelines/em_qwen/run_gen_lens.py \
  --config configs/em_qwen/gen_lens/chat_template/risky_14.json
```

The same generation config controls the prompt source, prompt format, template choice, system prompt, decoding behavior, batching, and generated length before any prism-style decomposition is read from the saved outputs.

## Shared use

In practice, prompt prisms and generation prisms are the same style of follow-up analysis over two different artifact families. The important consistency requirement is that both routes stay tied to the same wrapper-controlled tokenization, masks, and readout surface.

For prompt-side reuse, the recommended capture surface is still:

- `--force-include-input` for the input embedding view
- `--force-include-output` for the output-side L+1 view
- `--norm-modes raw model_norm` when you want both direct and final-norm readouts

## How to interpret the result

Use prism views as localization tools. If one component or one subblock carries most of the divergence, that is a strong hint about where to inspect next, where to patch, and which representation family is most relevant to the observed difference.
