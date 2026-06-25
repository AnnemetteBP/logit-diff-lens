# Wrappers and Readouts

![LogitDiff Overview](../assests/docs_figures/logit_diff_framework_overview_1.png)

## Overview

The wrapper layer is what lets `LogitDiff Lens` run prompt lens, generation lens, custom generation workflows, and patching workflows through one toolkit surface across different model families. This is part of `LogitDiff` itself, not a separate legacy path.

## When to use it

Use the wrappers when you want to:

- keep prompt and generation analysis under the same interface
- compare `raw`, `ModelNorm`, and `Tuned Lens` readouts
- handle padding, masks, and special tokens consistently
- reuse captured hidden states across multiple downstream analyses
- run prompt, batch, dataset, and generation-style workflows without changing the whole analysis setup

## Main wrappers

- `LogitLensWrapper` for prompt-side forward capture and prompt-lens analysis
- `GenerateLensWrapper` for standard generation-lens runs over continuations
- `CustomGenerationLensWrapper` for custom generation behavior and generation-time analysis variants
- `PatchingLensWrapper` for prompt-side and generation-focused intervention workflows

## Readout choices

The main readout choices are:

- `raw`, which decodes the hidden state directly
- `ModelNorm`, which applies the model's own final normalization before the LM head
- `Tuned Lens`, which applies a learned readout before decoding

These are method choices, not minor implementation details. If two readouts behave differently, that difference is part of the result.

## Prompt formatting and token handling

The same wrapper layer is also where `LogitDiff` keeps prompt formatting and token handling consistent.

This includes:

- plain prompts with no template
- chat-template formatting
- prefix-style formatting such as `user_assistant_prefix`
- optional `system_prompt` injection when the method uses it
- attention-mask-aware processing
- padding-aware downstream analysis
- special-token-aware tokenization and decoding
- dataset-style tokenization paths
- generation-time token growth under the same wrapper interface

## Data, batching, and masks

The wrapper layer is also what keeps runs aligned when you move from one prompt to many prompts.

This includes:

- single prompts
- batches
- dataset-style analysis
- prompt-side masking
- padding-aware filtering of meaningless token positions
- generation-time continuation handling under the same tokenizer and model surface

## Example import

```python
from logit_diff_lens.wrappers import (
    CustomGenerationLensWrapper,
    GenerateLensWrapper,
    LogitLensWrapper,
    PatchingLensWrapper,
)
```

## How to interpret the result

The wrappers are not the analysis result by themselves. Their role is to keep model loading, token handling, normalization, and hidden-state exposure stable so that heatmaps, patchscopes, prisms, and lens comparisons are based on a consistent interface.
