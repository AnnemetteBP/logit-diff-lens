# Logit Diff Lens Implementation Spec Grounded In Current Code

## Why this document exists

The repo already contains real logic, but it is split across:

- thin re-export modules in `src/logit_diff_lens/lenses/`
- actual wrapper logic in `src/logit_diff_lens/wrappers/`
- prompt collection logic in `src/logit_diff_lens/collectors/prompt.py`
- schema fragments in `src/logit_diff_lens/schemas/`

This document maps the implementation plan onto the code that actually exists right now.

This is not a checklist only. It is intended to say:

- what each current file is
- what definitions already exist
- what to change
- what new definitions to add
- what not to touch yet

## Core Definitions And Semantics

This section defines the terms used throughout the implementation.

### Residual hidden state

When this document says `hidden state` or `h_l`, it means the residual-stream representation captured at a layer boundary.

For the current prompt collector, these are the tensors stored in the per-layer records under:

- `hidden`

The collector currently stores:

- embedding output as layer `-1`
- block outputs for each transformer layer
- optional output/final synthetic layer if requested

### Raw lens

The raw lens means:

```text
z_l = W_U h_l
```

where:

- `h_l` is the saved residual hidden state at layer `l`
- `W_U` is the output projection / LM head

No final model norm is applied.

In current code, this corresponds to the `raw` branch inside:

- `normalize_activations(...)`
- `lmhead_project(...)`

and the stored prompt-collector field:

- `logits_raw`

### ModelNorm lens

The ModelNorm lens means:

```text
z_l = W_U LN(h_l)
```

where:

- `LN` is the model’s final layer norm or RMSNorm
- `W_U` is the LM head

This is the decode mode that is closest to the paper’s baseline logit lens for pre-LN models.

In current code, this corresponds to:

- `mode == "model_norm"` in `normalize_activations(...)`
- projection through `lmhead_project(...)`

and the stored prompt-collector field:

- `logits_model_norm`

### Bias-only lens

The bias-only lens means:

```text
z_l = W_U LN(h_l + b_l)
```

where:

- `b_l` is one learned residual-space bias vector for layer `l`

This is the paper-style debiased logit lens analogue in this project.

The purpose is to test how much improvement comes from adding only a learned offset before final norm + unembed, without learning a full affine translator.

### Tuned lens

The tuned lens means:

```text
z_l = W_U LN(A_l h_l + b_l)
```

where:

- `A_l` is a learned linear translator
- `b_l` is a learned bias

This is the full tuned-lens readout.

### "With final layer" and "without final layer"

These phrases come from the tuned-lens paper and must not be confused with “with or without final norm.”

`Without final layer` means:

```text
W_U LN(h_l)
```

That is:

- do not apply the final transformer block
- do apply the final norm
- then unembed

`With final layer` means:

```text
W_U LN(h_l + F_L(h_l))
```

That is:

- apply the final transformer block residual update
- then final norm
- then unembed

So “without final layer” still includes the final norm.

### Mapping to the tuned-lens paper

The paper defines:

```text
LogitLens(h_l) = LayerNorm[h_l] W_U
```

and the extended form:

```text
LogitLens_ext(h_l) = LayerNorm[h_l + F_L(h_l)] W_U
```

So in this repo’s terminology:

- paper logit lens ≈ `modelnorm`
- paper debiased logit lens ≈ `bias_only`
- paper tuned lens ≈ `tuned`

### Prism decomposition

When this document refers to prism decomposition, the intended decomposition is into:

- embedding contribution
- attention contribution
- MLP contribution
- full residual stream

The purpose is to localize where decoded disagreement or change arises.

This is not part of milestone 1 implementation, but the canonical decode artifact must preserve enough information for prism decomposition later.

### Canonical artifact

A canonical artifact means one saved decode result that contains enough information to:

- inspect tokens
- inspect layers
- access hidden states if needed
- access `raw` logits
- access `model_norm` logits
- later derive additional readouts such as `bias_only` and `tuned`
- plot a heatmap without rerunning the model

The important point is that the artifact is not just a figure input. It is the reusable saved research object.

## Current Code Reality

### 1. `src/logit_diff_lens/lenses/raw.py`

Current contents:

```python
from ..wrappers import LogitLensWrapper
```

So this is **not** the raw lens implementation. It is only a re-export.

### 2. `src/logit_diff_lens/lenses/modelnorm.py`

Current contents:

```python
from ..wrappers import lmhead_project, normalize_activations
```

So this is **not** a full ModelNorm lens object. It is only a re-export of utility functions.

### 3. `src/logit_diff_lens/lenses/tuned.py`

Current contents:

- re-exports `TunedLens`
- defines `load_pretrained_tuned_lens(...)`

So this is only a tuned-lens loading helper right now, not a canonical decode interface.

### 4. `src/logit_diff_lens/wrappers/lens_wrappers/logit_lens_wrapper.py`

This is the main real prompt-lens wrapper implementation that currently exists.

Important existing definitions:

- `class LogitLensWrapper(BaseLensWrapper)`
- `tokenize_inputs(...)`
- `forward_pass(...)`
- `attach_hooks(...)`
- `release_hooks(...)`

Important existing state:

- `self.embedding`
- `self.lm_head`
- `self.final_norm`
- `self.blocks`
- `self.layer_registry`
- `self.component_registry`
- `self.activations`

This wrapper is the real current base for prompt-lens collection.

### 5. `src/logit_diff_lens/wrappers/wrapper_utils.py`

This file already contains the key low-level decode operations.

Important existing definitions:

- `detect_architecture(model)`
- `resolve_backbone(model, arch)`
- `find_final_norm(model)`
- `build_layer_registry(...)`
- `resolve_block_component_module(block, component)`
- `build_component_registry(blocks)`
- `normalize_activations(...)`
- `lmhead_project(...)`

These are the real decode primitives that the lens plan must be built around.

### 6. `src/logit_diff_lens/collectors/prompt.py`

This file already contains the main prompt collection logic.

Important existing definitions:

- `PromptLensActivationCollectorConfig`
- `collect_prompt_lens_activations(wrapper, config)`
- `_collect_layer_records(...)`

Most of the current prompt-lens artifact structure is being implicitly built here through per-layer records.

The current per-layer record fields already include most of the milestone-1 information needed:

- `layer_index`
- `layer_name`
- `tokens`
- `attention_mask`
- `hidden`
- `logits_raw`
- `logits_model_norm`
- optional attention/MLP outputs and logits

### 7. Existing schemas

Current schema files:

- `src/logit_diff_lens/schemas/differential.py`
- `src/logit_diff_lens/schemas/wrappers.py`

Important existing definitions:

From `wrappers.py`:

- `WrapperCapabilities`
- `WrapperModelMetadata`
- `WrapperOutputSemantics`
- `PromptForwardResult`
- `GenerationForwardResult`

From `differential.py`:

- `PairwiseModelSpec`
- `DifferentialPromptExample`
- `DifferentialGenerationMetadata`
- `LayerPositionMetricBundle`

These are useful, but they do **not** yet define a canonical saved per-lens decode artifact.

### 8. `src/logit_diff_lens/plotting/heatmaps.py`

This currently just re-exports legacy plotters from:

- `src/logit_diff_lens/_legacy/logitdiff_toolkit/...`

So there is not yet a new canonical artifact-driven heatmap implementation here.

## Actual Structural Problem

The core problem is not “missing code.”

The problem is:

1. Real logic lives in `wrappers/` and `collectors/prompt.py`
2. `lenses/` mostly contains thin re-exports, not canonical implementations
3. No canonical saved decode result schema exists yet
4. Plotting still points at legacy plotter exports instead of a new artifact-driven path

So the first milestone must be built around the code that is already real:

- `LogitLensWrapper`
- `normalize_activations`
- `lmhead_project`
- `collect_prompt_lens_activations`

and must preserve the decode semantics defined above.

## First Milestone Reframed Around Current Code

The first milestone is:

1. Keep `LogitLensWrapper` as the real collection wrapper
2. Introduce one canonical decode artifact schema
3. Make prompt collection emit that schema
4. Wrap `raw`, `modelnorm`, `tuned`, and `bias_only` around the existing wrapper/decode utilities
5. Add one non-legacy heatmap path that consumes the canonical artifact

Nothing more is required for milestone 1.

## Files To Add

### `src/logit_diff_lens/schemas/lens_outputs.py`

This file does not exist yet.

Add these dataclasses:

#### `LensRunConfig`

Required fields:

- `model_name: str`
- `lens_name: str`
- `prompt_text: str`
- `token_ids: list[int]`
- `token_text: list[str]`
- `layer_indices: list[int]`
- `metadata: dict[str, Any]`

Use existing wrapper metadata where possible:

- `WrapperModelMetadata`
- `WrapperCapabilities`

Do not duplicate those objects in full. Store them inside `metadata` if needed.

#### `LensLayerRecord`

This should formalize the dict shape currently being created in `_collect_layer_records(...)`.

Required fields:

- `layer_index: int`
- `layer_name: str`
- `hidden: torch.Tensor`
- `tokens: torch.Tensor`
- `attention_mask: torch.Tensor`

Optional fields:

- `logits_raw`
- `logits_model_norm`
- `attention_output`
- `mlp_output`
- `attention_logits_raw`
- `attention_logits_model_norm`
- `mlp_logits_raw`
- `mlp_logits_model_norm`

This is important because the collector already emits almost exactly this shape, but only as loose dicts.

The design intent is:

- `hidden` remains the reusable source representation
- `logits_raw` and `logits_model_norm` are stored eagerly because the collector already computes them
- tuned and bias-only readouts can either be stored later in separate artifacts or derived downstream from `hidden`

#### `LensDecodeResult`

Required fields:

- `config: LensRunConfig`
- `layers: list[LensLayerRecord]`

Optional fields:

- `wrapper_metadata: dict[str, Any] | None`
- `capabilities: dict[str, Any] | None`

This should be the canonical output of prompt collection in milestone 1.

### `src/logit_diff_lens/diffing/io.py`

This file does not exist yet.

Add:

- `save_lens_decode_result(output_dir, result)`
- `load_lens_decode_result(output_dir)`

Recommended save layout:

```text
<run_dir>/
  config.json
  wrapper_metadata.json
  capabilities.json
  layers.pt
```

Since `LensLayerRecord` holds tensors, the simplest batch-1 approach is:

- JSON for config and metadata
- `torch.save` for the list of layer records or a dict representation of them

### `src/logit_diff_lens/lenses/bias_only.py`

This file does not exist yet.

Add a first-pass implementation with these functions:

- `load_bias_only_parameters(path) -> dict[int, torch.Tensor]`
- `decode_bias_only_layer(...)`

The batch-1 implementation does not need a trainer yet.
It only needs to decode using:

```text
hidden_state + layer_bias
-> final norm
-> lm head
```

Use existing utilities:

- `normalize_activations(...)`
- `lmhead_project(...)`

Do not reimplement projection math.

## Files To Edit

### `src/logit_diff_lens/collectors/prompt.py`

This is the most important milestone-1 file.

Current important definitions already here:

- `PromptLensActivationCollectorConfig`
- `_collect_layer_records(...)`
- `collect_prompt_lens_activations(...)`

What to change:

1. Do not remove current collection logic
2. Change the output path so it can produce `LensDecodeResult`
3. Convert per-layer dict records into `LensLayerRecord`
4. Convert run metadata into `LensRunConfig`

Add a new public function:

```python
collect_prompt_lens_result(
    wrapper: LogitLensWrapper,
    config: PromptLensActivationCollectorConfig,
) -> LensDecodeResult
```

This should:

- call the existing collection code
- build canonical schema objects
- return `LensDecodeResult`

Do **not** break `collect_prompt_lens_activations(...)` immediately if existing code still depends on it.
Make the new function sit beside it first.

### `src/logit_diff_lens/lenses/raw.py`

Right now this is only:

```python
from ..wrappers import LogitLensWrapper
```

What it should become:

- keep the re-export if needed
- add one decode helper that uses the canonical collector result and extracts the `raw` lens path

Recommended addition:

```python
def decode_raw_from_result(result: LensDecodeResult) -> LensDecodeResult:
    ...
```

But since the collector already stores `logits_raw`, the better design for batch 1 is:

- treat `raw` as a named readout mode generated during collection
- do not create a separate second collection path

So this file should become a small canonical accessor module for the raw mode, not a wrapper rewrite.

### `src/logit_diff_lens/lenses/modelnorm.py`

Right now this only re-exports:

- `normalize_activations`
- `lmhead_project`

What it should become:

- keep those re-exports
- add a canonical readout-mode helper for `model_norm`

Again, since `collect_prompt_lens_activations(...)` already computes `logits_model_norm`, the milestone-1 path should use that directly rather than build a separate collector.

### `src/logit_diff_lens/lenses/tuned.py`

Current definitions:

- `load_pretrained_tuned_lens(...)`

What to add:

- a decode helper that applies a loaded tuned lens to saved hidden states or collected hidden states

Recommended addition:

```python
def decode_tuned_from_hidden_states(...)
```

This will need to be explicit about input shape and layer indexing.

This file is currently only a loader, so it needs actual decode-path code for milestone 1.

### `src/logit_diff_lens/lenses/__init__.py`

Add one resolver:

```python
def get_lens_mode(name: str) -> str:
    ...
```

Or better:

```python
SUPPORTED_LENS_MODES = ("raw", "model_norm", "tuned", "bias_only")
```

This file does not need a complex factory in batch 1.
It only needs one canonical place that defines supported lens mode names.

### `src/logit_diff_lens/plotting/heatmaps.py`

Current state:

- only re-exports legacy plotters

Do not delete those yet.

Add one new plotting function:

```python
def plot_prompt_heatmap_from_artifacts(...)
```

Input:

- one or more loaded `LensDecodeResult`s
- selected readout mode if relevant

Output:

- one figure using the canonical schema only

This is the first non-legacy plotting path.

## Practical Lens Design For Batch 1

The cleanest way to avoid rewriting too much is:

- keep one collection pass with `LogitLensWrapper`
- store both `raw` and `model_norm` outputs during collection
- add `tuned` as a downstream decode on saved hidden states
- add `bias_only` as a downstream decode on saved hidden states

That means:

- `raw` and `model_norm` are already naturally supported by the current collector
- `tuned` and `bias_only` should consume the stored per-layer `hidden`

This is much lower risk than trying to make four separate collector implementations immediately.

That means the milestone-1 artifact should be treated as:

- a prompt-level capture of hidden states
- plus two immediately available readout modes:
  - `raw`
  - `model_norm`

Then:

- `bias_only` is a downstream transform of `hidden`
- `tuned` is a downstream transform of `hidden`

This is the exact reason the schema must retain `hidden` explicitly.

## Canonical Milestone-1 Execution Path

The actual recommended batch-1 path is:

1. Use `LogitLensWrapper`
2. Run `collect_prompt_lens_result(...)`
3. Save the canonical result
4. For `raw` and `model_norm`, use the stored logits in the layer records
5. For `tuned` and `bias_only`, add decode helpers that consume the stored `hidden`
6. Save those as separate canonical artifacts
7. Plot all four from loaded artifacts

This uses what already exists instead of pretending the repo already has four full lens implementations.

## Files Not To Touch Yet

Do not refactor these in milestone 1 unless absolutely necessary:

- `src/logit_diff_lens/logit_lens/*`
- `src/logit_diff_lens/collectors/generation.py`
- `src/logit_diff_lens/diffing/prisms.py`
- `src/logit_diff_lens/attribution/prisms.py`
- `pipelines/*`

Milestone 1 should be confined to:

- schema
- prompt collector
- artifact I/O
- new decode helpers
- one new plotting path

## Acceptance Criteria

Milestone 1 is complete when all of the following are true:

- [ ] `collect_prompt_lens_result(...)` exists and returns `LensDecodeResult`
- [ ] `LensDecodeResult` is saveable/loadable from disk
- [ ] one saved artifact contains enough information to replot without rerunning the model
- [ ] `raw` heatmap can be produced from canonical artifacts
- [ ] `model_norm` heatmap can be produced from canonical artifacts
- [ ] `tuned` decode can be run from saved hidden states
- [ ] `bias_only` decode can be run from saved hidden states

## What The Implementer Should Understand Before Editing

Before touching code, the implementer should understand:

1. `lenses/raw.py` and `lenses/modelnorm.py` are not full implementations right now; they are re-export modules.
2. The real prompt execution path is centered on `LogitLensWrapper` and `collect_prompt_lens_activations(...)`.
3. The current collector already computes `raw` and `model_norm` logits per layer.
4. The missing piece is not model execution but canonical structure:
   - typed schema
   - artifact I/O
   - downstream readout normalization
5. `bias_only` and `tuned` should be implemented as decoders over saved `hidden`, not as a second independent prompt collector in milestone 1.

## Immediate Next Step For Implementation

The first concrete edit should be:

1. create `src/logit_diff_lens/schemas/lens_outputs.py`
2. add `collect_prompt_lens_result(...)` in `src/logit_diff_lens/collectors/prompt.py`
3. make that function wrap the existing `_collect_layer_records(...)` output into dataclasses

That should be done before touching any plotting or tuned/bias-only decode logic.
