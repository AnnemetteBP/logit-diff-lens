# Reproducibility Pipeline Spec

![Forward Capture Artifact](../assests/docs_figures/logit_diff_forward_artifact_2.png)

## Purpose

This document defines the analysis objects, metrics, data flow, and artifact boundaries used by `logit-diff-lens`. It is intended to serve as the canonical reference for:

- reproducible experiments
- method definitions
- figure provenance
- implementation conventions

## Scope

This spec covers:

- prompt-lens pipelines
- generation-lens compatibility requirements
- hidden-state comparisons
- readout/lens comparisons
- patchscope interventions
- prism decomposition outputs
- SVD analysis outputs
- artifact structure

This spec does not require deleting or rewriting the generation path.

## Core distinction: hidden space vs readout space

All analyses fall into one of two spaces.

### Hidden-space branch

This branch compares internal representations before any decode/readout transformation.

For model `m`, layer `l`, token position `t`, let:

```text
h_l,t^(m) ∈ R^d
```

be the residual hidden state at that layer/token.

For two models `A` and `B`, the hidden difference is:

```text
Δh_l,t = h_l,t^(A) - h_l,t^(B)
```

Typical hidden-space analyses include:

- hidden L2 distance
- hidden cosine similarity
- hidden SVD
- hidden prism decomposition
- representational drift analyses

### Readout-space branch

This branch compares decoded outputs after a chosen lens/readout transformation.

Let `R_l^(m)` denote a lens-specific readout at layer `l` for model `m`. Then:

```text
z_l,t^(m) = R_l^(m)(h_l,t^(m)) ∈ R^V
```

where `V` is the vocabulary size.

Probability distributions are:

```text
p_l,t^(m) = softmax(z_l,t^(m))
```

Typical readout-space analyses include:

- logit differences
- probability differences
- KL/JSD
- top-k overlap / Jaccard
- target-token rank
- target-token probability
- readout-space SVD

## Differential analysis families

Two differential families are used throughout the project.

### Family A: decoded model-vs-model diff

This is the standard `logitdiff` family used for decoded model comparisons.

For two models `A` and `B`, choose a lens/readout `R` and compare:

```text
z_l,t^(A,R) = R_l^(A)(h_l,t^(A))
z_l,t^(B,R) = R_l^(B)(h_l,t^(B))
```

Then define:

```text
Δz_l,t^(A,B,R) = z_l,t^(A,R) - z_l,t^(B,R)
```

and likewise:

```text
p_l,t^(A,R) = softmax(z_l,t^(A,R))
p_l,t^(B,R) = softmax(z_l,t^(B,R))
```

This family includes metrics such as:

- JSD between `p^(A,R)` and `p^(B,R)`
- KL between `p^(A,R)` and `p^(B,R)`
- top-k Jaccard overlap
- target-token probability difference
- target-token rank difference

belong.

The object of comparison is the decoded output of model `A` versus the decoded output of model `B`.

### Family B: hidden-delta-first analysis

This is a separate family based on residual-space deltas.

First compute the residual-space delta directly:

```text
Δh_l,t^(A,B) = h_l,t^(A) - h_l,t^(B)
```

The residual delta is itself the primary object of analysis.

From this branch, there are two subcases.

#### B1. analyze the hidden delta directly

In this case the main object is `Δh_l,t^(A,B)` itself.

Examples:

- hidden delta norm
- hidden delta cosine against another direction
- hidden-space SVD on stacked `Δh`
- prism decomposition of `Δh`

#### B2. decode the hidden delta

In some analyses, the hidden delta is later pushed through a decode operator:

```text
z_l,t^(Δ,R) = R_l(Δh_l,t^(A,B))
```

For example:

```text
z_l,t^(Δ,modelnorm) = W_U LN(Δh_l,t^(A,B))
```

or, if the analysis explicitly wants a raw linear readout:

```text
z_l,t^(Δ,raw) = W_U Δh_l,t^(A,B)
```

This differs from:

```text
R_l(h_l,t^(A)) - R_l(h_l,t^(B))
```

unless the readout is strictly linear over the exact same space and does not include a nonlinear normalization step. In particular, for ModelNorm:

```text
W_U LN(h_A - h_B) != W_U LN(h_A) - W_U LN(h_B)
```

in general.

This distinction is mathematically important and should be preserved in both code and documentation.

### Naming convention

Artifact and function names should encode which family they belong to:

- `model_vs_model_*` for decoded `A vs B` comparisons
- `hidden_delta_*` for direct `Δh` analyses
- `decoded_hidden_delta_*` for analyses that first form `Δh` and then decode it

This is especially important for any ADL-style path, which belongs to Family B rather than standard decoded `logitdiff`.

## Patchscope family

Patchscopes are a separate intervention family built on top of saved forward artifacts.

They are not reducible to plain decoded comparison metrics, because they operate by inserting a selected representation into a target run and then observing the resulting decoded output.

### Core patchscope object

Let:

```text
h_l,t^(A)
```

be a saved source representation from system `A`.

A patchscope selects:

- a source system or source artifact
- a source layer `l_s`
- a source token position `t_s`
- a target system or target run
- a target layer `l_t`
- a target token position `t_t`
- a mapping `f`

and inserts:

```text
f(h_l_s,t_s^(A))
```

into the target run at `(l_t, t_t)`.

The current implemented primitive uses:

```text
f = identity
```

### Current artifactized form

The current first-class implementation is prompt-first and artifact-first:

1. load a saved `PromptDecodeArtifact`
2. select a source hidden state from that artifact
3. patch it into a target prompt run
4. decode the patched target run
5. save a `PatchscopePromptArtifact`

This preserves provenance and allows the patchscope result to be tied back to the same hidden-state capture used for comparison heatmaps, Logit Prisms, and hidden-delta analyses.

### Patchscope output space

Patchscope outputs belong to readout space after intervention.

For a chosen readout `R`, the patched decoded output is:

```text
z_t^(patched,R)
```

with probabilities:

```text
p_t^(patched,R) = softmax(z_t^(patched,R))
```

Downstream patchscope summaries may compare these outputs against:

- an unpatched target run
- a baseline source/target pairing
- another patchscope condition

### Recommended future sweep artifacts

Patchscope analyses should eventually support structured sweeps over:

- source positions
- target positions
- source layers
- target layers
- chosen token subsets

These should be stored in dedicated sweep artifacts rather than as loose script outputs.

## Canonical operand ordering and sign convention

All signed differentials use one canonical ordering:

```text
comparison - reference
```

For the main finetuning setting, this becomes:

```text
ft - base
```

This convention applies consistently to:

- saved tensors
- schema field names
- metric computation
- plot subtitles
- colorbar interpretation
- paper equations
- figure captions

### Default model roles

Unless stated otherwise:

- `comparison model = ft`
- `reference model = base`

The default interpretation of a positive signed quantity is:

```text
more present in ft than in base
```

The default interpretation of a negative signed quantity is:

```text
more present in base than in ft
```

### Canonical definitions

#### Hidden-state delta

```text
Δh_l,t = h_l,t^(ft) - h_l,t^(base)
```

#### Readout/logit delta

For a chosen readout `R`:

```text
Δz_l,t^(R) = z_l,t^(ft,R) - z_l,t^(base,R)
```

where:

```text
z_l,t^(ft,R) = R_l^(ft)(h_l,t^(ft))
z_l,t^(base,R) = R_l^(base)(h_l,t^(base))
```

#### Probability delta

```text
Δp_l,t^(R) = p_l,t^(ft,R) - p_l,t^(base,R)
```

#### Prism component delta

For any prism component `c` such as embedding, attention, MLP, or full stream:

```text
Δc_l,t = c_l,t^(ft) - c_l,t^(base)
```

This means that the standard prism-diff analysis is:

```text
prism(ft) - prism(base)
```

not the reverse.

#### Decoded hidden-delta object

If the analysis first forms a hidden delta and then decodes it, the hidden delta must still be formed as:

```text
Δh_l,t = h_l,t^(ft) - h_l,t^(base)
```

and then decoded:

```text
z_l,t^(Δ,R) = R_l(Δh_l,t)
```

### Naming convention

Names should encode operand order to prevent silent sign flips.

Preferred names:

- `ft_minus_base_hidden`
- `ft_minus_base_logits_modelnorm`
- `ft_minus_base_probs_modelnorm`
- `ft_minus_base_prism_attention`

Avoid ambiguous names such as:

- `diff`
- `delta_logits`
- `comparison`

unless the field is nested under an object that already fixes the order as `ft_minus_base`.

### Asymmetric metrics

Some metrics are directional and should not be described only as "diff".

#### KL divergence

KL must always be written with explicit argument order:

```text
KL(p_ft || p_base)
```

or

```text
KL(p_base || p_ft)
```

These are different quantities and should be stored under different names.

Preferred field names:

- `kl_ft_to_base`
- `kl_base_to_ft`

KL should not be stored under a generic field such as `kl_diff`.

#### JSD

JSD is symmetric, but inputs should still be passed and documented in canonical order:

```text
JSD(p_ft, p_base)
```

Preferred field name:

- `jsd_ft_base`

### Rank-based metrics

Token rank requires a separate definition because lower rank is better.

Let:

```text
r_ft = rank of target token under ft
r_base = rank of target token under base
```

Then define the raw rank delta as:

```text
Δrank_raw = r_ft - r_base
```

Interpretation:

- negative = target token moved to a better rank in ft
- positive = target token moved to a worse rank in ft

If the analysis wants an "improvement" quantity where positive means better in ft, define a separate metric:

```text
rank_improvement_ft_over_base = r_base - r_ft
```

These two quantities should not share the same field name.

### Similarity and overlap metrics

For metrics like:

- top-k Jaccard
- overlap count
- intersection-over-union

the value itself is symmetric, but operand labels must still be attached in canonical order:

- compare `ft` against `base`
- save under names like `topk_jaccard_ft_base`

### Plotting convention

Every signed heatmap should state the sign convention in the metadata or subtitle.

The default interpretation must be:

```text
positive = ft > base
negative = base > ft
```

This is especially important for:

- prism diff plots
- decoded hidden-delta plots
- logit-diff heatmaps
- probability-diff heatmaps

## Canonical lens definitions

For all definitions below:

- `h_l` is the residual hidden state at layer `l`
- `LN` is the model’s final layer norm or RMSNorm
- `W_U` is the LM head / output projection

### Raw lens

```text
z_l = W_U h_l
```

Meaning:

- decode directly from the residual stream
- no final norm

### ModelNorm lens

```text
z_l = W_U LN(h_l)
```

Meaning:

- apply the model’s final norm to each intermediate residual hidden state
- then decode with the LM head

This is the baseline lens most directly aligned with the tuned-lens paper’s pre-LN logit lens.

### Bias-only ModelNorm lens

```text
z_l = W_U LN(h_l + b_l)
```

where:

- `b_l ∈ R^d` is one learned residual-space bias vector per layer

Meaning:

- same decode path as ModelNorm
- but with a learned per-layer offset before the final norm

### Tuned lens

```text
z_l = W_U LN(A_l h_l + b_l)
```

where:

- `A_l ∈ R^(d×d)` is a learned per-layer linear translator
- `b_l ∈ R^d` is a learned per-layer bias

## Relation to the tuned-lens paper

For pre-LN models, the paper defines the logit lens as:

```text
LogitLens(h_l) = LayerNorm[h_l] W_U
```

and the extended form:

```text
LogitLens_ext(h_l) = LayerNorm[h_l + F_L(h_l)] W_U
```

where `F_L` is the final transformer block residual update.

So:

- paper logit lens ≈ ModelNorm lens
- paper debiased logit lens ≈ Bias-only ModelNorm lens
- paper tuned lens ≈ Tuned lens

### With final layer vs without final layer

These phrases do **not** refer to whether final norm is applied.

`Without final layer` means:

```text
W_U LN(h_l)
```

`With final layer` means:

```text
W_U LN(h_l + F_L(h_l))
```

So “without final layer” still includes the final norm.

## What the prompt collector must save

A canonical prompt artifact must save enough information to support:

- raw lens analyses
- ModelNorm analyses
- downstream Bias-only decode
- downstream Tuned decode
- hidden diff analyses
- prism decomposition later
- plotting without rerunning the model

Therefore the canonical per-layer record must include:

- `layer_index`
- `layer_name`
- `tokens`
- `token_text`
- `attention_mask`
- `hidden`
- `logits_raw`
- `logits_model_norm`

Optional but desirable in the same record:

- `attention_output`
- `mlp_output`
- `attention_logits_raw`
- `attention_logits_model_norm`
- `mlp_logits_raw`
- `mlp_logits_model_norm`

## Formal metric definitions

Metrics must be attached to either hidden-space or readout-space explicitly.

### Hidden-space metrics

#### Hidden L2 distance

For each layer/token:

```text
L2_l,t = || h_l,t^(A) - h_l,t^(B) ||_2
```

#### Hidden cosine similarity

```text
cos_l,t = (h_l,t^(A) · h_l,t^(B)) / (||h_l,t^(A)|| ||h_l,t^(B)||)
```

#### Hidden mean diff norm

For a set of samples `S`:

```text
|| E_(x,t∈S)[Δh_l,t] ||_2
```

#### Hidden delta covariance / SVD input

When hidden-delta SVD is performed, the matrix to factorize must be defined explicitly.

For a selected set of positions:

```text
X_hidden_delta ∈ R^(N × d)
```

where each row is one flattened residual-space delta:

```text
X_hidden_delta[i, :] = Δh_l,t^(A,B)
```

for a particular sample-position-layer index included in the analysis.

Then:

```text
X_hidden_delta = U Σ V^T
```

and the right singular vectors `V` live in hidden space.

### Readout-space metrics

#### Logit difference

```text
Δz_l,t = z_l,t^(A) - z_l,t^(B)
```

#### Probability difference

```text
Δp_l,t = p_l,t^(A) - p_l,t^(B)
```

#### Decoded hidden-delta logits

If the analysis decodes the hidden delta itself, define:

```text
z_l,t^(Δ,R) = R_l(Δh_l,t^(A,B))
```

and keep it separate from:

```text
Δz_l,t^(A,B,R) = R_l(h_l,t^(A)) - R_l(h_l,t^(B))
```

These are different metrics objects and must never share the same field name.

#### Same-model KL-to-final

Used to measure lens faithfulness to the model’s own final logits.

For model `m`:

```text
KL_to_final_l,t^(m) = KL( p_final,t^(m) || p_l,t^(m) )
```

Recommended dataset-level aggregation:

- mean over token positions
- then mean over samples

#### Cross-model JSD

Used to compare decoded distributions between models or between lens modes.

```text
JSD_l,t(A, B) = JSD( p_l,t^(A), p_l,t^(B) )
```

Recommended dataset-level aggregation:

- mean over token positions
- then mean over samples

#### Top-k Jaccard

Let `TopK_l,t^(m)` be the set of top-k predicted tokens.

```text
Jaccard_l,t = |TopK_l,t^(A) ∩ TopK_l,t^(B)| / |TopK_l,t^(A) ∪ TopK_l,t^(B)|
```

#### Target-token rank

For reference token `y_t`:

```text
rank_l,t^(m) = rank of y_t under z_l,t^(m)
```

#### Target-token probability

```text
p_l,t^(m)(y_t)
```

#### Correctness

```text
correct_l,t^(m) = 1[argmax z_l,t^(m) = y_t]
```

### Bias metric from tuned-lens framing

For vocabulary item `v`, define mean marginal probability over the dataset:

```text
q_l(v) = E_(x,t)[ q_l(v | x_<t) ]
p(v)   = E_(x,t)[ p(v | x_<t) ]
```

Bias can be measured by:

```text
KL_bias_l = KL( p(v) || q_l(v) )
```

This is optional for milestone 1, but should be retained in the spec because it may be used later.

## Prism decomposition definition

Prisms should be computed as decomposed contribution objects at either hidden-space or readout-space, but the branch must be explicit.

The intended decomposition categories are:

- embedding
- attention
- MLP
- full residual stream

### Hidden prism output

For each model, layer, token:

- `embedding_hidden`
- `attention_hidden`
- `mlp_hidden`
- `full_hidden`

These are compared before decode.

### Readout prism output

For each model, layer, token, and chosen lens mode:

- `embedding_logits`
- `attention_logits`
- `mlp_logits`
- `full_stream_logits`

These are compared after decode.

### Why both matter

This separation allows the paper to ask two different questions:

1. where does the representational difference live?
2. where does the decoded prediction difference appear?

Those are not the same question.

## SVD definition

SVD must be defined over an explicit matrix object. The matrix choice must be stored in the artifact metadata.

### Hidden-space SVD

For one layer `l`, stack hidden diffs over sample/token positions:

```text
M_hidden^(l) ∈ R^(N × d)
```

where each row is:

```text
Δh_l,t
```

Then compute:

```text
M_hidden^(l) = U Σ V^T
```

Use cases:

- dominant hidden difference directions
- representational drift structure

### Readout-space SVD

For one layer `l`, stack logit diffs over sample/token positions:

```text
M_logit^(l) ∈ R^(N × V)
```

where each row is:

```text
Δz_l,t
```

Then compute:

```text
M_logit^(l) = U Σ V^T
```

Use cases:

- dominant vocabulary-space difference directions
- quantization case-study interpretation

### Prism-specific SVD

Same as above, but use one subblock-only diff matrix, e.g.:

- attention-only logit diffs
- MLP-only logit diffs
- full-stream logit diffs

### Vocabulary projection

For readout-space SVD, the right singular vectors correspond to vocabulary-space directions.

For each selected right singular vector `v_i`:

- compute top positive tokens
- compute top negative tokens

These token lists are the vocabulary projection summary.

## Prompt vs generation scope

Milestone 1 focuses on prompt-lens stabilization.

This does **not** mean the generation path should be deleted or ignored.

### Prompt path requirements

Prompt artifacts must be canonical first because:

- they are simpler
- they support the main lens comparison logic
- they are enough to stabilize the artifact schema

### Generation path constraint

The generation path must remain in the repo and must not be deleted merely because it is not milestone 1.

The canonical artifact design should be chosen so generation artifacts can later fit into the same family.

Minimum compatibility requirement:

- prompt artifact schema decisions must not make generation integration impossible later

## Runtime backend contract

The runtime backend must be treated as part of the experimental configuration, not as an invisible implementation detail.

### Backend dimensions

Each artifact must record at least the following backend dimensions:

- model loading backend
- activation collection backend
- decode backend
- device placement policy
- numeric precision
- quantization mode

### Canonical backend fields

Each run config or metadata object should record:

- `model_backend`
- `activation_backend`
- `decode_backend`
- `device_policy`
- `dtype_compute`
- `dtype_storage`
- `quantization`
- `device_map`

### Current intended defaults

For milestone 1, the default contract should be:

- `model_backend = transformers`
- `activation_backend = wrapper`
- `decode_backend = wrapper_utils`
- `device_policy = follow_lm_head`
- `dtype_storage = float32_cpu`

This matches the current code more closely than introducing a second execution stack.

### Backend semantics

#### Model loading backend

This defines how the causal LM is instantiated and where its parameters live.

Examples:

- `transformers`
- `nnsight`
- `transformer_lens`
- `vllm` for generation-only or future decode-serving use

Milestone 1 should assume `transformers` as the canonical backend for prompt collection unless another backend is explicitly validated against it.

#### Activation collection backend

This defines how intermediate activations are captured.

Examples:

- `wrapper` for the current hook-based wrapper path
- `direct` for direct model-path collection if added later
- `nnsight` for trace-based interception if integrated later

For prompt reproducibility, one backend must be designated canonical. Right now that should be the wrapper path already used by `LogitLensWrapper`.

#### Decode backend

This defines how saved hidden states are converted into logits.

Examples:

- direct LM-head matmul
- final-norm plus LM-head projection
- tuned-lens readout
- bias-only readout

The decode backend must record whether it:

- applies final norm
- uses a learned translator
- uses a learned bias
- performs dequantization before matmul

#### Device policy

Device movement must be explicit because the model body, final norm, and LM head may not live on the same device for quantized or sharded models.

The default policy should be:

1. capture activations in the model execution path
2. move decode inputs to the final-norm device if final norm is applied
3. move normalized activations to the LM-head device before projection
4. detach and store outputs on CPU in float32

This is the effective behavior that should be standardized across wrappers.

### Backend compatibility requirement

Any non-canonical backend must satisfy a parity check against the canonical backend on the same prompt, model, and decode mode.

Minimum parity checks:

- same tokenization
- same number of collected layers
- same hidden tensor shapes
- same decoded logit tensor shapes
- numerical agreement within configured tolerance

Without this parity check, a backend should not be treated as reproducible for paper experiments.

## Numerical validity and runtime assertions

Numerical checks are part of the implementation contract.

### Required finite-value assertions

The pipeline must assert finiteness at the following stages:

- captured hidden states
- normalized hidden states
- decoded logits
- softmax probabilities
- metric outputs
- saved SVD inputs
- saved prism contribution tensors

In practice this means every tensor used for downstream analysis should satisfy:

```text
not NaN and not +/-Inf
```

before it is saved or passed to the next analysis stage.

### Required shape assertions

The pipeline must assert:

- token dimension matches the number of prompt tokens
- layer count matches the expected probed layer count
- hidden size is consistent within a run
- vocabulary dimension is consistent within a run
- top-k outputs have the configured `k`

### Required ordering assertions

The pipeline must assert that layer ordering is canonical and stable:

- embedding or synthetic pre-block layer first if included
- transformer blocks in increasing order
- optional output/final synthetic layer last if included

This matters because plotting and comparison code assume a stable low-to-high layer order.

### Required metadata assertions

Before saving an artifact, the pipeline must confirm that metadata records:

- model identifier
- tokenizer identifier
- revision if applicable
- lens mode
- backend fields
- operand ordering convention
- prompt text or prompt identifier
- tokenization used for the prompt

### Numerical stabilization rules

The implementation should follow these stabilization rules:

- compute stored artifacts in `float32` on CPU unless there is a strong reason not to
- allow model execution in lower precision, but detach saved analysis tensors in stable dtype
- dequantize LM-head weights before stable matmul when required by the backend
- apply softmax only on finite logits
- fail fast on invalid tensors instead of silently masking them

If a helper intentionally replaces invalid values, that behavior must be explicit and recorded; silent `nan_to_num` should not be the default for canonical analysis artifacts.

## Test and validation plan

The implementation plan is incomplete unless it specifies what must pass.

### Unit-level tests

The following unit tests should exist for the canonical path:

- final-norm decode returns the same shape as raw decode
- raw and ModelNorm decode preserve token axis length
- top-k extraction returns sorted token IDs and scores of length `k`
- Jaccard computation matches a hand-checked overlap case
- JSD is symmetric within tolerance
- KL directionality is preserved by field naming and value differences
- rank-delta and rank-improvement conventions have the intended sign
- artifact save/load roundtrip preserves tensor shapes and metadata fields

### Wrapper-level tests

The wrapper path should be validated on a small causal LM by checking:

- tokenization succeeds
- hooks attach and release cleanly
- expected layer count is collected
- hidden states are finite
- decoded logits are finite
- saved CPU tensors have expected dtype

### Cross-backend parity tests

Whenever a second backend is introduced, add parity tests for:

- wrapper vs direct
- wrapper vs nnsight
- canonical prompt collection vs downstream artifact reload + decode

These tests should compare:

- token IDs
- layer count
- hidden shapes
- decoded logits
- top-k token identities

Tolerance should be recorded in the test itself and in the run config if relevant.

### Quantization regression tests

Because this project aims to support quantized models, at least one regression path should test:

- non-quantized LM head
- quantized LM head
- tied embeddings
- untied embeddings if a supported model exposes them

The test requirement is not exact equality across quantization modes, but:

- no crashes
- no NaNs/Infs
- correct tensor shapes
- plausible decoded outputs

### Artifact reproducibility tests

At minimum, one test should verify:

1. run prompt collection once
2. save artifact
3. reload artifact
4. recompute a heatmap-ready metric from the artifact only
5. confirm no model rerun is required

This is central to the reproducibility claim.

### Fail-fast assertions in production code

The implementation should include runtime assertions or explicit validation calls for:

- finite hidden tensors
- finite logits
- finite probabilities
- matching prompt length across saved fields
- matching layer count across compared artifacts
- matching vocabulary size across compared artifacts unless a remapping layer is explicitly implemented

### Minimum paper-ready pass criteria

Before using a pipeline in the paper, the following should hold:

- canonical prompt collection passes on at least one reference model
- saved artifact can be reloaded and replotted without rerunning the model
- `raw` and `modelnorm` decode paths both pass finite-value checks
- `ft - base` ordering is enforced in saved comparison outputs
- Jaccard/JSD/KL/rank metrics match documented conventions
- layer ordering is correct in saved artifacts and plots
- at least one quantized path passes without NaNs/Infs
- at least one backend parity check passes if more than one backend is supported

## Artifact structure

### Prompt decode artifact

Recommended directory layout:

```text
<run_dir>/
  config.json
  wrapper_metadata.json
  capabilities.json
  layers.pt
```

Where:

- `config.json` stores run configuration
- `wrapper_metadata.json` stores model/backend metadata
- `capabilities.json` stores wrapper capability flags
- `layers.pt` stores the structured per-layer records

### Comparison artifact

Recommended layout:

```text
<comparison_dir>/
  comparison.json
  metrics.pt
  logit_diff.pt          # optional
  prob_diff.pt           # optional
  hidden_diff.pt         # optional
```

### SVD artifact

Recommended layout:

```text
<svd_dir>/
  svd_config.json
  singular_values.pt
  left_vectors.pt
  right_vectors.pt
  vocab_projection.json
```

## Current code mapping

### Real current implementation path

The real prompt execution path today is centered on:

- `LogitLensWrapper`
- `normalize_activations(...)`
- `lmhead_project(...)`
- `collect_prompt_lens_activations(...)`

### Important implication

Milestone 1 should not pretend that `src/logit_diff_lens/lenses/raw.py` and `modelnorm.py` are already full implementations.

They are currently thin re-export modules.

So the canonical milestone-1 design should be:

- use `LogitLensWrapper` for prompt collection
- formalize its output into typed schema
- preserve `hidden`
- preserve `logits_raw`
- preserve `logits_model_norm`
- implement `bias_only` and `tuned` as downstream readouts over saved `hidden`

That is the lowest-risk path grounded in the code that exists.

## First milestone implementation target

The exact first target is:

1. Add canonical dataclasses for decode results
2. Convert current prompt collector output into those dataclasses
3. Save/load those dataclasses
4. Plot one heatmap from saved artifacts
5. Add downstream decode helpers for:
   - bias-only
   - tuned

Milestone 1 does not require:

- full prism rewrite
- generation refactor
- all pairwise comparison paths
- SVD implementation

## Non-negotiable implementation rules

- Do not delete generation-lens code
- Do not replace current prompt collection with four disconnected collection paths
- Do not compute Jaccard directly on hidden states.
- Jaccard is defined on discrete token sets derived from a decode/readout step, typically the top-k tokens from logits or softmax probabilities.
- For layer-to-layer or model-to-model overlap, first decode the layer output, extract the chosen top-k token set, and then compute Jaccard on those token sets.
- Do not compute hidden diffs after readout
- Always label whether a metric or decomposition is hidden-space or readout-space
- Always save enough information to replot without rerunning the model

## What this spec is for in the paper

This spec should be sufficient to describe:

- what the four lens modes mean
- what hidden-space and readout-space comparisons mean
- how metrics are computed
- how prism decomposition is defined
- what SVD is computed over
- which artifacts are produced at each stage

That is the minimum bar for reproducibility-oriented methods writing.
