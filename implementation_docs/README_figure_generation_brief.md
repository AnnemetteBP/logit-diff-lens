# Figure Generation Brief for GPT Pro

Reference figure set generated from this brief:

- `assests/docs_figures/logit_diff_framework_overview_1.png`
- `assests/docs_figures/logit_diff_forward_artifact_2.png`
- `assests/docs_figures/logit_diff_comparison_artifact_3.png`
- `assests/docs_figures/logit_diff_logit_prisms_4.png`
- `assests/docs_figures/logit_diff_patchscope_5.png`
- `assests/docs_figures/logit_diff_backward_artifact_6.png`
- `assests/docs_figures/logit_diff_weight_vocab_7.png`

Use this document as a direct brief for generating polished, publication-style figures for the `LogitDiff` repository.

The goal is to produce clean, modern figures in the visual style commonly seen in recent LLM / AI papers:

- minimal but elegant
- clear visual hierarchy
- restrained color palette
- sharp typography
- simple geometric layouts
- publication-friendly labeling
- no clutter, gimmicks, or decorative nonsense

These figures should look like they belong in:

- a strong interpretability paper
- a systems-for-research paper
- a modern LLM methodology paper

---

# Important terminology and notation

These naming conventions are mandatory.

## Required capitalization and spelling

Use:

- `LogitDiff`
- `ModelNorm`
- `Tuned Lens`
- `Logit Lens`
- `Backward Artifact`
- `Forward Capture Artifact`
- `Comparison Artifact`
- `Prompt Artifact`
- `Generation Artifact`
- `Logit Prisms`

Do **not** write:

- `logit diff`
- `model norm`
- `logitdiff lens`
- `model norm lens`

unless those appear inside running explanatory prose where lowercase is grammatically required. In titles and labels, preserve the exact forms above.

## Generic comparison notation

Do **not** hardcode the toolkit to only compare a base model against a finetuned model.

The figures must use generic notation that allows `LogitDiff` to compare any two systems, settings, or readouts.

Use generic entities such as:

- `System A`
- `System B`
- `Source A`
- `Source B`
- `Readout A`
- `Readout B`
- `Condition A`
- `Condition B`

Only include examples like:

- base model vs finetuned model
- ModelNorm vs Tuned Lens
- raw logit lens vs tuned lens
- same model with different temperatures
- same model with different chat templates
- same model with different sampling settings

as small annotations or examples, not as the primary notation of the figure.

## Meaning of LogitDiff

Very important:

`LogitDiff` is not primarily a “winner/loser” or “better/worse” toolkit.

Its main purpose is to study:

- divergence
- representational differences
- readout differences
- subblock contributions
- vocabulary-space shifts
- hidden-state differences
- target-conditioned backward signals

So the figures should emphasize:

- comparison
- contrast
- decomposition
- divergence
- reusable artifacts

not:

- ranking systems
- declaring one method superior
- benchmarking-only framing

---

# What the repository is

`LogitDiff` is a reusable research toolkit for:

- logit-lens-style analysis
- differential comparisons between two systems or conditions
- prompt and generation logit-lens analysis
- single-prompt, batched, and dataset-level analysis
- attention-mask-aware and padding-aware processing
- special-token-aware handling
- forward hidden-state capture when useful
- subblock decomposition
- vocabulary-space projection
- backward-pass target-conditioned capture
- future low-rank, quantization, and MoE analysis

Reusable saved captures are useful, but they should be presented as one workflow option rather than the main attraction of the toolkit.

---

# Figure set to generate

Create a coherent figure family with a shared visual language.

The most important figure is the main toolkit overview. The rest are method-specific subfigures or companion figures.

## Figure 1: Main toolkit overview

This should be the primary root README figure.

### Purpose

Show the overall `LogitDiff` workflow across prompt analysis, generation analysis, batching or datasets, and downstream analysis views.

### Content

Represent the flow as something like:

1. Input / system pair / condition pair
2. Wrapper / backend / model execution
3. Prompt or generation run
4. Optional backward artifact
5. Comparison artifact
6. Downstream analysis branches

The downstream branches should include:

- divergence heatmaps
- Logit Prisms
- patchscope interventions
- weight / vocabulary-space methods
- SVD / low-rank analysis
- backward target-conditioned analysis
- generation comparison
- batch / dataset aggregation

### Conceptual structure

Use a clean central pipeline with branching analysis outputs.

Suggested high-level flow:

```text
System A / System B
        ↓
   Wrapper + Backend
        ↓
Prompt Lens / Generation Lens
   ↙        ↓         ↘
Single Prompt  Batch/Dataset  Generation
        ↓
Comparison / Intervention / Interpretation
```

### Important message

The figure should communicate:

- compare generically
- support prompt and generation workflows
- support single examples, batches, and datasets
- handle real tokenization details such as masking, padding, and special tokens

---

## Figure 2: Forward capture artifact

### Purpose

Explain what a saved forward run contains when the user chooses to persist it.

### Content

Show:

- tokenization
- residual hidden states
- optional attention output
- optional MLP output
- metadata
- optional cached readouts
- attention masks
- special-token handling
- meaningful-token filtering for padded positions

### Important message

This is one useful storage path for later analysis, not the whole point of the toolkit.

The figure should emphasize that:

- saved runs can support later analysis
- masking and padding should be handled correctly
- the same toolkit also supports direct prompt, batch, dataset, and generation analysis

---

## Figure 3: Comparison artifact / LogitDiff heatmap workflow

### Purpose

Show the generic comparison path.

### Content

Show two saved runs entering a comparison step:

```text
Artifact A + Artifact B
        ↓
Comparison
        ↓
one possible ordering is:
comparison-minus-reference
        ↓
metrics + heatmap-ready payload
```

But do not lock the notation to finetuned/base.

Use labels like:

- `A - B`
- `Comparison - Reference`

and optionally mention that a common setting is `ft - base`.

### Metrics to visually suggest

- JSD
- KL
- top-k overlap / Jaccard
- rank change
- hidden-space distances

### Important message

This is about divergence and comparison, not “who wins.”

---

## Figure 4: Logit Prisms / subblock decomposition

### Purpose

Show how a saved capture artifact can be decomposed into interpretable components.

### Content

Visually split a residual update or decoded contribution into:

- embedding
- attention
- MLP
- full residual stream

Potentially show two modes:

- hidden-space component analysis
- vocab/logit-space component analysis

### Important message

The same saved artifact can be used to localize where a divergence arises.

---

## Figure 5: Patchscope workflow

### Purpose

Show how `LogitDiff` uses saved forward artifacts as the source of patchable representations.

### Content

Show a source prompt artifact on the left and a target prompt run on the right.

The figure should visually mark:

- source layer / source position
- target layer / target position
- identity mapping for the current implementation
- patched target readout
- saved `PatchscopePromptArtifact`

Also visually suggest the future sweep direction:

- patch across positions
- patch across layers
- patch selected high-difference tokens

### Important message

The patchscope path should look like a natural extension of the artifact-first design:

- capture once
- select source representation
- intervene on a target run
- save reusable intervention artifacts

This figure should not look like a one-off debugging script. It should look like a proper analysis family inside the toolkit.

---

## Figure 6: Backward artifact workflow

### Purpose

Show how backward-pass analysis differs from ordinary forward logit-lens analysis.

### Content

Show:

- prompt
- chosen target token
- forward logits
- NLL loss
- single backward pass
- captured VJPs / backward signals
- backward artifact

### Important message

Backward analysis is target-conditioned and is a separate artifact family, not the same thing as a standard forward readout.

---

## Figure 7: Weight / vocabulary-space methods

### Purpose

Show the static vs dynamic distinction for SVD and vocabulary projection methods.

### Content

Split into two sides:

### Static side

- model weights
- OV / MLP / embedding / unembedding matrices
- SVD / spectral decomposition
- vocabulary projection of directions

### Dynamic side

- forward capture artifact
- activation-conditioned contributions
- projected sub-updates or band occupancy

### Important message

This toolkit supports both:

- model-weight-level interpretability
- prompt-conditioned activation analysis

under one shared artifact philosophy.

---

# Style requirements

## Overall aesthetic

Make the figure style look like a modern AI paper.

Target qualities:

- crisp
- geometric
- restrained
- professional
- slightly premium
- easy to parse quickly

Avoid:

- childish icons
- cartoonish doodles
- saturated rainbow colors
- overly dense text
- cheesy gradients
- generic “AI chip” clipart

## Layout

Use:

- clean boxes
- arrows with consistent thickness
- subtle grouping
- consistent spacing
- aligned columns or lanes

The main figure should feel balanced and symmetric enough to read easily, but not sterile.

## Color palette

Use a restrained palette, for example:

- deep slate / charcoal
- muted blue
- muted teal
- soft red or rust for contrast
- light neutral background

Avoid purple-heavy defaults.

If color semantics are used:

- neutral artifact/storage objects can be blue/gray
- comparison/divergence objects can use teal vs rust
- backward artifacts may use a distinct warm accent

## Typography

Use a clean, publication-like sans serif.

Text should be:

- concise
- not too small
- title case for figure headings
- exact method names preserved

## Labels

Use labels like:

- `Forward Capture Artifact`
- `Comparison Artifact`
- `Backward Artifact`
- `Logit Prisms`
- `Weight / Vocab Methods`
- `SVD / Low-Rank Analysis`
- `Prompt Workflow`
- `Generation Workflow`

Keep labels short and readable.

---

# Semantic requirements

## Reusability

The figures must visually communicate that:

- artifacts are reusable
- one capture supports many analyses
- multiple methods are built on shared saved states

## Genericity

The figures must show that `LogitDiff` can compare:

- model A vs model B
- readout A vs readout B
- same model under different decoding settings
- same model under different templates
- different lens types

without visually committing to only “base vs finetuned.”

## Extensibility

The figures should leave room conceptually for:

- quantization-aware analysis
- low-rank adapters
- MoE support
- new backends
- additional lens families

## Reproducibility

The figures should imply:

- typed artifacts
- validation
- backend metadata
- canonical ordering conventions

---

# Deliverables requested from GPT Pro

Please generate:

1. One main overview figure for the repository root README
2. One smaller companion figure for each major method family:
   - forward capture
   - comparison / heatmaps
   - Logit Prisms
   - patchscope workflow
   - backward artifacts
   - weight / vocabulary-space methods
3. All figures in a coherent shared style
4. Prefer vector-style outputs or clean high-resolution diagram outputs

If you provide captions, keep them concise and paper-style.

---

# Final instruction

Do not frame `LogitDiff` as a benchmark leaderboard or a tool whose main purpose is proving one method better than another.

Frame it as:

- a divergence-focused interpretability toolkit
- a practical analysis system for prompts, batches, datasets, and generations
- a reusable foundation for many kinds of logit-lens, differential, prism, weight-space, and backward-pass analyses
