# Conference Restructure Plan

## Goal

Turn this repository into a conference-ready artifact without destroying prior work.

The key principle is:

- keep all existing research code available
- define one clean core package
- move borrowed or vendored code out of the package namespace
- make methods depend on our wrapper/backend interfaces rather than on external projects directly


## Current Problems

The repository currently mixes several roles:

- core package code
- exploratory research code
- one-off pipelines
- vendored external repositories
- paper outputs and temporary artifacts

This makes it hard to answer:

- what is the actual public package?
- what is stable API versus exploratory code?
- what code is ours versus borrowed?
- what exactly is required to reproduce the paper?


## Core Design Decision

The conference artifact should be organized around **our wrapper API**.

External projects like `nnsight`, `TransformerLens`, and `tuned-lens` should not define the structure of our codebase.

Instead:

- our package defines the interfaces
- our wrappers expose stable capabilities
- collectors, metrics, attribution, and plotting consume those interfaces
- external libraries are optional backends, adapters, or vendored support code


## Target Top-Level Layout

```text
.
├── configs/
├── datasets/
├── docs/
├── pipelines/
├── src/
│   └── logit_diff_lens/
├── tests/
├── third_party/
├── research/
└── tmp/
```

### Intended meanings

- `src/logit_diff_lens/`
  - the only first-class package source
- `third_party/`
  - vendored or modified external code that must remain in-repo
- `pipelines/`
  - reproducible experiment entry points
- `configs/`
  - named run configs for artifact and paper experiments
- `research/`
  - exploratory scripts, legacy analyses, transitional code not part of the clean package API
- `tmp/`
  - outputs only, never imported


## Target Package Layout

```text
src/logit_diff_lens/
├── __init__.py
├── backends/
├── wrappers/
├── collectors/
├── lenses/
├── metrics/
├── attribution/
├── plotting/
├── schemas/
├── cli/
└── utils/
```

### Package responsibilities

- `backends/`
  - backend-specific adapters such as Hugging Face, optional `nnsight`, optional `vllm`
- `wrappers/`
  - stable wrapper interfaces and concrete implementations
- `collectors/`
  - prompt/generation activation and logit collection
- `lenses/`
  - logit lens, ModelNorm lens, tuned-lens integration
- `metrics/`
  - JSD, KL, IoU/Jaccard, hidden-state distance, consistency metrics
- `attribution/`
  - logit prisms and differential prisms
- `plotting/`
  - reusable plotting functions only
- `schemas/`
  - typed result payloads and run metadata
- `cli/`
  - stable artifact commands
- `utils/`
  - genuinely shared utilities only


## Current To Target Mapping

### Keep and migrate into the core package

- `src/logitdiff-toolkit/logit_lens_methods/wrapper/`
  - target: `src/logit_diff_lens/wrappers/`
- `src/logitdiff-toolkit/logit_lens_methods/base_collector_scripts/`
  - target: `src/logit_diff_lens/collectors/`
- `src/logitdiff-toolkit/logit_lens_methods/prompt_lens/`
  - target: `src/logit_diff_lens/collectors/` and `src/logit_diff_lens/lenses/`
- `src/logitdiff-toolkit/logit_lens_methods/logit_prisms/`
  - target: `src/logit_diff_lens/attribution/`
- `src/logitdiff-toolkit/logit_lens_methods/plotting/`
  - target: `src/logit_diff_lens/plotting/`
- `src/logitdiff-toolkit/logit_lens_methods/logitdiff_gen/`
  - target: split across `collectors/`, `metrics/`, and `schemas/`
- `src/logitdiff-toolkit/logit_lens_methods/logitdiff/`
  - target: `src/logit_diff_lens/metrics/` and `src/logit_diff_lens/schemas/`

### Keep, but move to `research/` or `legacy/` during transition

- `src/logitdiff-toolkit/logit_lens_methods/logitdiff_adl/`
- `src/logitdiff-toolkit/logit_lens_methods/logitdiff_ldl/`
- `src/logitdiff-toolkit/logit_lens_methods/logitdiff_analyses/`
- `src/logitdiff-toolkit/logit_lens_methods/pipelines/`
- one-off plot builders tied directly to paper/tmp outputs

These are likely valuable, but they should not all remain first-class package API.

### Keep outside package as experiment entry points

- `pipelines/em_qwen/`
- `pipelines/pythia/`
- `pipelines/quant_llama/`

These should call the clean package, not contain package logic themselves.


## External Projects Strategy

External repositories should not live under `src/` as if they are part of our package.

### Preferred order of options

1. **Dependency only**
   - use a released package or pinned Git dependency
   - keep only our adapter code in `src/logit_diff_lens/`

2. **Vendored support code**
   - if modifications are needed, move code to `third_party/`
   - add a README with:
     - upstream repo
     - upstream commit or version
     - local modifications
     - reason vendoring was required

3. **External maintained fork**
   - if divergence is large, maintain a real fork outside the package namespace

### Recommendation for this repo

Move external projects out of `src/` and either:

- replace them with pinned dependencies plus adapters, or
- move modified copies into `third_party/`

Nested `.git` histories should not remain hidden inside the main package source tree.


## Wrapper and Backend Plan

This is the most important part of the refactor.

### Stable wrapper contract

The package should define a stable wrapper API that methods can rely on.

Minimum capabilities:

- tokenize prompt inputs
- run prompt forward pass
- run generation pass
- expose layer registry
- expose residual activations
- expose optional attention and MLP component outputs
- expose final norm
- expose LM-head projection
- report quantization/backend metadata

### Recommended wrapper split

- `wrappers/base.py`
  - abstract wrapper protocol
- `wrappers/prompt.py`
  - teacher-forced prompt analysis wrapper
- `wrappers/generation.py`
  - `model.generate(...)` and generation replay wrapper
- `wrappers/patching.py`
  - activation patching wrapper
- `backends/hf.py`
  - default Hugging Face implementation
- `backends/nnsight.py`
  - optional backend adapter
- `backends/vllm.py`
  - optional inference backend later

### Why this matters

This gives us one place to solve:

- architecture differences
- quantization differences
- tied versus untied embeddings
- final norm differences
- hidden-state access differences
- generate behavior differences

That is exactly what the conference artifact needs for reproducibility.


## Metrics and Attribution Plan

### Metrics

Move stable metrics into `metrics/`:

- `jsd`
- directional `kl`
- `iou` / Jaccard top-k overlap
- hidden cosine / distance
- next-token correctness flags

### Attribution

Use the existing prism work as the basis for:

- residual prism
- attention prism
- MLP prism
- later: differential prism

Target location:

- `src/logit_diff_lens/attribution/prisms.py`
- or a small attribution package under `attribution/`


## What the Conference Artifact Should Support First

Keep the scope deliberately narrow at first.

Recommended initial artifact scope:

- supported family: `Pythia` first
- one or two additional families only if wrappers are stable
- lenses:
  - logit lens
  - ModelNorm lens
  - tuned-lens integration
- comparisons:
  - prompt-level comparison
  - one generation-level comparison path
- attribution:
  - residual-level prism first
- plots:
  - a small stable set only

Do not try to ship every exploratory method in the first artifact.


## Phased Migration Plan

### Phase 1: define package boundary

- create `src/logit_diff_lens/`
- stop treating `diffing` or `logitdiff-toolkit` as the final public package namespace
- decide what is core versus research versus third-party

### Phase 2: wrapper-first extraction

- move the wrapper utilities and wrapper classes into the new package
- define the stable wrapper API
- keep shims in the old locations temporarily if needed

### Phase 3: collectors and metrics

- move prompt/generation collectors into the new package
- move metric computation into `metrics/`
- introduce stable result schemas

### Phase 4: plotting

- move reusable plotting functions into `plotting/`
- leave paper-specific figure builders in `research/` until cleaned

### Phase 5: attribution

- migrate prism code into `attribution/`
- add a first differential prism runner

### Phase 6: package and CLI cleanup

- rewrite `pyproject.toml`
- add minimal stable CLI entry points
- separate optional extras from required dependencies

### Phase 7: tests and artifact docs

- add smoke tests for:
  - wrapper initialization
  - prompt collection
  - generation collection
  - metric computation
  - plotting serialization
- add reproduction docs for paper experiments


## Proposed `pyproject.toml` Direction

The new package should:

- expose only `logit_diff_lens`
- declare `requires-python`
- keep core dependencies minimal
- move notebook/dev-heavy dependencies into extras
- provide optional extras for fragile integrations

Suggested extras:

- `plotting`
- `quant`
- `nnsight`
- `tuned-lens`
- `dev`


## Immediate Next Step

The best next implementation step is:

1. create the new package root `src/logit_diff_lens/`
2. move or copy the wrapper layer there first
3. add compatibility shims only if needed
4. then rewrite `pyproject.toml` around that new package root

That preserves prior work while giving the project a clean architectural spine.
