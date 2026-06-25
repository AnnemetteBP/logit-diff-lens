# Repository Structure Audit

This audit describes what already exists in the repository, what is only a
temporary bridge or scaffold, what is duplicated, and what is still missing for
the cleaner conference-oriented structure.

It is intentionally descriptive rather than prescriptive. The goal is to avoid
mistaking placeholders, donor code, or stale references for finished package
structure.

## 1. What Exists Today

### 1.1 Main package roots

There are currently three relevant Python package roots:

1. `src/diffing/`
   - This is **not** the old full package anymore.
   - It currently contains a single compatibility `__init__.py`.
   - Its job is to keep `diffing.*` imports working by extending
     `diffing.__path__` to the legacy code locations.

2. `diffing-toolkit/`
   - This contains the bulk of the non-logit-lens diffing framework:
     - `methods/`
     - `utils/`
     - `evaluators/`
     - `cli/`
   - This is real implementation code, not a placeholder.

3. `src/logitdiff-toolkit/`
   - This contains the bulk of the current logit-lens and logit-diff code:
     - `logit_lens_methods/`
     - `logit_lens_pipelines/`
   - This is also real implementation code, not a placeholder.

### 1.2 New package scaffold

`src/logit_diff_lens/` already exists and is the start of the intended clean
package. Right now it is only partially implemented.

Implemented so far:

- `wrappers/`
- `schemas/`

Present as scaffold directories with only `__init__.py` and no real module
implementation yet:

- `backends/`
- `cli/`
- `collectors/`
- `lenses/`
- `metrics/`
- `plotting/`
- `attribution/`
- `utils/`

Concrete count in `src/logit_diff_lens/` at audit time:

- 7 non-`__init__` Python files under `wrappers/`
- 1 non-`__init__` Python file under `schemas/`
- 0 non-`__init__` Python files in the other subpackages

### 1.3 Pipelines and experiment entry points

Top-level pipeline entry points already exist:

- `pipelines/em_qwen/`
- `pipelines/pythia/`
- `pipelines/quant_llama/`

These are real scripts and appear to be the intended experiment entry points for
different model families or study types.

### 1.4 Configs and datasets

The repository already has organized config and dataset directories:

- `configs/em_qwen/`
- `configs/pythia/`
- `configs/quant_llama/`
- `configs/streaming_prompt_comparison/`
- `datasets/base_data/`
- `datasets/qwen_generated_datasets/`

So the repo is not missing experiment data/config organization in general.

### 1.5 Tests

There are top-level tests already present:

- `tests/test_activation_collector.py`
- `tests/test_activation_dataset_analysis.py`
- `tests/test_latent_shift_analysis.py`
- `tests/test_logit_prism.py`
- `tests/test_representation_analysis.py`
- `tests/test_research_comparison.py`
- `tests/test_research_plots.py`

So testing is not absent, but the current tests are tightly coupled to older
module paths and some stale assumptions.

### 1.6 Documentation

The repository already contains a few important docs:

- `README.md`
- `implementation_docs/conference_restructure_plan.md`
- `docs/differential_lens_methods_README.md`

These cover:

- a minimal root usage note
- a restructuring plan
- a methods/planning document for differential lens work

### 1.7 Vendored / nested external repositories already in the repo

These directories have their own nested Git history:

- `TransformerLens/.git`
- `nnsight/.git`
- `tuned-lens/.git`
- `WordLlamaDetectAM/.git`
- `aiarbiter/.git`

So the repo already contains external projects directly in-tree rather than in a
dedicated `third_party/` area.


## 2. What Is Real vs Temporary

### 2.1 `src/diffing/` is a compatibility shim

`src/diffing/__init__.py` is currently a bridge layer, not the implementation
home for the old `diffing` package.

It extends the import path to:

- `diffing-toolkit/`
- `src/logitdiff-toolkit/`

This means:

- the import name `diffing` is still active
- the code behind `diffing` is physically split across multiple directories
- `src/diffing/` itself should not be mistaken for the true package contents

### 2.2 `src/logit_diff_lens/` is only partially real so far

The new clean package is not fake, but it is incomplete.

Real implementation already copied there:

- wrapper utilities
- wrapper classes
- wrapper-related typed schemas

Not yet migrated there:

- collectors
- metric implementations
- plotting code
- attribution/prism code
- stable backend adapters
- stable CLI commands

### 2.3 Donor code is still the real implementation source for most features

For most current functionality, the code still lives in:

- `diffing-toolkit/`
- `src/logitdiff-toolkit/`

So if you need the current implementation of a feature, those directories are
still the main source of truth.


## 3. Duplicated or Transitional Areas

### 3.1 Wrapper code exists in two places

There is active duplication between:

- `src/logitdiff-toolkit/logit_lens_methods/wrapper/`
- `src/logit_diff_lens/wrappers/`

This duplication is expected during migration, but it means the wrapper layer
does not yet have one authoritative home.

### 3.2 Package naming is transitional

The repo currently mixes:

- `diffing.*`
- `logit_lens_methods.*`
- `logit_lens_pipelines.*`
- `logit_diff_lens.*`

That means there is not yet a single canonical namespace.

### 3.3 Old `src/diffing/...` tree has been replaced

Git status shows that many files that used to live directly under
`src/diffing/...` are now deleted from that location, because the shim replaced
them.

That does **not** mean the implementation itself is gone:

- much of it still exists in `diffing-toolkit/`
- much of the logit-lens side still exists in `src/logitdiff-toolkit/`

But it does mean older path assumptions may no longer match the new physical
layout.


## 4. Stale or Broken References Already Present

These are important because they can make the repository look more complete than
it really is.

### 4.1 Stale README statements in `logit_lens_methods`

`src/logitdiff-toolkit/logit_lens_methods/README.md` says:

- `activation_collector.py` exists at the package root as a thin alias
- the real collector is under `base_collector_scripts/activation_collector.py`

At audit time, those exact paths do **not** exist.

What does exist instead is prompt-lens collector code under:

- `base_collector_scripts/prompt_lens/collect_prompt_lens_activations_batched.py`
- `base_collector_scripts/prompt_lens/collect_prompt_lens_activations.py`
- `base_collector_scripts/prompt_lens/collect_prompt_lens_logits.py`

So that README is stale and should not be treated as authoritative structure
documentation.

### 4.2 Some test imports point to modules that are genuinely absent

Examples of internal module paths referenced by tests or code that do not exist
on disk in the current layout:

- `diffing.logit_lens_methods.activation_collector`
- `diffing.logit_lens_methods.logitdiff_ldl.activation_dataset_analysis`
- `diffing.logit_lens_methods.logitdiff_ldl.latent_shift_analysis`
- `diffing.logit_lens_methods.logitdiff_ldl.representation_analysis`
- `diffing.logit_lens_methods.logitdiff_ldl.research_plots`
- `diffing.utils.dashboards`

This means some parts of the current test/doc story still point to an earlier
layout or naming scheme.

### 4.3 `logitdiff_ldl/` exists, but not the analysis modules tests expect

`src/logitdiff-toolkit/logit_lens_methods/logitdiff_ldl/` currently exports:

- `collect_ldl_logits_batched.py`
- `apply_ldl_batched.py`

The analysis modules the tests import are not located there.

Some similarly named analysis code exists elsewhere, for example:

- `prompt_lens/activation_dataset_analysis.py`
- `logitdiff_analyses/latent_shift_analysis.py`
- `logitdiff_analyses/representation_analysis.py`
- `logitdiff_analyses/research_comparison.py`

So there is a real naming/location mismatch between tests and implementation.

### 4.4 Eager imports make package import brittle

`src/logitdiff-toolkit/logit_lens_methods/__init__.py` eagerly imports several
subpackages, including prompt-lens analysis and plotting support.

This means importing a seemingly unrelated `diffing.logit_lens_methods.*`
submodule can fail early if optional or plotting dependencies are missing.

This is one reason the package currently feels fragile outside the interactive
setup.


## 5. Current Packaging and Test Reality

### 5.1 Main `pyproject.toml`

There is now one main top-level `pyproject.toml`, which is good.

It currently:

- treats `src/` as the package root
- discovers `diffing*` and `logit_diff_lens*`
- includes a broad dependency list for the current mixed codebase

### 5.2 There are still multiple nested project files in vendored repos

Even though the main repo now has one top-level `pyproject.toml`, the repository
still physically contains nested project metadata inside vendored/external
directories, for example:

- `TransformerLens/pyproject.toml`
- `nnsight/pyproject.toml`
- `tuned-lens/pyproject.toml`
- `WordLlamaDetectAM/pyproject.toml`
- `aiarbiter/pyproject.toml`

That is expected as long as these remain vendored external repos, but it means
the repo is not yet visually or structurally simplified.

### 5.3 Tests do not currently collect cleanly from the repo root

Observed current behavior:

1. Running pytest from the environment without `PYTHONPATH=src` fails because
   `diffing` is not importable by default.
2. Running with `PYTHONPATH=src` gets past that first problem, but test
   collection still fails immediately on missing dependencies such as
   `matplotlib`.

So the top-level test workflow is not yet a clean "clone and run tests" story.


## 6. What Is Missing Relative to the Intended Clean Structure

These are the main missing pieces if the goal is a cleaner conference-ready
artifact.

### 6.1 Missing real implementation in most of `src/logit_diff_lens/`

The clean package exists mostly as a scaffold. Missing migrated implementation:

- `collectors/`
- `lenses/`
- `metrics/`
- `plotting/`
- `attribution/`
- `backends/`
- most `utils/`
- stable `cli/`

### 6.2 Missing a single canonical namespace

The codebase does not yet have one obvious answer to:

- what users should import
- what internal code should import
- what is legacy versus stable API

That is still split across:

- `diffing`
- `logit_lens_methods`
- `logit_diff_lens`

### 6.3 Missing `third_party/` and `research/` separation

The restructuring plan proposed:

- `third_party/`
- `research/`

At audit time, both are missing as top-level directories.

That means:

- vendored external repos are still mixed into the root
- transitional or exploratory code has not been separated from package code

### 6.4 Missing stable package-level CLI entry points

No `project.scripts` or similar top-level console entry points are currently
defined in the main project metadata.

So the project still relies mainly on direct script execution rather than a
stable package CLI.

### 6.5 Missing a clean architecture doc for the current mixed state

There is a restructuring plan and a methods plan, but there is not yet a single
short document that explains:

- what the current package roots are
- which directories are active implementation
- which ones are vendored external repos
- which namespace is temporary
- what users should import right now

This audit partially fills that gap.

### 6.6 Missing alignment between tests and current module layout

The tests have not yet been updated to match the current physical module
locations and naming.

Some tests still assume:

- root-level `activation_collector`
- LDL analysis modules under `logitdiff_ldl`
- import behavior that no longer matches the current tree


## 7. Short Bottom Line

The repository is **not** empty or structureless. It already contains:

- substantial diffing code
- substantial logit-lens code
- working experiment pipelines
- configs and datasets
- tests
- a start on the clean wrapper-first package

But it is also **not** yet a clean unified project. Right now it is best
described as:

- a real research codebase
- plus vendored external projects
- plus a temporary import bridge
- plus a partially migrated clean package scaffold
- plus some stale tests/docs from earlier layouts

If the goal is a conference-ready artifact, the biggest missing piece is not the
core methods themselves. The biggest missing piece is finishing the consolidation
so that:

- one namespace is canonical
- donor code and vendored code are clearly separated
- the new `logit_diff_lens` package contains the real implementations
- tests and docs match the actual current layout
