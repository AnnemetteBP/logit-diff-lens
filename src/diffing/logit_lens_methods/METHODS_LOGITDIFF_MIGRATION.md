methods/logitdiff -> logit_lens_methods

Exact mapping

- `src/diffing/methods/logitdiff/method.py`
  - current target: `src/diffing/logit_lens_methods/logitdiff/core.py`
  - status: partially migrated
  - note: this is the old autoregressive `generate(...)` reference path

- `src/diffing/methods/logitdiff/jaccard_heatmap_plotter.py`
  - current target: `src/diffing/logit_lens_methods/plotting/heatmaps/logitdiff_gen_plotter.py`
  - status: migrated

- `src/diffing/methods/logitdiff/agents.py`
  - current target: `src/diffing/logit_lens_methods/agents/agent.py`
  - status: migrated by role

- `src/diffing/methods/logitdiff/agent_tools.py`
  - current target: `src/diffing/logit_lens_methods/agents/agent_tools.py`
  - status: migrated by role

- `src/diffing/methods/logitdiff/logit_extraction.py`
  - current target: no 1:1 moved file
  - status: not needed as-is in current wrapper-based logit_lens path

- `src/diffing/methods/logitdiff/__init__.py`
  - current target: `src/diffing/logit_lens_methods/logitdiff/__init__.py`
  - status: migrated package entrypoint

What is already in logit_lens_methods

- autoregressive logitdiff logic:
  - `logitdiff/core.py`

- autoregressive generation heatmap plotting:
  - `plotting/heatmaps/logitdiff_gen_plotter.py`

- teacher-forcing verification heatmap plotting:
  - `plotting/heatmaps/logitdiff_tf_plotter.py`

- pairwise comparison heatmap plotting:
  - `plotting/heatmaps/logitdiff_pair_heatmap_plotter.py`

- agent layer:
  - `agents/agent.py`
  - `agents/agent_tools.py`
  - `agents/agent_io.py`

What still needs cleanup

- `logitdiff/core.py` and `wrapper/lens_wrappers/generation_lens_wrapper.py`
  - both now contain generation-related behavior
  - this should be collapsed to one clean release path

- `src/diffing/methods/logitdiff/`
  - should be treated as legacy reference until the last needed behavior is fully absorbed
