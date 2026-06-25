def run_gen_lens_pipeline(*args, **kwargs):
    raise NotImplementedError("Use the root pipelines while CLI migration is in progress.")


def run_prompt_lens_pipeline(*args, **kwargs):
    raise NotImplementedError("Use the root pipelines while CLI migration is in progress.")


def run_single_prompt_patch_sweep(*args, **kwargs):
    raise NotImplementedError("Use the root pipelines while CLI migration is in progress.")

__all__ = [
    "run_gen_lens_pipeline",
    "run_prompt_lens_pipeline",
    "run_single_prompt_patch_sweep",
]
