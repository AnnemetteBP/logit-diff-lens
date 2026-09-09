from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from .live_diff import (
    build_live_generation_logitdiff_payload,
    build_live_prompt_logitdiff_payload,
)
from .plotting.generation_jaccard_heatmap_plotter import plot_logitdiff_jaccard_heatmap
from .plotting.prompt_jaccard_heatmap_plotter import plot_jaccard_heatmap
from .plotting.single_model_logit_lens_plotter import plot_single_model_logit_lens_heatmap
from .wrappers import GenerateLensWrapper, LogitLensWrapper


def _single_metric_kwargs(metric_key: str) -> dict[str, bool]:
    return {
        "logits_std": {"logits_std": True},
        "logit_margin": {"logit_margin": True},
        "max_prob": {"probs": True},
        "probs_std": {"probs_std": True},
        "gt_prob": {"ground_truth_probs": True},
        "entropy": {"entropy": True},
        "perplexity": {"perplexity": True},
        "kl_div_prev": {"kl_div_prev": True},
        "kl_div_last": {"kl_div_last": True},
        "js_div_prev": {"js_div_prev": True},
        "js_div_last": {"js_div_last": True},
        "cos_sim_prev": {"cos_sim_prev": True},
        "cos_sim_last": {"cos_sim_last": True},
        "l2_dist_prev": {"l2_dist_prev": True},
        "l2_dist_last": {"l2_dist_last": True},
        "jaccard_prev": {"jaccard_prev": True},
        "jaccard_last": {"jaccard_last": True},
        "accuracy_topk": {"topk_accuracy": True},
        "logits_mean": {},
    }.get(metric_key, {})


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    device_map: str | dict[str, Any] | None = "auto",
    torch_dtype: str | None = "auto",
    trust_remote_code: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    return model, tokenizer


def make_prompt_wrapper(
    model,
    tokenizer,
    *,
    include_final_norm: bool = True,
    stable_analysis: bool = True,
    debug: bool = False,
) -> LogitLensWrapper:
    return LogitLensWrapper(
        model=model,
        tokenizer=tokenizer,
        include_final_norm=include_final_norm,
        stable_analysis=stable_analysis,
        debug=debug,
    )


def make_generation_wrapper(
    model,
    tokenizer,
    *,
    include_final_norm: bool = True,
    stable_analysis: bool = True,
    debug: bool = False,
) -> GenerateLensWrapper:
    return GenerateLensWrapper(
        model=model,
        tokenizer=tokenizer,
        include_final_norm=include_final_norm,
        stable_analysis=stable_analysis,
        debug=debug,
    )


def build_single_model_widget(
    model,
    tokenizer,
    *,
    default_prompt: str = "The capital of France is",
):
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception as exc:  # pragma: no cover
        raise ImportError("ipywidgets is required for the notebook widget.") from exc

    wrapper = make_prompt_wrapper(model, tokenizer)

    prompt_wdg = widgets.Textarea(
        value=default_prompt,
        description="Prompt:",
        layout=widgets.Layout(width="100%", height="120px"),
    )
    norm_wdg = widgets.Dropdown(
        options=[("ModelNorm", "model_norm"), ("Raw", "raw")],
        value="model_norm",
        description="Lens:",
    )
    metric_wdg = widgets.Dropdown(
        options=[
            ("Logits", "logits_mean"),
            ("Logits std", "logits_std"),
            ("Top-2 logit margin", "logit_margin"),
            ("Max probability", "max_prob"),
            ("Probability std", "probs_std"),
            ("Ground-truth probability", "gt_prob"),
            ("Entropy", "entropy"),
            ("Perplexity", "perplexity"),
            ("KL vs Previous", "kl_div_prev"),
            ("KL vs Last", "kl_div_last"),
            ("JSD vs Previous", "js_div_prev"),
            ("JSD vs Last", "js_div_last"),
            ("Cosine vs Previous", "cos_sim_prev"),
            ("Jaccard vs Last", "jaccard_last"),
            ("Jaccard vs Previous", "jaccard_prev"),
            ("Accuracy@5", "accuracy_topk"),
            ("Cosine vs Last", "cos_sim_last"),
            ("L2 vs Last", "l2_dist_last"),
            ("L2 vs Previous", "l2_dist_prev"),
        ],
        value="js_div_last",
        description="Metric:",
    )
    topk_wdg = widgets.BoundedIntText(value=5, min=1, max=20, description="Top-k:")
    add_special_wdg = widgets.Checkbox(value=True, description="Add special tokens")
    input_wdg = widgets.Checkbox(value=True, description="Force include input")
    output_wdg = widgets.Checkbox(value=True, description="Force include output")
    marginals_wdg = widgets.Checkbox(value=False, description="Show marginals")
    mark_correct_wdg = widgets.Checkbox(value=True, description="Mark correct preds")
    block_steps_wdg = widgets.BoundedIntText(value=1, min=1, max=32, description="Layer collapse:")
    start_wdg = widgets.IntText(value=0, description="Start idx:")
    end_wdg = widgets.IntText(value=-1, description="End idx:")
    run_btn = widgets.Button(description="Run Lens", button_style="primary")
    out = widgets.Output()

    def _run(_):
        with out:
            out.clear_output(wait=True)
            end_idx = None if int(end_wdg.value) < 0 else int(end_wdg.value)
            fig = plot_single_model_logit_lens_heatmap(
                wrapper,
                prompt=prompt_wdg.value,
                norm_mode=norm_wdg.value,
                topk=int(topk_wdg.value),
                add_special_tokens=bool(add_special_wdg.value),
                force_include_input=bool(input_wdg.value),
                force_include_output=bool(output_wdg.value),
                show_marginals=bool(marginals_wdg.value),
                mark_correct_preds=bool(mark_correct_wdg.value),
                block_steps=int(block_steps_wdg.value),
                start_idx=max(0, int(start_wdg.value)),
                end_idx=end_idx,
                **_single_metric_kwargs(metric_wdg.value),
            )
            display(fig)

    run_btn.on_click(_run)
    controls_row_1 = widgets.HBox([norm_wdg, metric_wdg, topk_wdg, block_steps_wdg])
    controls_row_2 = widgets.HBox([start_wdg, end_wdg, add_special_wdg, marginals_wdg])
    controls_row_3 = widgets.HBox([input_wdg, output_wdg, mark_correct_wdg, run_btn])
    ui = widgets.VBox([prompt_wdg, controls_row_1, controls_row_2, controls_row_3, out])
    return ui


def build_prompt_payload_widget(
    payload_path: str | Path | None = None,
):
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception as exc:  # pragma: no cover
        raise ImportError("ipywidgets is required for the notebook widget.") from exc

    path_wdg = widgets.Text(
        value=str(payload_path or ""),
        description="Payload:",
        layout=widgets.Layout(width="100%"),
    )
    prompt_index_wdg = widgets.IntText(value=0, description="Prompt idx:")
    prompt_text_wdg = widgets.Text(value="", description="Prompt text:")
    include_prompt_wdg = widgets.Checkbox(value=True, description="Include prompt")
    include_generated_wdg = widgets.Checkbox(value=True, description="Include generated")
    start_wdg = widgets.IntText(value=0, description="Start idx:")
    end_wdg = widgets.IntText(value=-1, description="End idx:")
    display_top_wdg = widgets.BoundedIntText(value=5, min=1, max=20, description="Show top-k:")
    max_layers_wdg = widgets.IntText(value=8, description="Max layers:")
    layer_selection_wdg = widgets.Dropdown(
        options=[("Most divergent", "most_divergent"), ("Least divergent", "least_divergent"), ("First N", "all")],
        value="most_divergent",
        description="Layers:",
    )
    marginals_wdg = widgets.Checkbox(value=False, description="Show marginals")
    run_btn = widgets.Button(description="Plot", button_style="primary")
    out = widgets.Output()

    def _run(_):
        with out:
            out.clear_output(wait=True)
            path = Path(path_wdg.value).expanduser()
            fig = plot_jaccard_heatmap(
                path,
                prompt_index=int(prompt_index_wdg.value),
                prompt_text=prompt_text_wdg.value or None,
                include_prompt_tokens=bool(include_prompt_wdg.value),
                include_generated_tokens=bool(include_generated_wdg.value),
                start_idx=max(0, int(start_wdg.value)),
                end_idx=None if int(end_wdg.value) < 0 else int(end_wdg.value),
                display_top_tokens=int(display_top_wdg.value),
                max_layers=None if int(max_layers_wdg.value) <= 0 else int(max_layers_wdg.value),
                layer_selection=layer_selection_wdg.value,
                show_marginals=bool(marginals_wdg.value),
            )
            display(fig)

    run_btn.on_click(_run)
    return widgets.VBox(
        [
            path_wdg,
            widgets.HBox([prompt_index_wdg, prompt_text_wdg]),
            widgets.HBox([include_prompt_wdg, include_generated_wdg, display_top_wdg, max_layers_wdg]),
            widgets.HBox([start_wdg, end_wdg, layer_selection_wdg, marginals_wdg, run_btn]),
            out,
        ]
    )


def build_prompt_logitdiff_widget(
    base_model,
    base_tokenizer,
    compare_model,
    compare_tokenizer,
    *,
    default_prompt: str = "The capital of France is",
):
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception as exc:  # pragma: no cover
        raise ImportError("ipywidgets is required for the notebook widget.") from exc

    base_wrapper = make_prompt_wrapper(base_model, base_tokenizer)
    compare_wrapper = make_prompt_wrapper(compare_model, compare_tokenizer)

    prompt_wdg = widgets.Textarea(
        value=default_prompt,
        description="Prompt:",
        layout=widgets.Layout(width="100%", height="120px"),
    )
    norm_wdg = widgets.Dropdown(
        options=[("ModelNorm", "model_norm"), ("Raw", "raw")],
        value="model_norm",
        description="Lens:",
    )
    topk_wdg = widgets.BoundedIntText(value=5, min=1, max=20, description="Top-k:")
    add_special_wdg = widgets.Checkbox(value=True, description="Add special tokens")
    input_wdg = widgets.Checkbox(value=True, description="Force include input")
    output_wdg = widgets.Checkbox(value=True, description="Force include output")
    start_wdg = widgets.IntText(value=0, description="Start idx:")
    end_wdg = widgets.IntText(value=-1, description="End idx:")
    display_top_wdg = widgets.BoundedIntText(value=5, min=1, max=20, description="Show top-k:")
    max_layers_wdg = widgets.IntText(value=8, description="Max layers:")
    layer_selection_wdg = widgets.Dropdown(
        options=[("Most divergent", "most_divergent"), ("Least divergent", "least_divergent"), ("All / first N", "all")],
        value="most_divergent",
        description="Layers:",
    )
    marginals_wdg = widgets.Checkbox(value=False, description="Show marginals")
    run_btn = widgets.Button(description="Run Prompt Diff", button_style="primary")
    out = widgets.Output()

    def _run(_):
        with out:
            out.clear_output(wait=True)
            payload = build_live_prompt_logitdiff_payload(
                base_wrapper,
                compare_wrapper,
                prompt=prompt_wdg.value,
                readout_mode=norm_wdg.value,
                top_k=int(topk_wdg.value),
                add_special_tokens=bool(add_special_wdg.value),
                force_include_input=bool(input_wdg.value),
                force_include_output=bool(output_wdg.value),
            )
            fig = plot_jaccard_heatmap(
                payload,
                prompt_index=0,
                include_prompt_tokens=True,
                include_generated_tokens=False,
                start_idx=max(0, int(start_wdg.value)),
                end_idx=None if int(end_wdg.value) < 0 else int(end_wdg.value),
                display_top_tokens=int(display_top_wdg.value),
                max_layers=None if int(max_layers_wdg.value) <= 0 else int(max_layers_wdg.value),
                layer_selection=layer_selection_wdg.value,
                show_marginals=bool(marginals_wdg.value),
            )
            display(fig)

    run_btn.on_click(_run)
    return widgets.VBox(
        [
            prompt_wdg,
            widgets.HBox([norm_wdg, topk_wdg, display_top_wdg, max_layers_wdg]),
            widgets.HBox([add_special_wdg, input_wdg, output_wdg, marginals_wdg]),
            widgets.HBox([start_wdg, end_wdg, layer_selection_wdg, run_btn]),
            out,
        ]
    )


def build_generation_payload_widget(
    payload_path: str | Path | None = None,
):
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception as exc:  # pragma: no cover
        raise ImportError("ipywidgets is required for the notebook widget.") from exc

    path_wdg = widgets.Text(
        value=str(payload_path or ""),
        description="Payload:",
        layout=widgets.Layout(width="100%"),
    )
    prompt_index_wdg = widgets.IntText(value=0, description="Prompt idx:")
    prompt_text_wdg = widgets.Text(value="", description="Prompt text:")
    include_prompt_wdg = widgets.Checkbox(value=False, description="Include prompt")
    include_generated_wdg = widgets.Checkbox(value=True, description="Include generated")
    start_wdg = widgets.IntText(value=0, description="Start idx:")
    end_wdg = widgets.IntText(value=-1, description="End idx:")
    display_top_wdg = widgets.BoundedIntText(value=5, min=1, max=20, description="Show top-k:")
    max_layers_wdg = widgets.IntText(value=8, description="Max layers:")
    layer_selection_wdg = widgets.Dropdown(
        options=[("Most divergent", "most_divergent"), ("Least divergent", "least_divergent"), ("All / first N", "all")],
        value="most_divergent",
        description="Layers:",
    )
    x_tick_wdg = widgets.Dropdown(
        options=[
            ("FT generated", "ft_generated"),
            ("Base generated", "base_generated"),
            ("Input tokens", "input_tokens"),
            ("FT top-1", "ft_top1"),
            ("Base top-1", "base_top1"),
            ("Position", "position"),
        ],
        value="ft_generated",
        description="X-axis 1:",
    )
    x_tick_secondary_wdg = widgets.Dropdown(
        options=[
            ("Base generated", "base_generated"),
            ("FT generated", "ft_generated"),
            ("Input tokens", "input_tokens"),
            ("FT top-1", "ft_top1"),
            ("Base top-1", "base_top1"),
            ("Position", "position"),
            ("None", None),
        ],
        value="base_generated",
        description="X-axis 2:",
    )
    marginals_wdg = widgets.Checkbox(value=False, description="Show marginals")
    run_btn = widgets.Button(description="Plot", button_style="primary")
    out = widgets.Output()

    def _run(_):
        with out:
            out.clear_output(wait=True)
            path = Path(path_wdg.value).expanduser()
            fig = plot_logitdiff_jaccard_heatmap(
                path,
                prompt_index=int(prompt_index_wdg.value),
                prompt_text=prompt_text_wdg.value or None,
                include_prompt_tokens=bool(include_prompt_wdg.value),
                include_generated_tokens=bool(include_generated_wdg.value),
                start_idx=max(0, int(start_wdg.value)),
                end_idx=None if int(end_wdg.value) < 0 else int(end_wdg.value),
                display_top_tokens=int(display_top_wdg.value),
                max_layers=None if int(max_layers_wdg.value) <= 0 else int(max_layers_wdg.value),
                layer_selection=layer_selection_wdg.value,
                x_tick_mode=x_tick_wdg.value,
                x_tick_mode_secondary=x_tick_secondary_wdg.value,
                show_marginals=bool(marginals_wdg.value),
            )
            display(fig)

    run_btn.on_click(_run)
    return widgets.VBox(
        [
            path_wdg,
            widgets.HBox([prompt_index_wdg, prompt_text_wdg]),
            widgets.HBox([include_prompt_wdg, include_generated_wdg, display_top_wdg, max_layers_wdg]),
            widgets.HBox([start_wdg, end_wdg, layer_selection_wdg, marginals_wdg]),
            widgets.HBox([x_tick_wdg, x_tick_secondary_wdg, run_btn]),
            out,
        ]
    )


def build_generation_logitdiff_widget(
    base_model,
    base_tokenizer,
    compare_model,
    compare_tokenizer,
    *,
    default_prompt: str = "The capital of France is",
):
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception as exc:  # pragma: no cover
        raise ImportError("ipywidgets is required for the notebook widget.") from exc

    base_wrapper = make_prompt_wrapper(base_model, base_tokenizer)
    compare_wrapper = make_prompt_wrapper(compare_model, compare_tokenizer)

    prompt_wdg = widgets.Textarea(
        value=default_prompt,
        description="Prompt:",
        layout=widgets.Layout(width="100%", height="120px"),
    )
    norm_wdg = widgets.Dropdown(
        options=[("ModelNorm", "model_norm"), ("Raw", "raw")],
        value="model_norm",
        description="Lens:",
    )
    topk_wdg = widgets.BoundedIntText(value=5, min=1, max=20, description="Top-k:")
    new_tokens_wdg = widgets.BoundedIntText(value=24, min=1, max=256, description="New toks:")
    add_special_wdg = widgets.Checkbox(value=True, description="Add special tokens")
    do_sample_wdg = widgets.Checkbox(value=False, description="Sample")
    temperature_wdg = widgets.FloatText(value=1.0, description="Temp:")
    seed_wdg = widgets.IntText(value=0, description="Seed:")
    include_prompt_wdg = widgets.Checkbox(value=False, description="Include prompt")
    include_generated_wdg = widgets.Checkbox(value=True, description="Include generated")
    start_wdg = widgets.IntText(value=0, description="Start idx:")
    end_wdg = widgets.IntText(value=-1, description="End idx:")
    display_top_wdg = widgets.BoundedIntText(value=5, min=1, max=20, description="Show top-k:")
    max_layers_wdg = widgets.IntText(value=8, description="Max layers:")
    layer_selection_wdg = widgets.Dropdown(
        options=[("Most divergent", "most_divergent"), ("Least divergent", "least_divergent"), ("All / first N", "all")],
        value="most_divergent",
        description="Layers:",
    )
    x_tick_wdg = widgets.Dropdown(
        options=[
            ("FT generated", "ft_generated"),
            ("Base generated", "base_generated"),
            ("Input tokens", "input_tokens"),
            ("FT top-1", "ft_top1"),
            ("Base top-1", "base_top1"),
            ("Position", "position"),
        ],
        value="ft_generated",
        description="X-axis 1:",
    )
    x_tick_secondary_wdg = widgets.Dropdown(
        options=[
            ("Base generated", "base_generated"),
            ("FT generated", "ft_generated"),
            ("Input tokens", "input_tokens"),
            ("FT top-1", "ft_top1"),
            ("Base top-1", "base_top1"),
            ("Position", "position"),
            ("None", None),
        ],
        value="base_generated",
        description="X-axis 2:",
    )
    marginals_wdg = widgets.Checkbox(value=False, description="Show marginals")
    run_btn = widgets.Button(description="Run Gen Diff", button_style="primary")
    out = widgets.Output()

    def _run(_):
        with out:
            out.clear_output(wait=True)
            payload = build_live_generation_logitdiff_payload(
                base_wrapper,
                compare_wrapper,
                prompt=prompt_wdg.value,
                readout_mode=norm_wdg.value,
                top_k=int(topk_wdg.value),
                max_new_tokens=int(new_tokens_wdg.value),
                add_special_tokens=bool(add_special_wdg.value),
                do_sample=bool(do_sample_wdg.value),
                temperature=float(temperature_wdg.value),
                seed=None if int(seed_wdg.value) < 0 else int(seed_wdg.value),
            )
            fig = plot_logitdiff_jaccard_heatmap(
                payload,
                prompt_index=0,
                include_prompt_tokens=bool(include_prompt_wdg.value),
                include_generated_tokens=bool(include_generated_wdg.value),
                start_idx=max(0, int(start_wdg.value)),
                end_idx=None if int(end_wdg.value) < 0 else int(end_wdg.value),
                display_top_tokens=int(display_top_wdg.value),
                max_layers=None if int(max_layers_wdg.value) <= 0 else int(max_layers_wdg.value),
                layer_selection=layer_selection_wdg.value,
                x_tick_mode=x_tick_wdg.value,
                x_tick_mode_secondary=x_tick_secondary_wdg.value,
                show_marginals=bool(marginals_wdg.value),
            )
            display(fig)

    run_btn.on_click(_run)
    return widgets.VBox(
        [
            prompt_wdg,
            widgets.HBox([norm_wdg, topk_wdg, new_tokens_wdg, display_top_wdg]),
            widgets.HBox([add_special_wdg, do_sample_wdg, temperature_wdg, seed_wdg]),
            widgets.HBox([include_prompt_wdg, include_generated_wdg, max_layers_wdg, marginals_wdg]),
            widgets.HBox([start_wdg, end_wdg, layer_selection_wdg]),
            widgets.HBox([x_tick_wdg, x_tick_secondary_wdg, run_btn]),
            out,
        ]
    )


def build_payload_diff_widget(
    payload_path: str | Path | None = None,
    *,
    default_mode: str = "prompt",
):
    try:
        import ipywidgets as widgets
    except Exception as exc:  # pragma: no cover
        raise ImportError("ipywidgets is required for the notebook widget.") from exc

    prompt_tab = build_prompt_payload_widget(payload_path=payload_path if default_mode == "prompt" else None)
    generation_tab = build_generation_payload_widget(payload_path=payload_path if default_mode == "generation" else None)
    tabs = widgets.Tab(children=[prompt_tab, generation_tab])
    tabs.set_title(0, "Prompt LogitDiff")
    tabs.set_title(1, "Gen LogitDiff")
    tabs.selected_index = 0 if default_mode == "prompt" else 1
    return tabs


def build_model_loader_widget(
    *,
    default_model_name: str = "meta-llama/Llama-3.2-1B",
    default_prompt: str = "The capital of France is",
):
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception as exc:  # pragma: no cover
        raise ImportError("ipywidgets is required for the notebook widget.") from exc

    model_wdg = widgets.Text(
        value=default_model_name,
        description="Model:",
        layout=widgets.Layout(width="100%"),
    )
    prompt_wdg = widgets.Textarea(
        value=default_prompt,
        description="Prompt:",
        layout=widgets.Layout(width="100%", height="110px"),
    )
    device_wdg = widgets.Dropdown(
        options=[("auto", "auto"), ("cpu", "cpu")],
        value="auto",
        description="Device:",
    )
    load_btn = widgets.Button(description="Load Model", button_style="primary")
    out = widgets.Output()

    def _run(_):
        with out:
            out.clear_output(wait=True)
            print(f"Loading {model_wdg.value} ...")
            model, tokenizer = load_model_and_tokenizer(
                model_wdg.value,
                device_map=device_wdg.value,
            )
            display(
                build_single_model_widget(
                    model,
                    tokenizer,
                    default_prompt=prompt_wdg.value,
                )
            )

    load_btn.on_click(_run)
    return widgets.VBox([model_wdg, prompt_wdg, widgets.HBox([device_wdg, load_btn]), out])


def build_notebook_hub_widget(
    *,
    model=None,
    tokenizer=None,
    default_model_name: str = "meta-llama/Llama-3.2-1B",
    default_prompt: str = "The capital of France is",
    default_payload_path: str | Path | None = None,
):
    try:
        import ipywidgets as widgets
    except Exception as exc:  # pragma: no cover
        raise ImportError("ipywidgets is required for the notebook widget.") from exc

    single_tab = (
        build_single_model_widget(model, tokenizer, default_prompt=default_prompt)
        if model is not None and tokenizer is not None
        else build_model_loader_widget(
            default_model_name=default_model_name,
            default_prompt=default_prompt,
        )
    )
    prompt_tab = build_prompt_payload_widget(payload_path=default_payload_path)
    generation_tab = build_generation_payload_widget(payload_path=default_payload_path)
    notes = widgets.HTML(
        value=(
            "<b>Notes</b><br>"
            "Load a model first and use <code>Logit Lens</code> for direct single-model inspection.<br>"
            "Use <code>Prompt LogitDiff</code> and <code>Gen LogitDiff</code> with saved payloads built from your pairwise pipelines.<br>"
            "This notebook entry point mirrors the Tuned Lens workflow more closely by separating model loading from plotting controls."
        )
    )
    tabs = widgets.Tab(children=[single_tab, prompt_tab, generation_tab, notes])
    tabs.set_title(0, "Logit Lens")
    tabs.set_title(1, "Prompt LogitDiff")
    tabs.set_title(2, "Gen LogitDiff")
    tabs.set_title(3, "Notes")
    return tabs


__all__ = [
    "load_model_and_tokenizer",
    "make_prompt_wrapper",
    "make_generation_wrapper",
    "build_model_loader_widget",
    "build_prompt_logitdiff_widget",
    "build_generation_logitdiff_widget",
    "build_prompt_payload_widget",
    "build_generation_payload_widget",
    "build_notebook_hub_widget",
    "build_single_model_widget",
    "build_payload_diff_widget",
]
