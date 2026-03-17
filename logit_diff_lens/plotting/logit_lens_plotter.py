from typing import Optional, List, Dict, Any
import torch
import numpy as np
from matplotlib import cm, colors
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from ..wrapper.arch_wrapper import ArchWrapper
from ..ldl.apply_logit_lens_prompt import apply_logit_lens_plotter



# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _make_layer_names(
    layer_indices: List[int] | range,
    force_include_input: bool = True,
    force_include_output: bool = True,
) -> List[str]:

    if not layer_indices:
        return []

    names = []
    layer_indices = sorted(set(layer_indices))

    # --- Embedding ---
    if force_include_input and -1 in layer_indices:
        names.append("Embedding")

    # --- Core transformer layers ---
    core_layers = [li for li in layer_indices if li >= 0]
    if core_layers:
        max_core = max(core_layers)
        # Exclude the max index for now if it’s reserved for output
        core = [li for li in core_layers if li < max_core or not force_include_output]
        for li in sorted(core):
            names.append(f"Layer {li}")

        # --- Output projection ---
        if force_include_output and max_core in layer_indices:
            names.append("Output")

    elif force_include_output and len(layer_indices) > 1:
        # Edge case: no transformer blocks, only embedding + output
        names.append("Output")

    return names


def _plot_logit_lens(
    data: Dict,
    metric: str = "jaccard",
    mode: str = "raw",
    topk: int = 5,
    force_include_input: bool = False,
    force_include_output: bool = False,
    block_steps: int = 1,
    start_idx: int = None,
    end_idx: int = None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    auto_vmin_vmax: bool = False,
    cmap: str | None = None,
    fig_width: int = None,
    fig_height: int = None,
    mark_correct_preds: bool = True,
    show_marginals: bool = True,
) -> go.Figure:


    # ------------------------------------------------------------------
    # Metric configs
    # ------------------------------------------------------------------
    METRIC_CONFIGS = {
        # Logits and Probs
        f"logits_mean_{mode}": {"type": "distribution", "cmap": "Blues", "title": "Logits", "label": "Logits", "vmin": 0, "vmax": 1},
        f"logits_std_{mode}": {"type": "distribution", "cmap": "Blues", "title": "Logits Std.", "label": "Logits Std.", "vmin": 0, "vmax": 1},
        f"logit_margin_{mode}": {"type": "distribution", "cmap": "Blues", "title": "Topk-2 Logit Margin", "label": "Logit", "vmin": 0, "vmax": 1},
        f"max_prob_{mode}": {"type": "distribution", "cmap": "Blues", "title": "Probs", "label": "Probs", "vmin": 0, "vmax": 1},
        f"probs_std_{mode}": {"type": "distribution", "cmap": "Blues", "title": "Probs Std.", "label": "Probs Std.", "vmin": 0, "vmax": 1},
        f"gt_prob_{mode}": {"type": "distribution", "cmap": "Blues", "title": "Ground Truth Probability Assignment", "label": "GT Prob", "vmin": 0, "vmax": 1},
        f"entropy_{mode}": {"type": "distribution", "cmap": "Reds", "title": "Entropy", "label": "Entropy", "vmin": 0, "vmax": 10},
        f"perplexity_{mode}": {"type": "distribution", "cmap": "RdBu_r", "title": "Perplexity", "label": "Perplexity", "vmin": -1000, "vmax": 1000},
        f"gt_prob_diff_{mode}": {"type": "distribution", "cmap": "RdBu_r", "title": "Ground Truth Probability Difference", "label": "ΔGT Prob", "vmin": 0, "vmax": 1},
        # KL Div
        f"kl_div_prev_{mode}": {"type": "distribution", "cmap": "Reds", "title": "Kullback-Lieber Divergence (Current Layer‖Previous Layer)", "label": "KL Div", "vmin": 0, "vmax": 10},
        f"kl_div_last_{mode}": {"type": "distribution", "cmap": "Reds", "title": "Kullback-Lieber Divergence (Current Layer‖Last Layer)", "label": "KL Div", "vmin": 0, "vmax": 10},
        # JSD
        f"js_div_prev_{mode}": {"type": "distribution", "cmap": "Blues", "title": "Jensen–Shannon Divergence (Current Layer||Previous Layer)", "label": "JSD", "vmin": 0, "vmax": 1},
        f"js_div_last_{mode}": {"type": "distribution", "cmap": "Blues", "title": "Jensen–Shannon Divergence (Current Layer||Last Layer)", "label": "JSD", "vmin": 0, "vmax": 1},
        # Hidden-state Cosine Similarity
        f"cos_sim_prev_{mode}": {"type": "distribution", "cmap": "RdBu_r", "title": "Hidden States Cosine Similarity (Current Layer, Previous Layer)", "label": "Cos Sim", "vmin": 0, "vmax": 1},
        f"cos_sim_last_{mode}": {"type": "distribution", "cmap": "RdBu_r", "title": "Hidden States Cosine Similarity (Current Layer, Last Layer)", "label": "Cos Sim", "vmin": 0, "vmax": 1},
        # Hidden-state L2 Distance
        f"l2_dist_prev_{mode}": {"type": "distribution", "cmap": "Reds", "title": "Hidden States L2 Distance (Current Layer, Previous Layer)", "label": "L2", "vmin": 0, "vmax": 1000},
        f"l2_dist_last_{mode}": {"type": "distribution", "cmap": "Reds", "title": "Hidden States L2 Distance (Current Layer, Last Layer)", "label": "L2", "vmin": 0, "vmax": 1000},
        # Token overlap / set metrics
        f"jaccard_prev_{mode}": {"type": "overlap", "cmap": "Greens", "title": f"Top-{topk} Jaccard(Current Layer, Previous Layer)", "label": "IoU", "vmin": 0, "vmax": 1},
        f"jaccard_last_{mode}": {"type": "overlap", "cmap": "Greens", "title": f"Top-{topk} Jaccard(Current Layer, Last Layer)", "label": "IoU", "vmin": 0, "vmax": 1},
        # Prediction correctness
        f"accuracy_topk_{mode}": {"type": "accuracy", "title": f"Top-{topk} Correct Next Token Predictions", "cmap": "PRGn", "label": f"Top-{topk} Accuracy", "vmin": 0, "vmax": 1},
    }

    # Build metric key
    if f"{metric}_{mode}" in data["metrics"]:
        metric_key = f"{metric}_{mode}"
    else:
        metric_key = metric
    if metric_key not in data["metrics"]:
        raise KeyError(
            f"Metric '{metric_key}' not found in data['metrics']. "
            f"Available: {list(data['metrics'].keys())}"
        )

    # ------------------------------------------------------------------
    # Metric matrix (supports keyed metrics)
    # ------------------------------------------------------------------
    cfg = METRIC_CONFIGS.get(metric_key, {})
    cmap = cmap or cfg.get("cmap", "RdBu_r")
    metric_label = cfg.get("label", metric)
    metric_title = cfg.get("title", metric_label)

    metric_map = data["metrics"][metric_key]
    metric_layers = sorted(metric_map.keys(), key=lambda x: x[0])
    M = torch.stack([metric_map[k] for k in metric_layers]).numpy()
    all_layers = [k[0] for k in metric_layers]

    # ------------------------------------------------------------------
    # Apply block_steps layer collapsing
    # ------------------------------------------------------------------
    if block_steps > 1:
        core_layers = [li for li in all_layers if li >= 0]
        if core_layers:
            reduced = core_layers[::block_steps]
            if core_layers[-1] not in reduced:
                reduced.append(core_layers[-1])
            if force_include_input and -1 in all_layers:
                reduced = [-1] + reduced
            if force_include_output and max(all_layers) not in reduced:
                reduced.append(max(all_layers))
            all_layers = sorted(reduced)

        # Filter M accordingly (only keep selected layers)
        layer_idx_map = {li: idx for idx, li in enumerate([k[0] for k in metric_layers])}
        selected_indices = [layer_idx_map[li] for li in all_layers if li in layer_idx_map]
        M = M[selected_indices, :]

    n_layers, n_pos = M.shape
    if auto_vmin_vmax:
        finite_vals = M[np.isfinite(M)]
        vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
    else:
        vmin = vmin if vmin is not None else cfg.get("vmin", np.nanmin(M))
        vmax = vmax if vmax is not None else cfg.get("vmax", np.nanmax(M))

    # ------------------------------------------------------------------
    # Tokens + layers
    # ------------------------------------------------------------------
    preds = data["topk_preds"]
    # Ensure we only use the layers already present in metrics
    mode = next(iter({k[1] for k in preds.keys()}), mode)
    layers = [preds[(li, mode)] for li in all_layers if (li, mode) in preds]

    layer_labels = _make_layer_names(
        layer_indices=all_layers,
        force_include_input=force_include_input,
        force_include_output=force_include_output,
    )

    # ------------------------------------------------------------------
    # Reverse order for display (top = output)
    # ------------------------------------------------------------------
    M = M[::-1, :]
    layers = layers[::-1]
    layer_labels = layer_labels[::-1]

    # ------------------------------------------------------------------
    # Token cleanup helpers
    # ------------------------------------------------------------------
    def clean_token(t):
        return str(t).replace("Ġ", " ").replace("▁", " ").strip()

    def truncate_text(text, max_len=12):
        return text if len(text) <= max_len else text[:max_len - 1] + "…"
    
    def shorten_list_str(lst, max_len=80):
        s = ", ".join(lst)
        return s if len(s) <= max_len else s[:max_len - 3] + "..."


    input_tokens = [clean_token(t) for t in data.get("tokens", [])[:n_pos]]
    target_tokens = [clean_token(t) for t in data.get("target_tokens", [])[:n_pos]]

    # ------------------------------------------------------------------
    # Optional positional slicing
    # ------------------------------------------------------------------
    n_positions_total = M.shape[1]

    if start_idx is None:
        start_idx = 0
    if end_idx is None or end_idx > n_positions_total:
        end_idx = n_positions_total
    if end_idx - start_idx < 1:
        end_idx = min(start_idx + 1, n_positions_total)

    # Slice metric matrix
    M = M[:, start_idx:end_idx]

    # Slice token lists
    input_tokens = input_tokens[start_idx:end_idx]
    target_tokens = target_tokens[start_idx:end_idx]

    # Slice per-layer top-k predictions correctly
    layers = [layer[start_idx:end_idx] for layer in layers]

    # Update position count
    n_pos = end_idx - start_idx
    print(f"[LogitLens] Showing positions {start_idx}:{end_idx} (n={n_pos})")
    
    # ------------------------------------------------------------------
    # Structured text with adaptive color
    # ------------------------------------------------------------------
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap if isinstance(cmap, str) else "RdBu_r")
    cell_text = np.empty_like(M, dtype=object)
    hovertext = np.empty_like(M, dtype=object)

    for li, layer_preds in enumerate(layers):
        for t in range(n_pos):
            # Top-k predictions for this layer/position
            top_tokens = [clean_token(tok) for tok in layer_preds[t][:topk]]
            top1 = top_tokens[0] if top_tokens else "—"
            true_tok = clean_token(target_tokens[t]) if t < len(target_tokens) else ""

            # Color contrast handling
            rgba = colormap(norm(M[li, t]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if luminance < 0.45 else "black"

            # Text in cell = top-1 prediction
            cell_text[li, t] = (
                f"<span style='color:{color}'><b>{truncate_text(top1)}</b></span>"
            )

            # Hover shows full top-k list and GT token
            topk_str = ", ".join(top_tokens)
            hovertext[li, t] = (
                f"<b>Input:</b> {input_tokens[t]}<br>"
                f"<b>Target:</b> {true_tok}<br>"
                f"<b>Top-1:</b> {top1}<br>"
                f"<b>Top-{topk}:</b> {shorten_list_str(top_tokens)}"
            )

    # ------------------------------------------------------------------
    # Layout setup
    # ------------------------------------------------------------------
    mean_per_pos = np.nanmean(M, axis=0)
    mean_per_layer = np.nanmean(M, axis=1)

    if show_marginals:
        fig = make_subplots(
            rows=2, cols=2,
            row_heights=[0.08, 0.92], 
            column_widths=[0.86, 0.14],
            specs=[[{"type": "xy"}, None],
                   [{"type": "heatmap"}, {"type": "xy"}]],
            horizontal_spacing=0.015,
            vertical_spacing=0.02,
        )
        main_row, main_col = 2, 1
    else:
        fig = make_subplots(rows=1, cols=1)
        main_row, main_col = 1, 1

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------
    fig.add_trace(go.Heatmap(
        z=M,
        text=cell_text,               
        hovertext=hovertext,           
        hoverinfo="text",
        texttemplate="%{text}",
        textfont={"size": 11},
        colorscale=cmap,
        zmin=vmin, zmax=vmax,
        colorbar=dict(
            title=metric_label,
            orientation="h",
            yanchor="top",
            y=-0.05,
            len=0.7,
            x=0.46,
            xanchor="center",
        ),
    ), row=main_row, col=main_col)

    # ------------------------------------------------------------------
    # Overlay correctness borders
    # ------------------------------------------------------------------
    if mark_correct_preds:
        cell_shapes = []
        cell_spacing = 0.02
        hm_trace = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        xref, yref = hm_trace.xaxis, hm_trace.yaxis

        for li, layer_preds in enumerate(layers):
            for t in range(n_pos):
                true_tok = clean_token(target_tokens[t]) if t < len(target_tokens) else ""
                top_tokens = [clean_token(tok) for tok in layer_preds[t][:topk]]
                top1_correct = (len(top_tokens) > 0 and top_tokens[0] == true_tok)
                topk_correct = (true_tok in top_tokens)

                if top1_correct:
                    dash, color = "solid", "black"
                elif topk_correct:
                    dash, color = "solid", "gray"
                else:
                    continue

                x0, x1 = t - 0.5 + cell_spacing, t + 0.5 - cell_spacing
                y0, y1 = li - 0.5 + cell_spacing, li + 0.5 - cell_spacing
                cell_shapes.append(dict(
                    type="rect",
                    xref=xref, yref=yref,
                    x0=x0, x1=x1, y0=y0, y1=y1,
                    line=dict(color=color, width=2, dash=dash),
                    fillcolor="rgba(0,0,0,0)",
                    layer="above",
                ))
        fig.update_layout(shapes=cell_shapes)


    # ------------------------------------------------------------------
    # Marginals and axes
    # ------------------------------------------------------------------
    if show_marginals:
        # Top marginal (mean per position)
        fig.add_trace(
            go.Bar(
                x=list(range(n_pos)),
                y=mean_per_pos,
                marker_color="lightgray",
                hovertext=[f"<b>Pos {t}</b><br>Mean {metric_label}: {v:.3f}<br>n={n_layers}"
                        for t, v in enumerate(mean_per_pos)],
                hoverinfo="text",
                showlegend=False,
            ),
            row=1, col=1,
        )

        # Side marginal (mean per layer)
        fig.add_trace(
            go.Bar(
                x=mean_per_layer,
                y=list(range(n_layers)),
                orientation="h",
                marker_color="lightgray",
                hovertext=[
                    f"<b>{layer_labels[i]}</b><br>Mean {metric_label}: {v:.3f}<br>n={n_pos}"
                    for i, v in enumerate(mean_per_layer)
                ],
                hoverinfo="text",
                showlegend=False,
            ),
            row=2, col=2,
        )

        # Ensure side marginal matches heatmap vertical order
        fig.update_yaxes(
            tickvals=list(range(n_layers)),
            ticktext=layer_labels,
            autorange="reversed",    
            row=2, col=2,
        )

        # Adjust marginal plot proportions for better alignment
        fig.update_layout(
            margin=dict(l=80, r=80, t=100, b=100),
        )
        fig.update_yaxes(
            title_text=f"Mean {metric_label}",
            title_standoff=10,
            title_font=dict(size=12),
            row=1, col=1,
        )
        fig.update_xaxes(
            title_text=f"Mean {metric_label}",
            title_standoff=10,
            title_font=dict(size=12),
            row=2, col=2,
        )

        fig.update_layout(
            grid=dict(rows=2, columns=2),
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            visible=True,
            title_font=dict(size=12),
            row=1, col=1,
        )
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_xaxes(visible=False, row=2, col=2)
        fig.update_yaxes(visible=False, row=2, col=2)

        # Re-align token arrows along bottom x-axis
        arrows = [f"{i} → {j}" for i, j in zip(input_tokens, target_tokens)]
        fig.update_xaxes(
            tickvals=list(range(n_pos)),
            ticktext=arrows,
            side="bottom",
            row=main_row, col=main_col,
        )

    else:
        fig.update_xaxes(
            tickvals=list(range(n_pos)),
            ticktext=input_tokens,
            side="bottom",
            row=main_row, col=main_col,
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(n_pos)),
                y=[None] * n_pos,
                xaxis="x2",
                mode="markers",
                marker_opacity=0,
                showlegend=False,
            )
        )
        fig.update_layout(
            xaxis2=dict(
                overlaying="x",
                side="top",
                tickvals=list(range(n_pos)),
                ticktext=target_tokens,
                showgrid=False,
                zeroline=False,
            )
        )

    # Y-axis for layers
    fig.update_yaxes(
        tickvals=list(range(n_layers)),
        ticktext=layer_labels,
        autorange="reversed",
        title="",
        row=main_row, col=main_col,
    )

    fig.update_layout(
        title=title or metric_title,
        font=dict(family="Noto Sans, DejaVu Sans", size=14),
        width=max(1000, n_pos * 70) if fig_width is None else fig_width,
        height=max(750, n_layers * 60) if fig_height is None else fig_height,
        margin=dict(l=80, r=80, t=110, b=110),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # ------------------------------------------------------------------
    # Hover behavior and responsive layout
    # ------------------------------------------------------------------
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font=dict(color="black", size=12),
            align="left",
        ),
        hovermode="closest",
        hoverdistance=5,
    )

    fig.show(config=dict(displayModeBar=False, responsive=True))
    return fig




# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def plot_logit_lens(
    arch_wrapper:"ArchWrapper",
    prompt:str,
    norm_mode: str = "raw", # ("raw", "unit_norm", "eps_norm", "model_norm")
    add_special_tokens: bool = False,
    topk: int = 1,
    force_include_input: bool = True,
    force_include_output: bool = True,
    mark_correct_preds: bool = True,
    show_marginals: bool = False,
    block_steps: int = 1,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    cmap: str = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    auto_vmin_vmax: bool = False,
    fig_width: int = None,
    fig_height: int = None,
    logits_std: bool = False,
    logit_margin: bool = False,
    probs: bool = False,
    probs_std: bool = False,
    ground_truth_probs: bool = False,
    entropy: bool = False,
    perplexity: bool = False,
    kl_div_prev: bool = False,
    kl_div_last: bool = False,
    js_div_prev: bool = False,
    js_div_last: bool = False,
    cos_sim_prev: bool = False,
    cos_sim_last: bool = False,
    l2_dist_prev: bool = False,
    l2_dist_last: bool = False,
    jaccard_prev: bool = False,
    jaccard_last: bool = False,
    topk_accuracy: bool = False
) -> None|go.Figure|Any:

    
    metric = (
        "logits_std" if logits_std else
        "logit_margin" if logit_margin else
        "max_prob" if probs else
        "probs_std" if probs_std else
        "gt_prob" if ground_truth_probs else
        "entropy" if entropy else
        "perplexity" if perplexity else
        "kl_div_prev" if kl_div_prev else
        "kl_div_last" if kl_div_last else
        "js_div_prev" if js_div_prev else
        "js_div_last" if js_div_last else
        "cos_sim_prev" if cos_sim_prev else
        "cos_sim_last" if cos_sim_last else
        "l2_dist_prev" if l2_dist_prev else
        "l2_dist_last" if l2_dist_last else
        "jaccard_prev" if jaccard_prev else
        "jaccard_last" if jaccard_last else
        "accuracy_topk" if topk_accuracy else
        "logits_mean"
    )

    # Run the collector
    results = apply_logit_lens_plotter(
        arch_wrapper=arch_wrapper,
        prompt=prompt,
        norm_mode=norm_mode,
        topk=topk,
        add_special_tokens=add_special_tokens,
        force_include_input=force_include_input,
        force_include_output=force_include_output,
    )

    metric_key = f"{metric}_{norm_mode}" if f"{metric}_{norm_mode}" in results["metrics"] else metric

    # Plot results
    fig = _plot_logit_lens(
        data=results,
        metric=metric_key,
        mode=norm_mode,
        cmap=cmap,
        topk=topk,
        force_include_input=force_include_input,
        force_include_output=force_include_output,
        block_steps=block_steps,
        start_idx=start_idx,
        end_idx=end_idx,
        title=title,
        vmin=vmin,
        vmax=vmax,
        auto_vmin_vmax=auto_vmin_vmax,
        fig_width=fig_width,
        fig_height=fig_height,
        mark_correct_preds=mark_correct_preds,
        show_marginals=show_marginals
    )

    if save_path is not None:
        #fig.write_html("heatmap.html")
        #fig.write_image("heatmap.png")
        fig.write_html(save_path)
        print(f"[SAVED HTML] {save_path}")
    
    return fig