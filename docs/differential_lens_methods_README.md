# Differential Lens Methods Plan

This README collects the methods discussed for a differential lens project.

It is intentionally **methods-only**. It does not decide which model families to use, where files should live in an existing project, or what experiments must be prioritized. It describes what each method requires, what it computes, what it can show, and what its limitations are.

---

## 1. Core Pairwise Setup

The basic object is a comparison between two related models:

\[
M_A, M_B
\]

For an input sequence or generation trajectory, collect hidden states:

\[
h^A_{\ell,t}, \quad h^B_{\ell,t}
\]

where:

- \(\ell\) is the layer;
- \(t\) is the token position;
- \(A\) and \(B\) denote the two compared models.

A lens/readout maps hidden states to vocabulary distributions:

\[
p^A_{\ell,t} = \mathrm{Lens}(h^A_{\ell,t})
\]

\[
p^B_{\ell,t} = \mathrm{Lens}(h^B_{\ell,t})
\]

The central object is the pairwise divergence:

\[
D_{\ell,t}(M_A, M_B)
\]

computed over hidden states, lens-induced distributions, or both.

---

## 2. Core Metrics

### 2.1 Hidden-State Metrics

#### Cosine distance

\[
D^{cos}_{\ell,t}
=
1 - \cos(h^A_{\ell,t}, h^B_{\ell,t})
\]

Use to measure angular representational drift.

#### L2 distance

\[
D^{L2}_{\ell,t}
=
\|h^A_{\ell,t} - h^B_{\ell,t}\|_2
\]

Use cautiously, since raw L2 can be dominated by residual norm differences.

#### Normalized L2

\[
D^{nL2}_{\ell,t}
=
\frac{
\|h^A_{\ell,t} - h^B_{\ell,t}\|_2
}{
\|h^A_{\ell,t}\|_2 + \epsilon
}
\]

Use when hidden-state norms differ substantially.

#### Norm ratio

\[
R_{\ell,t}
=
\frac{
\|h^B_{\ell,t}\|_2
}{
\|h^A_{\ell,t}\|_2 + \epsilon
}
\]

Use to diagnose whether observed differences are primarily scale effects.

---

### 2.2 Distributional Metrics

#### Jensen-Shannon divergence

\[
JSD(p^A_{\ell,t}, p^B_{\ell,t})
\]

Use as the main symmetric divergence between lens-induced predictive distributions.

#### Total variation distance

\[
TV(p^A_{\ell,t}, p^B_{\ell,t})
=
\frac{1}{2}
\sum_v
|p^A_{\ell,t}(v) - p^B_{\ell,t}(v)|
\]

Use to measure shifted probability mass.

#### Jaccard@k

\[
J@k =
\frac{
|TopK(p^A_{\ell,t}) \cap TopK(p^B_{\ell,t})|
}{
|TopK(p^A_{\ell,t}) \cup TopK(p^B_{\ell,t})|
}
\]

Often report divergence as:

\[
1 - J@k
\]

Use to measure overlap of top-k candidate tokens.

#### Top-1 agreement

\[
\mathbb{1}
[
\arg\max p^A_{\ell,t}
=
\arg\max p^B_{\ell,t}
]
\]

Use as a simple discrete agreement metric.

#### Reference-token probability shift

Let:

\[
y^* = \arg\max p^A_{L,t}
\]

where \(L\) is the final analyzed layer.

Track:

\[
p^B_{\ell,t}(y^*) - p^A_{\ell,t}(y^*)
\]

or the rank of \(y^*\) under \(p^B_{\ell,t}\).

Use to measure whether one model preserves the other model's preferred token.

---

## 3. Lens Types

### 3.1 Raw Lens

The raw lens directly applies the output head/unembedding to an intermediate hidden state:

\[
p^{Raw}_{\ell,t}
=
\mathrm{softmax}(W_U h_{\ell,t})
\]

#### Purpose

Use as the direct logit-lens baseline.

#### Known Issues

- residual norm drift;
- current-token leakage;
- missing final normalization;
- tied-embedding artifacts;
- unstable early-layer predictions.

#### Outputs

- Raw lens heatmaps;
- Raw lens layer profiles;
- Raw-vs-ModelNorm comparison plots.

---

### 3.2 ModelNorm Lens

The ModelNorm lens applies the model's final normalization before unembedding:

\[
p^{MN}_{\ell,t}
=
\mathrm{softmax}(W_U \mathrm{Norm}_{final}(h_{\ell,t}))
\]

#### Purpose

Use to test whether applying the model's own final normalization makes intermediate readouts more stable or meaningful.

#### Required Reporting

State whether final norm parameters are taken from:

1. each model's own final norm; or
2. a shared/reference model's final norm.

#### Outputs

- ModelNorm heatmaps;
- ModelNorm layer profiles;
- Raw-vs-ModelNorm scatter plots;
- ModelNorm-minus-Raw difference heatmaps.

---

### 3.3 Tuned Lens

A tuned lens learns an affine translator per layer:

\[
\tilde h_{\ell,t}
=
A_\ell h_{\ell,t} + b_\ell
\]

then decodes:

\[
p^{TL}_{\ell,t}
=
\mathrm{softmax}
(
W_U \mathrm{Norm}_{final}(\tilde h_{\ell,t})
)
\]

#### Purpose

Use as a readout-robustness check.

#### Required Design Choice

##### Self-readout

Each model is decoded with its own tuned lens.

Question:

> What does each model's own calibrated trajectory look like?

##### Reference-readout

Both models are decoded with the same tuned lens.

Question:

> Are hidden states from one model readable in the other model's calibrated readout coordinates?

These two settings answer different questions and should be labeled separately.

#### Outputs

- Raw vs ModelNorm vs Tuned layer profiles;
- Raw-vs-Tuned scatter plots;
- ModelNorm-vs-Tuned scatter plots;
- tuned-lens heatmaps;
- readout-robustness summary tables.

---

## 4. Prompt Lens Analysis

Prompt lens analysis compares two models on fixed prompts before free generation.

### Question

> Under the same prompt and token positions, where do two models differ internally and predictively?

### Required Inputs

For each prompt:

- tokenized prompt;
- matched token positions;
- hidden states per layer;
- lens logits/probabilities per layer and position.

### Computation

For each layer and position:

1. compute hidden-state divergence;
2. compute lens distributions;
3. compute probability-distribution divergence.

### Main Outputs

#### Heatmaps

Layer × position heatmaps for:

- JSD;
- TVD;
- \(1 - J@k\);
- cosine distance;
- normalized L2.

#### Layer Profiles

Layer on x-axis, aggregate divergence on y-axis.

Possible aggregations:

- mean over prompts and positions;
- median;
- max;
- prompt-bootstrap confidence intervals.

#### Scatter or Hexbin Plots

Useful relationships:

- hidden distance vs JSD;
- hidden distance vs TVD;
- Raw divergence vs ModelNorm divergence;
- intermediate-layer divergence vs final-layer divergence.

### Supported Claim

Prompt lens analysis supports claims about **pre-generation predictive drift**.

### Limitation

Prompt lens analysis alone does not establish semantic behavior across generated continuations.

---

## 5. Generation Lens Analysis

Generation lens analysis compares models along generated continuations.

### Question

> How do internal and predictive differences unfold during generation?

### Required Metadata

For every generation, record:

- prompt ID;
- template ID;
- system prompt;
- temperature;
- top-p/top-k;
- seed;
- max new tokens;
- generated text;
- generated token IDs;
- model identity;
- lens metrics per generated position;
- optional behavioral or semantic score.

### Trajectory Alignment Choices

#### Same-prefix forcing

Both models are evaluated on the same continuation.

Pros:

- directly matched positions.

Cons:

- may not reflect each model's natural generation.

#### Own-trajectory comparison

Each model is analyzed on its own generated output.

Pros:

- reflects natural generation.

Cons:

- token positions are not strictly matched.

#### Teacher-forced shared continuation

Both models are run on externally fixed continuations.

Pros:

- controlled comparison.

Cons:

- may miss natural generation dynamics.

The alignment rule must be stated.

### Main Outputs

- divergence-over-generation curves;
- layer × generated-position heatmaps;
- divergence vs behavioral/semantic score;
- selected trajectory examples;
- template/temperature sensitivity plots.

### Supported Claim

Generation lens analysis supports claims about **trajectory-level predictive drift** during generation.

---

## 6. Protocol Sensitivity Analysis

Protocol sensitivity analysis measures how internal divergence and output behavior depend on evaluation settings.

### Question

> How sensitive are internal divergence and output behavior to templates, system prompts, decoding settings, and seeds?

### Condition Definition

\[
c =
(
\text{template},
\text{system prompt},
\text{temperature},
\text{seed}
)
\]

### Required Inputs

For each condition:

- same prompt set;
- same model pair;
- recorded generation metadata;
- internal divergence metrics;
- optional semantic or behavioral scores.

### Computation

For each condition:

\[
D_c
=
\mathrm{mean}_{i,\ell,t}
D_{i,\ell,t,c}
\]

Also compute:

- variance across seeds;
- mean semantic/behavioral score;
- relation between internal divergence and behavior.

### Main Outputs

- template × temperature heatmap of internal divergence;
- template × temperature heatmap of behavioral/semantic score;
- seed-variance plot;
- internal divergence vs behavioral score scatter.

### Supported Claim

Protocol sensitivity analysis supports claims about **evaluation protocol sensitivity**.

---

## 7. Correlation, Confidence Interval, and p-value Analysis

Correlation analysis summarizes whether different divergence signals align.

It is not causal evidence.

---

### 7.1 Layer-vs-Final Correlation

For metric \(D_{i,t,\ell}\), aggregate within prompt:

\[
\bar D_{i,\ell}
=
\frac{1}{T_i}
\sum_t
D_{i,t,\ell}
\]

Then correlate each layer with the final analyzed layer \(L\):

\[
r_\ell
=
corr(
\bar D_{i,\ell},
\bar D_{i,L}
)
\]

Compute:

- Pearson \(r\);
- Spearman \(\rho\);
- bootstrap confidence intervals over prompts.

#### Outputs

- layerwise correlation curve;
- selected-layer scatter or hexbin plot;
- summary table.

#### Recommended Main Table Format

```text
Pair | Lens | Metric | Layer/Region | Pearson r [95% CI] | Spearman rho [95% CI]
```

Avoid main-text token-level p-values.

If reporting p-values:

- use prompt-level permutation;
- correct multiple comparisons when needed;
- label post-hoc strongest-layer results as descriptive unless split-half validated.

---

### 7.2 Hidden-vs-Predictive Correlation

Question:

> Does hidden-state divergence predict probability-distribution divergence?

For prompt-level aggregates:

\[
corr(
\bar D^h_{i,\ell},
\bar D^p_{i,\ell}
)
\]

where:

- \(D^h\) = cosine or normalized L2;
- \(D^p\) = JSD, TVD, or \(1-J@k\).

#### Outputs

- hidden distance vs JSD scatter/hexbin;
- Pearson/Spearman with bootstrap CIs;
- early/mid/late summaries if desired.

---

### 7.3 Raw-vs-ModelNorm Correlation

Question:

> Are divergence patterns robust to final-normalization readout?

Compute:

\[
corr(
D^{Raw}_{i,\ell},
D^{MN}_{i,\ell}
)
\]

#### Outputs

- Raw-vs-ModelNorm scatter;
- layer-profile rank correlation;
- ModelNorm-minus-Raw heatmap.

---

### 7.4 Strongest-Layer Selection

If reporting the strongest non-final layer, choose one of the following:

#### Descriptive selection

Report as post-hoc descriptive maximum.

#### Split-half selection

- use half of prompts to select strongest layer;
- use held-out half to estimate correlation and CI;
- optionally swap halves.

#### Fixed regions

Use pre-defined early/mid/late regions to avoid post-selection bias.

Recommended wording:

> Strongest-layer analyses are descriptive maxima selected after scanning layers. Confirmatory analyses use prompt-level aggregation and bootstrap confidence intervals.

---

## 8. Differential Logit-Prism Analysis

Differential logit prisms attribute model-pair logit differences to components.

### Question

> Which components account for the logit difference between two models?

### Target Requirement

A prism analysis requires a target.

It cannot target "misalignment" or "semantic change" unless that is operationalized as:

- a token;
- a margin;
- a token-set contrast;
- a response-derived contrast.

---

### 8.1 Single-Token Target

For target token \(v\):

\[
\Delta z_v
=
z^B_v - z^A_v
\]

Decompose:

\[
\Delta z_v
\approx
\sum_k
\Delta c_{k,v}
\]

where:

\[
\Delta c_{k,v}
=
c^B_{k,v}
-
c^A_{k,v}
\]

and \(k\) indexes components.

Possible components:

- residual stream;
- attention block;
- MLP block;
- attention head;
- subblock output.

---

### 8.2 Margin Target

For two tokens \(v_1, v_2\):

\[
\Delta z_{v_1-v_2}
=
(z^B_{v_1} - z^B_{v_2})
-
(z^A_{v_1} - z^A_{v_2})
\]

Use when the relevant behavior is a preference between alternatives.

---

### 8.3 Token-Set Contrast

For token sets \(A\) and \(B\):

\[
z_A
=
\log
\sum_{v \in A}
\exp(z_v)
\]

\[
\Delta z_{A-B}
=
(z^B_A - z^B_B)
-
(z^A_A - z^A_B)
\]

Use when the target is conceptual rather than a single token.

---

### 8.4 Reconstruction Check

Report reconstruction error:

\[
\epsilon
=
\left|
\Delta z_v
-
\sum_k
\Delta c_{k,v}
\right|
\]

or for a contrast:

\[
\epsilon
=
\left|
\Delta z_{A-B}
-
\sum_k
\Delta c_{k,A-B}
\right|
\]

If reconstruction error is high, attribution should be treated as unreliable.

### Main Outputs

- contribution-by-layer bar plot;
- attention-vs-MLP contribution plot;
- cumulative contribution curve;
- top-component table;
- reconstruction-error summary.

### Supported Claim

Differential prisms support component-level attribution for a chosen logit or contrast.

### Limitation

Differential prisms alone do not prove causality.

---

## 9. Activation Patching

Activation patching tests whether identified layers or components affect the output.

### Question

> If one model's activation is replaced with the other model's activation, does the output move accordingly?

### Base-to-Comparison Patch

\[
h^B_{\ell,t} \leftarrow h^A_{\ell,t}
\]

Measure whether model \(B\)'s output moves toward model \(A\).

### Comparison-to-Base Patch

\[
h^A_{\ell,t} \leftarrow h^B_{\ell,t}
\]

Measure whether model \(A\)'s output moves toward model \(B\).

### Metrics

- JSD to reference output;
- TVD to reference output;
- target logit shift;
- target margin shift;
- behavioral/semantic score if generation is involved.

Example recovery score:

\[
Recovery_{\ell}
=
D(p^B, p^A)
-
D(p^{patched}, p^A)
\]

Positive recovery means patching moved the model closer to the reference.

### Required Controls

- random layer;
- random position;
- low-divergence layer;
- shuffled-prompt patch;
- no-op patch.

### Main Outputs

- patch recovery by layer;
- selected-vs-control bar plot;
- target-logit movement plot;
- qualitative examples if generation changes.

### Supported Claim

Patching supports causal influence of the selected state or component on the measured output.

---

## 10. Residual-Delta Steering

Residual-delta steering derives a direction from model-pair differences.

### Direction Definition

For selected examples \(S\):

\[
v_\ell
=
\mathbb{E}_{(i,t)\in S}
[
h^B_{i,t,\ell}
-
h^A_{i,t,\ell}
]
\]

Inject into a model:

\[
h^A_{\ell,t}
\leftarrow
h^A_{\ell,t} + \alpha v_\ell
\]

or subtract from the comparison model:

\[
h^B_{\ell,t}
\leftarrow
h^B_{\ell,t} - \alpha v_\ell
\]

### Selection Sets

The set \(S\) must be defined.

Possible definitions:

- high-divergence positions;
- positions with large target-logit shift;
- positions identified by prism attribution;
- examples with a semantic/behavioral label;
- early divergence points;
- final-layer divergence peaks.

### Required Controls

- random vector;
- low-divergence vector;
- shuffled-label vector;
- opposite direction;
- layer sweep.

### Main Outputs

- steering-strength curve;
- layer sweep;
- selected-vector vs control-vector comparison.

### Supported Claim

Residual-delta steering tests whether model-pair difference directions can move predictions or behavior.

### Limitation

It does not identify a unique semantic direction unless validated across controls and contexts.

---

## 11. Concept and Branch Mining

Concept/branch mining is exploratory.

### Question

> Among high-divergence examples, are there recurring semantic, lexical, or behavioral patterns?

### Required Inputs

- high-divergence prompt/position/generation examples;
- changed tokens or top-k differences;
- generated text if using generation lens;
- optional embeddings or judge labels.

### Possible Computations

1. collect top-K high-divergence examples;
2. cluster prompts or generated completions;
3. extract discriminative tokens;
4. compare top shifted token sets;
5. label clusters manually or with an external classifier/judge;
6. identify branch points in generation where divergence sharply increases.

### Main Outputs

- cluster table;
- representative examples;
- top changed tokens per cluster;
- divergence trajectory around branch points.

### Supported Claim

Concept mining supports exploratory interpretation of divergence patterns.

### Limitation

Concept mining does not establish mechanism or causality.

---

## 12. Plot Inventory

### Heatmaps

Use for localization.

- layer × position JSD;
- layer × position TVD;
- layer × position hidden cosine distance;
- ModelNorm-minus-Raw difference heatmap;
- protocol-condition heatmaps if using generation settings.

### Layer Profiles

Use for summary.

- mean divergence by layer;
- max divergence by layer;
- early/mid/late region summaries;
- Raw vs ModelNorm vs Tuned overlays.

### Scatter / Hexbin Plots

Use for relationships.

- hidden distance vs JSD;
- hidden distance vs TVD;
- layer divergence vs final-layer divergence;
- Raw divergence vs ModelNorm divergence;
- tuned divergence vs ModelNorm divergence;
- internal divergence vs behavioral/semantic score.

### Prism Plots

Use for attribution.

- contribution-by-layer bar plot;
- cumulative contribution curve;
- MLP vs attention contribution plot;
- top-component table.

### Intervention Plots

Use for causality.

- patch recovery by layer;
- steering-strength curve;
- selected-vs-control intervention plot.

---

## 13. Statistical Reporting Rules

### Main Text

Prefer:

- effect sizes;
- prompt-level bootstrap CIs;
- descriptive heatmaps;
- prompt-level correlations.

Avoid emphasizing:

- token-level p-values;
- extremely small p-values from dependent observations;
- strongest-layer p-values selected post hoc.

### p-values

Use only if:

1. the unit is prompts, not token positions;
2. permutation or bootstrap test is used;
3. multiple comparisons are FDR-corrected where needed;
4. selected-layer analyses are clearly marked descriptive or estimated on held-out prompts.

### Recommended Wording

> We report prompt-level effect sizes and bootstrap confidence intervals. Token-position observations are not treated as independent for inferential testing. Strongest-layer analyses are descriptive unless the selection layer is chosen on a held-out split.

---

## 14. What Each Method Can Claim

| Method | Supports | Does not support |
|---|---|---|
| Pairwise lens divergence | where models differ | why they differ |
| Raw vs ModelNorm | readout sensitivity | true beliefs |
| Tuned lens | readout robustness | ground truth |
| Correlations | alignment/association | causality |
| Prompt lens | pre-generation drift | generated semantic behavior |
| Generation lens | trajectory drift | full behavioral explanation |
| Protocol sensitivity | evaluation dependence | universal behavior |
| Differential prisms | component attribution | causality alone |
| Patching | causal influence on output metric | broad semantic mechanism |
| Steering | directional influence | unique concept vector |
| Concept mining | exploratory patterns | mechanism |

---

## 15. Minimal Method Stack

The smallest coherent method stack is:

1. pairwise hidden and lens divergence;
2. Raw and ModelNorm readouts;
3. layer-position heatmaps;
4. layer profiles;
5. prompt-level correlations with bootstrap CIs.

This gives a descriptive differential lens analysis.

---

## 16. Extended Method Stack

Optional extensions:

1. tuned lens comparison;
2. generation lens;
3. protocol sensitivity;
4. differential logit-prism attribution;
5. activation patching;
6. residual-delta steering;
7. concept/branch mining.

Each extension has separate requirements and claims.

---

## 17. Method Checklist

### Lens Metrics

- [ ] token positions are aligned;
- [ ] lens type is labeled;
- [ ] normalization choice is explicit;
- [ ] vocabulary compatibility is handled;
- [ ] top-k uses a stated value of \(k\).

### Correlations

- [ ] prompt-level aggregation used;
- [ ] bootstrap over prompts;
- [ ] p-values not overinterpreted;
- [ ] selected layers marked descriptive or split-half validated;
- [ ] scatter/hexbin plotted for surprising correlations.

### Generation Lens

- [ ] template recorded;
- [ ] temperature recorded;
- [ ] seed recorded;
- [ ] trajectory alignment rule stated;
- [ ] behavioral/semantic score source stated.

### Differential Prisms

- [ ] target token/contrast defined;
- [ ] decomposition reconstructs logit difference;
- [ ] component indexing documented;
- [ ] plots show contribution signs and magnitudes.

### Patching / Steering

- [ ] target metric defined;
- [ ] controls included;
- [ ] intervention layer/position selected transparently;
- [ ] effect size reported, not just examples.

---

## 18. Summary

The framework consists of method modules, not model-specific claims.

The core method measures pairwise layer-position divergence in hidden states and lens-induced predictive distributions.

Correlation analysis summarizes alignment.

Raw/ModelNorm/Tuned comparisons test readout sensitivity.

Generation lens analysis extends the method to generated trajectories.

Protocol sensitivity measures dependence on evaluation settings.

Differential logit prisms attribute selected logit differences to components.

Patching and steering test whether identified states or directions influence outputs.

Concept mining is exploratory and should be labeled as such.
