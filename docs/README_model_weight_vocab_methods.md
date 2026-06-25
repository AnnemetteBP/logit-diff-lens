# Model Weight and Vocabulary-Space Interpretability Methods

This README summarizes three related methods for interpreting transformer language models by projecting weights, directions, or residual-stream signals into vocabulary space.

It also serves as a framework-alignment note for this repository, where these methods should be implemented as reusable analysis families tied to canonical artifacts rather than as isolated experiments.

## Source papers / posts

1. **The Singular Value Decompositions of Transformer Weight Matrices are Highly Interpretable**  
   Beren Millidge / beren and Sid Black, 2022  
   Main URL: https://www.alignmentforum.org/posts/mkbGjzxD8d8XqKHzA/the-singular-value-decompositions-of-transformer-weight  
   Mirror: https://www.lesswrong.com/posts/mkbGjzxD8d8XqKHzA/the-singular-value-decompositions-of-transformer-weight

2. **Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space**  
   Mor Geva, Avi Caciularu, Kevin Ro Wang, Yoav Goldberg, EMNLP 2022  
   ACL Anthology page: https://aclanthology.org/2022.emnlp-main.3/  
   PDF: https://aclanthology.org/2022.emnlp-main.3.pdf  
   arXiv: https://arxiv.org/abs/2203.14680

3. **Spectral Filters, Dark Signals, and Attention Sinks**  
   Nicola Cancedda, ACL 2024  
   ACL Anthology page: https://aclanthology.org/2024.acl-long.263/  
   PDF: https://aclanthology.org/2024.acl-long.263.pdf  
   arXiv: https://arxiv.org/abs/2402.09221  
   Meta AI publication page: https://ai.meta.com/research/publications/spectral-filters-dark-signals-and-attention-sinks/

---

## Common setup and definitions

Let the residual-stream vector at some token position be:

```text
h in R^{d_model}
```

Let the unembedding matrix be:

```text
W_U in R^{d_model x |V|}
```

where `|V|` is the vocabulary size.

The final logits are:

```text
logits = h @ W_U
```

This same projection can be used on intermediate vectors or model-internal directions:

```text
vocab_scores = direction @ W_U
```

The top-scoring tokens are interpreted as the vocabulary meaning of that residual-space direction.

### Key terms

- **Residual stream:** the main vector space passed through the transformer. Attention heads and MLPs read from and write to this space.
- **Embedding matrix (`W_E`):** maps token IDs into residual-space vectors.
- **Unembedding matrix (`W_U`):** maps final residual-space vectors into vocabulary logits.
- **Vocabulary projection:** projecting a residual-space vector through `W_U` and inspecting top or bottom tokens.
- **Logit lens:** projecting intermediate residual states through `W_U` to see what token distribution the model would predict at that point.
- **SVD:** singular value decomposition. For a matrix `M`,

```text
M = U Σ V^T
```

where the singular vectors in `U` and `V` identify important directions of the transformation.

---

## Framework alignment

For this repository, these methods should be split into three implementation categories.

### 1. Static model-weight methods

These methods operate directly on model parameters and do not require a prompt capture artifact.

Examples:

- SVD of attention OV matrices
- SVD of MLP `W_in` / `W_out`
- SVD of `W_U`
- SVD of `W_E`

These methods consume:

- model weights
- tokenizer
- architecture-aware weight extraction

These methods produce:

- reusable weight-direction artifacts
- singular spectra
- vocabulary-projected token summaries

### 2. Dynamic activation-conditioned methods

These methods depend on specific prompts, tokens, or residual/component activations.

Examples:

- FFN sub-update analysis
- prism-style component analysis
- spectral-band occupancy of actual residual vectors
- hidden-delta vocabulary projection

Whenever possible, these methods should consume the same canonical forward capture artifacts used elsewhere in the framework:

- tokenization
- residual hidden states
- optional subblock outputs
- backend/runtime metadata

This ensures that multiple reported analyses can be derived from the same saved hidden-state source of truth.

### 3. Low-rank / factor-analysis methods

These methods cut across both static and dynamic settings.

Examples:

- SVD of weight matrices
- SVD of hidden-state deltas
- SVD of logit diffs
- low-rank adapter analysis
- future mixture-of-low-rank and MoE factor analysis

These methods may consume:

- model weights
- forward capture artifacts
- comparison artifacts

and should produce:

- factor artifacts
- singular-value summaries
- vocabulary projections of factors
- intervention or compression summaries

## Repository-level implementation rule

Each method family should explicitly document:

- whether it is static or dynamic
- whether it depends on canonical forward capture artifacts
- whether it requires subblock outputs
- whether it consumes model weights directly
- which primary space its objects live in:
  - weight space
  - residual space
  - vocab/logit space
  - activation-conditioned space
- which reusable artifact it produces

---

# 1. SVD of transformer weight matrices

Source: https://www.alignmentforum.org/posts/mkbGjzxD8d8XqKHzA/the-singular-value-decompositions-of-transformer-weight

## Core idea

Take SVDs of transformer component weight matrices, especially attention OV circuits and MLP input/output matrices. Then project residual-space singular vectors into vocabulary space and inspect the top tokens.

This often reveals interpretable semantic directions.

### Framework classification

- static model-weight method
- low-rank / factor-analysis method

### Primary spaces

- weight space
- residual-space singular directions
- vocab-space projections of those directions

### Framework inputs

- model weights
- tokenizer
- unembedding matrix
- architecture-aware extraction of attention and MLP matrices

### Framework outputs

- weight-direction artifacts
- singular spectra
- top/bottom token summaries
- optional edited-weight artifacts

## Objects analyzed

### Attention OV circuit

For an attention head, define the OV circuit as the composition of the value and output matrices:

```text
W_OV = W_V @ W_O
```

Depending on library conventions, this may need transposes.

The OV matrix tells you what the head can write into the residual stream, ignoring the attention pattern.

### MLP input and output matrices

For an MLP block, typical matrices are:

```text
W_in   maps residual stream -> hidden MLP dimension
W_out  maps hidden MLP dimension -> residual stream
```

The SVD method can be applied to both.

## Method

For each matrix `M`:

```text
U, S, Vh = svd(M)
```

Then identify which singular-vector side lives in residual-stream space. That is the side with dimension `d_model`.

For a residual-space singular vector `v_i`:

```text
vocab_scores_i = v_i @ W_U
```

Inspect:

```text
top_tokens    = top_k(vocab_scores_i)
bottom_tokens = bottom_k(vocab_scores_i)
```

The top and bottom tokens often correspond to opposite semantic or syntactic directions.

## Rank-one deletion / editing

To remove one singular direction from a matrix:

```text
M_without_i = M - S[i] * outer(U[:, i], Vh[i, :])
```

This deletes the rank-one component associated with singular value `S[i]`.

## Practical implementation plan

1. Load a transformer model with accessible weights, for example GPT-2.
2. Extract `W_U`, attention `W_V`, attention `W_O`, MLP `W_in`, and MLP `W_out`.
3. For each layer and each attention head, compute `W_OV`.
4. Run SVD on each selected matrix.
5. Find the singular-vector side with shape `d_model`.
6. Project those singular vectors through `W_U`.
7. Save top and bottom tokens for each singular direction.
8. Assign human or LLM-generated labels to directions.
9. Optionally perform rank-one deletion and measure logit changes.
10. Build a searchable atlas of component directions.

## Minimal PyTorch sketch

```python
import torch

# M: matrix to analyze
# W_U: unembedding matrix with shape [d_model, vocab_size]
# tokenizer: model tokenizer

U, S, Vh = torch.linalg.svd(M, full_matrices=False)

# Example assumes rows of Vh are residual-space directions.
# Check shapes before using this.
for i in range(20):
    direction = Vh[i]
    scores = direction @ W_U
    top_ids = torch.topk(scores, k=20).indices.tolist()
    bottom_ids = torch.topk(-scores, k=20).indices.tolist()

    print("direction", i, "singular value", S[i].item())
    print("top", tokenizer.convert_ids_to_tokens(top_ids))
    print("bottom", tokenizer.convert_ids_to_tokens(bottom_ids))
```

---

# 2. FFN sub-updates in vocabulary space

Source: https://aclanthology.org/2022.emnlp-main.3/

PDF: https://aclanthology.org/2022.emnlp-main.3.pdf

## Core idea

A feed-forward network layer can be viewed as a key-value memory. Its output is a sum of many value-vector sub-updates. Each value vector can be projected into vocabulary space to see which concepts or tokens it promotes.

This method is more activation-aware than the pure SVD method because it asks which FFN value vectors are active for a specific input.

### Framework classification

- dynamic activation-conditioned method

### Primary spaces

- residual space
- activation-conditioned sub-updates
- vocab-space projections of static and dynamic FFN contributions

### Framework inputs

- canonical forward capture artifacts when possible
- optional saved FFN subblock outputs
- model weights for FFN matrix extraction
- tokenizer

### Framework outputs

- per-prompt FFN contribution artifacts
- static value-vector vocab atlas
- dynamic promoted-token summaries

## FFN decomposition

For an FFN layer, simplified:

```text
FFN(x) = f(x @ W_K) @ W_V
```

Equivalently:

```text
FFN(x) = sum_i m_i v_i
```

where:

```text
m_i = f(x · k_i)
```

and:

```text
v_i = value vector i
```

Each term:

```text
m_i v_i
```

is a sub-update to the residual stream.

## Static vocabulary projection

Project each value vector into vocabulary space:

```text
scores_i = v_i @ W_U
```

The top tokens show what that value vector tends to promote.

## Dynamic vocabulary projection

For a specific context, include the coefficient:

```text
dynamic_scores_i = (m_i * v_i) @ W_U
```

This shows what the value vector is promoting in that context.

## Dominant sub-updates

For a given input and layer:

1. Compute all coefficients `m_i`.
2. Rank value vectors by contribution size or by effect on a target token.
3. Inspect the top contributors.
4. Project those contributors into vocabulary space.

## Promotion vs. elimination

The paper studies whether FFN layers mostly:

- **promote** likely target concepts/tokens, or
- **eliminate** previously likely candidates.

The paper argues that FFN updates are mostly constructive: they build predictions by promoting concepts.

## Practical implementation plan

1. Load a transformer model and tokenizer.
2. Hook into the FFN layers.
3. For each layer, extract the first FFN matrix and second FFN matrix.
4. Treat the first matrix as keys and the second matrix as values.
5. Run the model on a dataset of prompts.
6. For each token position and layer, compute FFN hidden activations `m_i`.
7. Decompose the FFN output into `m_i v_i` terms.
8. Project static value vectors `v_i @ W_U`.
9. Project dynamic sub-updates `(m_i v_i) @ W_U`.
10. Record top promoted tokens and concept labels.
11. Cluster value vectors by cosine similarity.
12. Optionally intervene by increasing or decreasing chosen FFN coefficients.
13. Optionally implement early exit by checking when dominant promoted concepts stabilize.

## Minimal PyTorch sketch

```python
import torch

# x: residual vector at a token position, shape [d_model]
# W_K: first FFN matrix, shape [d_model, d_ff]
# W_V: second FFN matrix, shape [d_ff, d_model]
# W_U: unembedding matrix, shape [d_model, vocab_size]

m = torch.nn.functional.gelu(x @ W_K)  # shape [d_ff]

# Static interpretation of value vector i
value_i = W_V[i]                       # shape [d_model]
static_scores = value_i @ W_U

# Dynamic contribution from value vector i
subupdate_i = m[i] * value_i
dynamic_scores = subupdate_i @ W_U
```

---

# 3. Logit spectroscopy, spectral filters, dark signals, and attention sinks

Source: https://aclanthology.org/2024.acl-long.263/

PDF: https://aclanthology.org/2024.acl-long.263.pdf

arXiv: https://arxiv.org/abs/2402.09221

## Core idea

Instead of SVD-ing every component weight matrix, SVD the embedding and unembedding matrices themselves. Their singular vectors define spectral directions in the residual stream.

Some directions strongly affect logits. Others barely affect logits and form a “dark” subspace. These dark directions can still matter for internal computation, especially attention sinks.

### Framework classification

- static model-weight method at the projector-definition stage
- dynamic activation-conditioned method at the residual-occupancy stage
- low-rank / spectral factor-analysis method

### Primary spaces

- singular-direction space of `W_U` / `W_E`
- residual-space band projections
- vocab-sensitivity space

### Framework inputs

Static stage:

- `W_U`
- `W_E`
- tokenizer

Dynamic stage:

- canonical forward capture artifacts
- optional component/subblock outputs

### Framework outputs

- spectral projector artifacts
- per-band occupancy summaries
- dark-ratio summaries
- filtered-run intervention summaries

## SVD of unembedding

Let:

```text
W_U = U_U Σ_U V_U^T
```

The right singular vectors of `W_U` are residual-stream directions ordered by how much they affect vocabulary logits.

High singular values correspond to directions that strongly affect logits.

Low singular values correspond to directions that weakly affect logits.

## U-dark subspace

The U-dark subspace is the span of the bottom singular vectors of `W_U`, often the bottom 5%.

These directions have small direct effect on vocabulary logits, so ordinary logit-lens methods may miss them.

## E-dark subspace

Similarly, compute the SVD of the embedding matrix:

```text
W_E = U_E Σ_E V_E^T
```

The E-dark subspace is defined from the low-singular-value tail of the embedding spectrum.

## Spectral bands

Split singular vectors into bands, for example 20 bands of 5% each:

```text
band 1  = largest singular values
band 20 = smallest singular values / darkest directions
```

Then measure how much residual stream or component output lies in each band.

## Projection filter

A projection onto a selected spectral band is:

```text
Phi = V_band @ V_band.T
```

Apply it to a residual vector:

```text
h_filtered = h @ Phi
```

or, depending on convention:

```text
h_filtered = Phi @ h
```

## Dark ratio

A simple measure of how dark a vector is:

```text
U_dark_ratio = ||project_U_dark(h)|| / ||h - project_U_dark(h)||
```

A high ratio means the vector has large magnitude in the U-dark subspace.

## Practical implementation plan

1. Load a model with accessible `W_E`, `W_U`, residual streams, and component outputs.
2. Compute SVD of `W_U`.
3. Compute SVD of `W_E`.
4. Split singular vectors into spectral bands.
5. Define U-dark and E-dark projectors.
6. During inference, record residual vectors and component outputs by layer and token.
7. Project those vectors into spectral bands.
8. Measure how much norm each component reads from or writes to each band.
9. Filter selected spectral bands during inference.
10. Measure loss or negative log-likelihood after filtering.
11. Compare against random subspace filtering.
12. Analyze high-attention tokens and beginning-of-sequence tokens for high U-dark ratios.
13. Test whether preserving the dark tail preserves attention sinks and model performance.

## Minimal PyTorch sketch

```python
import torch

# W_U: unembedding matrix, shape [d_model, vocab_size]
# h: residual vector, shape [d_model]

U, S, Vh = torch.linalg.svd(W_U, full_matrices=False)

# Right singular vectors live in rows of Vh only if W_U shape is [vocab_size, d_model].
# If W_U shape is [d_model, vocab_size], residual-space singular vectors are columns of U.
# Always check shapes.

# Example for W_U shape [d_model, vocab_size], so U is [d_model, d_model].
num_dirs = U.shape[1]
band_size = num_dirs // 20

# Bottom 5% directions: band 20
V_dark = U[:, -band_size:]             # shape [d_model, band_size]
P_dark = V_dark @ V_dark.T             # shape [d_model, d_model]

h_dark = h @ P_dark
h_not_dark = h - h_dark
u_dark_ratio = torch.linalg.norm(h_dark) / torch.linalg.norm(h_not_dark)
```

---

# Combined research plan

## Goal

Build an interpretability toolkit that can answer three complementary questions:

1. Which semantic directions are present in component weight matrices?
2. Which FFN value vectors are active for a specific input, and what do they promote?
3. Which residual-stream spectral bands are used for hidden computation that may not be visible through the ordinary logit lens?

## Relationship to the repository artifact system

These methods should be implemented so that:

1. static projector and weight analyses produce reusable model-level artifacts
2. dynamic prompt-conditioned analyses consume canonical forward capture artifacts
3. low-rank comparison analyses consume canonical comparison artifacts when relevant

This keeps the implementation aligned with the rest of the framework:

- one shared hidden-state source of truth for dynamic prompt analyses
- explicit separation between static and dynamic methods
- reusable outputs for later interventions, comparisons, compression, and adapter analysis

This is especially important for future extensions such as:

- quantized vs non-quantized LM-head analysis
- sub-MLP block analysis such as `up` / `gate` / `down`
- low-rank adapter and mixture-of-low-rank analysis
- MoE-related factor and routing analyses

## Relationship to the repository artifact system

These methods should be implemented so that:

1. static projector and weight analyses produce reusable model-level artifacts
2. dynamic prompt-conditioned analyses consume canonical forward capture artifacts
3. low-rank comparison analyses consume canonical comparison artifacts when relevant

This keeps the implementation aligned with the rest of the framework:

- one shared hidden-state source of truth for dynamic prompt analyses
- explicit separation between static and dynamic methods
- reusable outputs for later interventions, comparisons, compression, and adapter analysis

## Suggested stages

### Stage 1: Weight-direction atlas

Implement the SVD method from the Millidge/Black post.

Output table:

```text
layer | component | singular_index | singular_value | top_tokens | bottom_tokens | label
```

### Stage 2: FFN promotion atlas

Implement the Geva et al. sub-update method.

Output table:

```text
prompt | token_position | layer | value_index | coefficient | top_promoted_tokens | label
```

### Stage 3: Spectral residual atlas

Implement the Cancedda logit-spectroscopy method.

Output table:

```text
prompt | token_position | layer | component | spectral_band | norm_fraction | loss_after_filtering
```

### Stage 4: Cross-method comparison

For each interpretable direction or value vector:

1. Project it into vocabulary space.
2. Assign a concept label.
3. Project it into unembedding spectral bands.
4. Check whether it lives mostly in light, middle, or dark bands.
5. Test whether filtering those bands changes model behavior.

### Stage 5: Interventions

Try three intervention types:

1. **SVD rank-one deletion:** remove selected singular directions from weight matrices.
2. **FFN coefficient steering:** increase or decrease selected value-vector activations.
3. **Spectral filtering:** suppress or preserve selected embedding/unembedding spectral bands.

Measure:

```text
logit changes
loss / negative log-likelihood
generation quality
concept-specific behavior changes
attention pattern changes
```

---

# Practical notes

## Shape conventions matter

Different libraries store matrices differently. Always print shapes before multiplying.

Useful sanity checks:

```python
print("W_E", W_E.shape)
print("W_U", W_U.shape)
print("residual", h.shape)
print("vocab_scores", (h @ W_U).shape)
```

If `h @ W_U` does not produce `[vocab_size]`, transpose `W_U` or adjust conventions.

## Tokenization matters

Top-token lists may include:

- leading-space tokens,
- subword fragments,
- punctuation,
- capitalization variants,
- byte fallback tokens,
- multilingual fragments.

Interpret token clusters cautiously.

## Suggested libraries

- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
- Hugging Face Transformers: https://github.com/huggingface/transformers
- PyTorch SVD documentation: https://pytorch.org/docs/stable/generated/torch.linalg.svd.html

## Suggested starting models

- GPT-2 small for reproducing older SVD and FFN analyses.
- Pythia models for open-weight scaling experiments.
- LLaMA-family models for logit spectroscopy, if you have access and sufficient compute.

---

# Summary

The three papers/posts form a natural stack:

```text
Millidge/Black:
    SVD component weights -> project singular directions into vocab space.

Geva et al.:
    Decompose FFN outputs -> project value-vector sub-updates into vocab space.

Cancedda:
    SVD embedding/unembedding -> define spectral bands and dark subspaces.
```

Together, they provide a roadmap for studying how transformer components encode, promote, hide, and route semantic information through the residual stream.
