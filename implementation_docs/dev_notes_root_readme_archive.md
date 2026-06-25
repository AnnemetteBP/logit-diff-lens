# Root README Archive Notes

This file preserves older scratch notes and commands that previously lived in the root `README.md`.

They are kept here for reference while the root README is repurposed into a cleaner front page for the repository.

## Tuned lens training command

```bash
python pipelines/pythia/train_tuned_lens.py \
  --model-name EleutherAI/pythia-1.4b-deduped \
  --dataset-path /media/am/AM/logit-diff-lens/src/tuned-lens/val.jsonl \
  --output-dir tmp/tuned_lens/pythia_1.4b_val_jsonl \
  --dtype bfloat16 \
  --seq-len 128 \
  --batch-size 4 \
  --epochs 10 \
  --max-records 2000
```

## Single-prompt tuned-vs-modelnorm plotting commands

```bash
python pipelines/pythia/plot_single_prompt_tuned_vs_modelnorm.py \
  --model-name EleutherAI/pythia-70m-deduped \
  --tuned-lens-dir tmp/tuned_lens/pythia_70m_val_jsonl \
  --prompt "If I had more time, I would travel more often." \
  --output-pdf tmp/tuned_lens/pythia_70m_val_jsonl_compare/tuned_vs_modelnorm_single_prompt_jsd.pdf \
  --top-k 10 \
  --layer-mode all \
  --metric jsd
```

```bash
python pipelines/pythia/plot_single_prompt_tuned_vs_modelnorm.py \
  --model-name EleutherAI/pythia-70m-deduped \
  --tuned-lens-dir tmp/tuned_lens/pythia_70m_val_jsonl \
  --prompt "If I had more time, I would travel more often." \
  --output-pdf tmp/tuned_lens/pythia_70m_val_jsonl_compare/tuned_vs_modelnorm_single_prompt_all_layers.pdf \
  --top-k 10 \
  --layer-mode all
```

```bash
python pipelines/pythia/plot_single_prompt_tuned_vs_modelnorm.py \
  --model-name EleutherAI/pythia-70m-deduped \
  --tuned-lens-dir tmp/tuned_lens/pythia_70m_val_jsonl \
  --prompt "If I had more time, I would travel more often." \
  --output-pdf tmp/tuned_lens/pythia_70m_val_jsonl_compare/tuned_vs_modelnorm_single_prompt_all_layers.pdf \
  --top-k 10 \
  --layer-mode most_divergent \
  --max-layers 5
```

## JSD reminder

```text
p = softmax(modelnorm_logits)
q = softmax(tuned_lens_logits)
m = 0.5 * (p + q)

KL(p || q) = sum_i p_i * (log p_i - log q_i)
KL(q || p) = sum_i q_i * (log q_i - log p_i)
JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)
```

## Direct tuned-lens CLI training command

```bash
cd /media/am/AM/logit-diff-lens

/home/am/miniconda3/envs/ldl-env/bin/python -m tuned_lens \
  --log_level INFO \
  train \
  --output tmp/tuned_lens/pythia_70m_val_jsonl_run2 \
  --seed 42 \
  --num_steps 2 \
  --tokens_per_step 512 \
  --loss KL \
  --model.name EleutherAI/pythia-70m-deduped \
  --precision bfloat16 \
  --model.revision main \
  --data.name tuned-lens/val.jsonl \
  --split validation \
  --text_column text \
  --max_seq_len 128 \
  --dataset_shuffle false \
  --dataset_shuffle_seed 42 \
  --weight_decay 0.001 \
  --lr_scale 1.0 \
  --momentum 0.9 \
  --optimizer ADAM \
  --per_gpu_batch_size 4 \
  --dataloader_shuffle true
```
