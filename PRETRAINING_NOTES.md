# Pretraining Notes

> Working notes connecting foundational language model research to the implementation in this repo — documenting my understanding and design choices, not a full tutorial or paper reproduction.

---

## Overview

A hands-on implementation of autoregressive pretraining from scratch, on the TinyStories dataset. Not an attempt to reproduce GPT-3 scale — the goal was to understand the core components by building them:

tokenization → vocabulary → token/positional embeddings → next-token prediction → causal self-attention → multi-head attention → Transformer blocks → cross-entropy loss → training/validation → autoregressive generation.

Progressed through three versions: character-level LM → longer context → custom BPE + Transformer.

---

## 1. Research Foundation

| Paper | Key Idea | Connection to This Project |
|---|---|---|
| **Attention Is All You Need** | Transformer architecture, self-attention | Basis for the Transformer implementation |
| **GPT-3 (Few-Shot Learners)** | Large-scale autoregressive modeling | Conceptual basis for next-token prediction / generation |
| **Scaling Laws for Neural LMs** | Model size, data, compute vs. performance | Context for interpreting a small model's limitations |
| **Chinchilla (Compute-Optimal LLMs)** | Balancing params vs. training data | Context for model capacity / dataset size trade-offs |
| **The Pile** | Large-scale dataset construction | Context for why data quality/scale matters in pretraining |

> This project doesn't reproduce these papers' scale, data, or results — they're the theoretical reference points for a much smaller implementation.

---

## 2. Pretraining Objective

The model learns `P(xₜ₊₁ | x₁, ..., xₜ)` — predict the next token given everything before it. Training pairs are just the input sequence shifted by one:

```
Input:  One day the little girl went to
Target: day the little girl went to the
```

## 3. Autoregressive Generation

Generation is iterative: predict next-token distribution → sample → append → repeat. The sequence probability factorizes as a product of conditionals:

```
P(x₁,...,xₙ) = P(x₁) · P(x₂|x₁) · P(x₃|x₁,x₂) · ... · P(xₙ|x₁,...,xₙ₋₁)
```

---

## 4–6. Tokenization: Character-Level → Custom BPE

Text can't be fed to the model directly — it's converted to token IDs. Started with character-level tokenization (simple but inefficient — `"Transformer"` → 11 tokens). Moved to a **custom BPE tokenizer**: start from characters, iteratively merge the most frequent adjacent pair, repeat until vocab target is reached.

**Implementation:**
- Pre-tokenized with `r"\w+|[^\w\s]|\s+"` to split words / punctuation / whitespace before applying merges.
- Trained to a vocab size of **1,024**, yielding **459,429** total BPE tokens over the TinyStories corpus.
- Correctness verified via round-trip: `decode(encode(text)) == text` → `True`.

## 7. Vocabulary

Two lookup directions are needed: `token_to_id` for encoding, `id_to_token` (`vocab`) for decoding. Both must come from the *same* tokenizer — mixing vocabularies (e.g. an old char-level `itos` with new BPE IDs) breaks decoding (see Troubleshooting #2).

## 8. Data Preparation

```
Total tokens: 459,429
Train:        413,486
Val:           45,943
```

## 9. Context Window & Batching

`block_size = 128` — the model attends to up to 128 prior tokens. Input/target are the same window shifted by one position, giving 128 next-token prediction tasks per sequence.

```python
def get_batch(split):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
```

With `batch_size=32`, each batch is `[32, 128]` → **4,096** next-token predictions per batch.

---

## 11. Token & Positional Embeddings

Token IDs `[32, 128]` → embedding layer (`n_embd=128`) → `[32, 128, 128]`. Since self-attention processes tokens in parallel (no inherent order awareness), **positional embeddings** are added to token embeddings so the model can distinguish token order (`"dog bites man"` vs `"man bites dog"`).

## 13–15. Self-Attention (Query/Key/Value, Causal Masking)

Each token gets Query, Key, Value vectors via learned linear projections. Attention scores come from `Query · Key`, normalized with softmax, then applied to Values to build a context-aware representation.

Because generation is autoregressive, tokens must not attend to future positions — enforced via a **causal mask** (lower-triangular attention pattern), so position *t* only sees positions `≤ t`.

## 16. Multi-Head Attention

`n_head=4`, so `head_size = n_embd/n_head = 128/4 = 32`. Four heads run in parallel on `[32,128,32]` slices, each free to learn different relationships (local context, subject-verb links, repeated patterns, etc.), then concatenate back to `[32,128,128]`.

## 17–19. Transformer Blocks

`n_layer=4` stacked blocks, each: `LayerNorm → Multi-Head Attention → residual add → LayerNorm → Feed-Forward → residual add`. The feed-forward network transforms each token position independently (vs. attention, which mixes information *across* positions). Residual connections + LayerNorm keep a 4-layer stack trainable.

## 20. Final Model Configuration

| Component | Value |
|---|---:|
| Vocabulary size | 1,024 |
| Batch size | 32 |
| Context window | 128 |
| Embedding dim | 128 |
| Attention heads | 4 |
| Transformer layers | 4 |
| Dropout | 0.1 |
| **Total parameters** | **~1.07M** |

## 21–22. Logits & Loss

Transformer output `[32,128,128]` → linear LM head → vocabulary logits `[32,128,1024]`. For `F.cross_entropy`, batch and sequence dims are flattened: logits → `[4096,1024]`, targets → `[4096]` (see Troubleshooting #6).

## 23–24. Training & Validation Results

Config: `n_embd=128, n_head=4, n_layer=4, dropout=0.1`, ~1.07M params.

| Step | Train Loss | Val Loss |
|---|---|---|
| 0 | 7.229 | 7.234 |
| 1000 | 2.838 | 2.834 |
| 2000 | 2.247 | 2.282 |
| 3000 | 2.025 | 2.083 |
| 4000 | 1.874 | 1.975 |
| **Final** | **1.773** | **1.908** |

Train/val losses tracked closely throughout — no notable overfitting at this scale.

## 25. Autoregressive Generation

```
context → Transformer → logits → softmax → sample token → append → repeat
```

## 26–31. Relationship to Larger LLMs (GPT-3, Scaling Laws, Chinchilla, The Pile)

Same core pipeline as GPT-style models (tokenize → embed → causal Transformer → next-token loss), just at a fraction of the scale (1.07M params / 1,024 vocab / 128 context vs. billions of params and web-scale corpora). Scaling-law and Chinchilla research gave useful context for *why* a small model trained briefly on a narrow dataset can't match production LLM output quality — model size, data volume, and compute all need to scale together, and this project deliberately doesn't. The Pile's data-construction methodology (filtering, dedup, diverse sources) highlighted how much simpler this project's single-dataset (TinyStories) pipeline is by comparison, and that data quality is as much a part of pretraining as architecture.

---

## 32. Version Progression

| Version | Implementation | Focus |
|---|---|---|
| V1 | Character-level LM | Basic next-token prediction, embeddings, loss, generation |
| V2 | Longer context (`block_size=128`) | Input/target windowing at scale |
| V3 | Custom BPE + Transformer | Full pipeline: tokenizer, causal multi-head attention, training, generation |

## 33. What I Learned

Building the full pipeline turned a one-line description (*text → tokenizer → Transformer → loss → generated text*) into a series of concrete engineering problems: tokenizer merge rules and vocab consistency, embeddings and positional info, QKV/causal masking/multi-head attention, feed-forward + residuals + LayerNorm, batch/loss reshaping, and debugging real issues along the way (BPE merge ordering, CUDA device-side asserts, tensor shape mismatches — see `TROUBLESHOOTING.md`).

---

## Final Pipeline

```
TinyStories → BPE Tokenizer (vocab 1024) → Token IDs → Train/Val Split
  → Batches → Token + Positional Embeddings → Transformer Block × 4
  (Multi-Head Causal Attention + Feed-Forward) → LM Head
  → Vocabulary Logits → Cross-Entropy Loss → Autoregressive Generation
```

**Result:** A ~1.07M-parameter Transformer LM trained on TinyStories with a custom 1,024-token BPE tokenizer and a 128-token context window — built to understand the pretraining pipeline end-to-end, not to match production-scale LLM quality.