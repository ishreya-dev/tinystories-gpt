# TinyStories GPT

A small GPT-style Transformer built from scratch — tokenizer, attention, training loop, and all — to understand how autoregressive language model pretraining actually works, end to end.

Not an attempt to reproduce GPT-3. The goal was to implement every stage of the pipeline by hand on a small scale (TinyStories dataset, ~1.07M parameters) and understand what's happening at each step.

```
Final Train Loss: 1.7727   |   Final Val Loss: 1.9079
Vocab: 1,024   |   Context: 128   |   Params: ~1.07M
```

**Sample generation (after training):**
> One day, Max went to a place with the mom and said, "Hey, okay, I want to fight?." Baby followed it and tried to wait and show her...

(Coherent-ish, still rough — see [Limitations](#limitations).)

---

## Pipeline

```
TinyStories → Custom BPE Tokenizer → Token IDs → Train/Val Split
  → Batches → Token + Positional Embeddings → Transformer Block × 4
  (Causal Multi-Head Attention + Feed-Forward) → LM Head
  → Vocabulary Logits → Cross-Entropy Loss → Autoregressive Generation
```

## Model Architecture

GPT-style decoder-only Transformer:

```
Token IDs → Token Emb + Position Emb → [Transformer Block × 4] → Final LayerNorm
  → LM Head → Vocabulary Logits

Each block: LayerNorm → Causal Multi-Head Attention → +residual
          → LayerNorm → Feed-Forward → +residual
```

## Configuration

| Component | Value |
|---|---:|
| Vocabulary size | 1,024 |
| Batch size | 32 |
| Context window | 128 tokens |
| Embedding dim | 128 |
| Attention heads | 4 |
| Transformer layers | 4 |
| Dropout | 0.1 |
| **Parameters** | **~1.07M** |

```python
batch_size, block_size = 32, 128
n_embd, n_head, n_layer = 128, 4, 4
dropout = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## Tokenizer

Custom BPE implementation (not an external library): pre-tokenize with `r"\w+|[^\w\s]|\s+"` → start from characters → iteratively merge the most frequent adjacent pair → repeat to target vocab size.

Trained to **1,024** vocab, producing **459,429** tokens over the corpus (413,486 train / 45,943 val). Verified round-trip correct: `decode(encode(text)) == text`.

## Training Results

| Step | Train Loss | Val Loss |
|---|---|---|
| 0 | 7.229 | 7.234 |
| 1000 | 2.838 | 2.834 |
| 2000 | 2.247 | 2.282 |
| 3000 | 2.025 | 2.083 |
| 4000 | 1.874 | 1.975 |
| **Final** | **1.773** | **1.908** |

---

## Project Versions

| Version | Focus |
|---|---|
| **V1** | Character-level LM — embeddings, next-token prediction, cross-entropy, generation |
| **V2** | Extended context (`block_size=128`) — batching, input/target shifting |
| **V3** | Custom BPE + full Transformer — causal self-attention, multi-head attention, complete training pipeline |

## Repository Structure

```
tinystories-gpt/
├── README.md
├── PRETRAINING_NOTES.md    # Research background + implementation notes
├── TROUBLESHOOTING.md      # Debugging issues and resolutions
└── notebooks/
    ├── V1/                 # Character-level language model
    ├── V2/                 # Extended context window
    └── V3/                 # Custom BPE + Transformer
```

## Documentation

| File | Description |
|---|---|
| [`PRETRAINING_NOTES.md`](./PRETRAINING_NOTES.md) | Research papers → implementation, concept-by-concept |
| [`TROUBLESHOOTING.md`](./TROUBLESHOOTING.md) | Bugs hit during development and how they were fixed |
| [`notebooks/`](./notebooks/) | V1–V3 development notebooks |

## Limitations

Small by design: ~1.07M params, 1,024 vocab, 128-token context, no distributed training, no fine-tuning/instruction-tuning, basic sampling (no top-k/top-p), no benchmark eval. Generated text can still show repeated phrases, grammatical slips, and weak long-range coherence — expected at this scale.

## Future Improvements

- Larger dataset / more training steps
- Larger context window & model capacity
- Learning rate scheduling
- Top-k / top-p / temperature sampling
- Checkpointing, perplexity eval, training curve visualization

## Tech Stack

Python · PyTorch · CUDA · Custom BPE Tokenizer · Jupyter · Google Colab

## References

- Vaswani et al. — *Attention Is All You Need*
- Brown et al. — *Language Models are Few-Shot Learners*
- Kaplan et al. — *Scaling Laws for Neural Language Models*
- Hoffmann et al. — *Training Compute-Optimal Large Language Models*
- Gao et al. — *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*

---

**Disclaimer:** Educational/experimental implementation built to understand Transformer pretraining mechanics — not intended to reproduce the scale or performance of production LLMs like GPT-3.