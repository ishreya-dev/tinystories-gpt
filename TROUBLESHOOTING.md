# Troubleshooting and Debugging Notes

Implementation issues encountered while building the TinyStories language model, their causes, and fixes.

**Stages covered:** BPE tokenization → encoding/decoding → data prep → batching → bigram baseline → Transformer implementation → GPU debugging → training/generation.

---

## 1. BPE encode/decode mismatch

Decoding an encoded sentence didn't reproduce the original text.

**Cause:** Token-to-ID conversion was happening *inside* the merge loop, so partially-merged sequences were appended repeatedly instead of only the final tokenization.

**Fix:** Moved ID conversion outside the loop, after all BPE merges completed.

```python
for piece in pieces:
    sequence = list(piece)
    for pair in self.merges:
        sequence = self.merge_pair(sequence, pair)
    for token in sequence:
        token_ids.append(token_to_id[token])
```

Encode → decode became fully reversible after the fix.

---

## 2. KeyError during decoding

`KeyError: 124` when decoding a training batch.

**Cause:** The decoder was still using `itos` from the earlier character-level tokenizer, but the data had already been encoded with the custom BPE vocabulary — the two mappings weren't compatible.

**Fix:** Updated `decode()` to use the BPE tokenizer's own vocabulary.

```python
def decode(ids):
    return "".join(tokenizer.vocab[i] for i in ids)
```

**Lesson:** Encoding and decoding must always use the same vocabulary system.

---

## 3. CUDA device-side assert error

`CUDA error: device-side assert triggered`, surfacing during `get_batch("train")` — but CUDA errors are asynchronous, so the stack trace didn't necessarily point to the real source.

**Debug check:** Verified token IDs were within valid vocab range:

```text
Vocab size: 1024 | min: 0 | max: 1023 | invalid IDs: False
```

IDs were valid, so the issue was a stale CUDA context. **Fix:** restarted the runtime and recreated the tokenizer, data, batches, and model from scratch.

**Lesson:** After a device-side assert, the CUDA context can stay corrupted even once the root cause is fixed — restart before continuing.

---

## 4. Input/target tensor verification

Verified batch generation was correctly preparing shifted-by-one sequences for next-token prediction.

```text
Input shape:  [32, 128]
Target shape: [32, 128]
```

Manually decoded a sample batch to confirm `target == input shifted left by 1`. Shape-checking alone wasn't sufficient — decoding actual sequences caught what shape checks alone would've missed.

---

## 5. Embedding / attention shape tracking

Config: `n_embd=128, n_head=4, block_size=128, batch_size=32`.

| Stage | Shape |
|---|---|
| Token IDs | `[32, 128]` |
| Token embeddings | `[32, 128, 128]` |
| Single attention head (`head_size = 128/4 = 32`) | `[32, 128, 32]` |
| Concatenated multi-head output | `[32, 128, 128]` |

Confirms embedding dimension is split across heads (`4 × 32 = 128`) and recombined after attention — no shape mismatch.

---

## 6. Cross-entropy loss shape

`F.cross_entropy()` expects `[N, C]` / `[N]`, but logits came out as `[batch, block, vocab] = [32, 128, 1024]`.

**Fix:** flattened batch and sequence dims before computing loss.

```python
B, T, C = logits.shape
logits = logits.view(B * T, C)     # [4096, 1024]
targets = targets.view(B * T)      # [4096]
```

Each of the `32 × 128 = 4096` token positions is one classification problem over the vocabulary.

---

## 7. Bigram baseline: incoherent output

Initial bigram model (`P(xₜ₊₁ | xₜ)`) produced fragments of real words but no grammar or long-range structure — expected, since it only conditions on a single previous token. Served its purpose as a baseline before moving to a Transformer with causal self-attention over a longer context.

---

## 8. Transformer training results

**Config:** `batch_size=32, block_size=128, n_embd=128, n_head=4, n_layer=4, dropout=0.1` — **1,071,360 parameters**.

| Step | Train Loss | Val Loss |
|---|---|---|
| 0 | 7.229 | 7.234 |
| 500 | 3.236 | 3.221 |
| 1000 | 2.838 | 2.834 |
| 2000 | 2.247 | 2.282 |
| 3000 | 2.025 | 2.083 |
| 4000 | 1.874 | 1.975 |
| 4500 | 1.829 | 1.946 |
| **Final** | **1.773** | **1.908** |

Train/val loss tracked closely throughout, with no signs of overfitting at this scale.

---

## 9. Generated text quality vs. loss

The Transformer produced clear improvements over the bigram baseline — character names, dialogue, partial sentence structure (e.g. *"One day, Max went to a place with the mom and said..."*) — but still showed grammatical errors, repeated phrases, and inconsistent characters across longer generations.

**Cause:** expected at this scale — ~1M params, 128-token context, limited training/compute. Loss and generation quality don't move in lockstep, so both need to be checked, not just the loss curve.

---

## Summary of Lessons

1. Finish all BPE merges before converting tokens to IDs.
2. Use one consistent vocabulary for both encoding and decoding.
3. Validate token IDs are in-range before training.
4. CUDA errors report asynchronously — the stack trace may lag the real cause.
5. Restart the runtime after a device-side assert.
6. Decode and manually inspect batches, not just shapes.
7. Track tensor shapes at every stage.
8. Multi-head attention splits embedding dim across heads, then concatenates.
9. Flatten batch/sequence dims before cross-entropy.
10. Start with a simple baseline to understand its limits before scaling up.
11. Monitor train *and* val loss.
12. Always inspect generated output — loss alone doesn't capture quality.

---

## Version Progression

| Version | Implementation | Focus |
|---|---|---|
| V1 | Character-level LM | Basic next-token prediction |
| V2 | Increased context length | Longer sequence handling |
| V3 | Custom BPE + Transformer | Tokenization, causal multi-head attention, full training pipeline |