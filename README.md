# Transformer WMT14 En-De Reproduction

This project implements the Transformer base model from "Attention Is All You Need" with PyTorch primitive modules only. It does not use `torch.nn.Transformer`, `torch.nn.MultiheadAttention`, fairseq, OpenNMT, or HuggingFace pretrained translation models.

## Target Experiment

- Dataset: WMT14 English -> German
- Train split: full `train`
- Validation: `validation` / newstest2013 in common WMT14 setups
- Test: `test` / newstest2014 in common WMT14 setups
- Paper reference: Transformer base Table 2 BLEU 27.3
- Main metric: sacreBLEU with tokenizer `13a`, mixed case, saved signature

## Tokenizer

Use the public OpenNMT WMT14 En-De shared SentencePiece model instead of training a tokenizer from a small subset.

```bash
python scripts/download_tokenizer.py
```

Default URL:

```text
https://huggingface.co/malloc/OpenNMT-py-English-German-Transformer/resolve/main/sentencepiece.model
```

The tokenizer is a preprocessing asset only. The Transformer model, loss, scheduler, beam search, and training loop are implemented in this repo.

## Base Configuration

See `configs/base_wmt14_ende.yaml`.

Important defaults:

```text
layers: encoder 6, decoder 6
d_model: 512
d_ff: 2048
heads: 8
dropout: 0.1
label_smoothing: 0.1
Adam betas: 0.9, 0.98
Adam eps: 1e-9
Noam warmup: 4000 optimizer updates
micro batch: 2048 target non-pad tokens
grad accumulation: 12
effective batch: about 24576 target tokens/update
beam size: 4
length penalty: 0.6
```

## Critical Reproduction Rules

- Batching budget is counted by target-side non-pad tokens.
- Scheduler step means optimizer update step, not epoch and not gradient accumulation micro-step.
- In fp16 AMP, if GradScaler skips `optimizer.step()`, Noam step and `update_step` do not advance.
- Source truncation always keeps EOS at the final position.
- Target truncation reserves BOS and EOS: content length is at most `max_tgt_len - 2`.
- Loss is label-smoothed CE summed over non-pad target tokens and normalized by the full token count of the accumulated update.
- Embeddings use `Normal(0, d_model^-0.5)`; linear layers use Xavier uniform.
- Training starts from scratch for the main result; old checkpoints should not be reused.

## Workflow

Install dependencies:

```bash
pip install -r requirements.txt
```

Download tokenizer:

```bash
python scripts/download_tokenizer.py
```

Prepare smoke data:

```bash
python scripts/prepare_wmt14.py --config configs/smoke_test.yaml --num-proc 4 --trust-remote-code
```

Run smoke training:

```bash
python scripts/train.py --config configs/smoke_test.yaml
```

Prepare full WMT14:

```bash
python scripts/prepare_wmt14.py --config configs/base_wmt14_ende.yaml --num-proc 8 --trust-remote-code
```

Train full base model:

```bash
python scripts/train.py --config configs/base_wmt14_ende.yaml
```

Average checkpoints:

```bash
python scripts/average_checkpoints.py --checkpoint-dir outputs/checkpoints --last 5 --output outputs/checkpoints/averaged.pt
```

Translate newstest2014:

```bash
python scripts/translate.py --config configs/base_wmt14_ende.yaml --checkpoint outputs/checkpoints/averaged.pt --output outputs/translations/newstest2014.hyp.txt
```

Evaluate BLEU:

```bash
python scripts/evaluate.py --config configs/base_wmt14_ende.yaml --hyp outputs/translations/newstest2014.hyp.txt
```

## Expected Result

A careful single-GPU base run should aim for roughly:

```text
sacreBLEU tok.13a mixed: about 25.5-26.8
paper-style/tokenized BLEU: about 26.5-27.3
```

Full 100k-update training on a single RTX 4090 can take several days depending on implementation speed, disk, dataloader workers, sequence lengths, and AMP mode.

