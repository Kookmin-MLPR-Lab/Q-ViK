# Q-ViK

Q-ViK is a training-based KV cache compression method for visual language models.
A lightweight student MLP+CNN is trained to predict which image tokens are important for future decoding,
and used at inference time to selectively prune the KV cache — without any changes to the base model.

## Environment

```bash
conda env create -f environment.yml
conda activate qvik
```

## Models

| Model | HuggingFace | Local path |
|---|---|---|
| LLaVA-OneVision-Qwen2-7B | [lmms-lab/llava-onevision-qwen2-7b-ov](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov) | `model/llava-onevision-qwen2-7b-ov` |
| LLaVA-1.5-7B | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) | `model/llava-v1.5-7b` |

## Datasets

### Training (teacher extraction)

- [TextVQA](https://textvqa.org/)
- [GQA](https://cs.stanford.edu/people/dorarad/gqa/)
- [ScienceQA](https://scienceqa.github.io/)

### Evaluation

- [TextVQA](https://textvqa.org/)
- [GQA](https://cs.stanford.edu/people/dorarad/gqa/)
- [ChartQA](https://github.com/vis-nlp/ChartQA)
- [DocVQA](https://www.docvqa.org/)
- [NoCaps](https://nocaps.org/)
- [TextCaps](https://textvqa.org/textcaps/)
- [MileBench](https://milebench.github.io/)

## Data Structure

```
data/
├── train/
│   ├── textvqa/          # TextVQA train images & annotations
│   ├── gqa/              # GQA train images & questions
│   ├── scienceqa/        # ScienceQA images & problems.json
│   └── teacher/
│       ├── llava15/      # extracted teacher scores for LLaVA-1.5
│       └── llava_onevision/  # extracted teacher scores for OneVision
└── eval/
    ├── TextVQA/
    ├── GQA/
    ├── ChartQA/
    ├── DocVQA/
    ├── NoCaps/
    ├── TextCaps/
    └── MileBench/
```

## Pipeline

### 1. Teacher extraction



```bash
# OneVision
for ds in textvqa gqa scienceqa; do
  python qvik/teacher/extract_llava_onevision.py \
    --model model/llava-onevision-qwen2-7b-ov \
    --dataset $ds --n-samples 600 \
    --output-root data/train/teacher/llava_onevision
done

# LLaVA-1.5
for ds in textvqa gqa scienceqa; do
  python qvik/teacher/extract_llava15.py \
    --model model/llava-v1.5-7b \
    --dataset $ds --n-samples 600 \
    --output-root data/train/teacher/llava15
done
```

The teacher is the attention from generated answer tokens to image tokens,
averaged over heads and answer tokens and normalized per layer (the zap
teacher used by the released checkpoints).

### 2. Student training


```bash
# OneVision
python qvik/train/llava_onevision.py \
  --teacher-root data/train/teacher/llava_onevision \
  --llava-path model/llava-onevision-qwen2-7b-ov \
  --epochs 20 \
  --output-dir ckpts/qvik_student_onevision

# LLaVA-1.5
python qvik/train/llava15.py \
  --teacher-root data/train/teacher/llava15 \
  --epochs 20 \
  --output-dir ckpts/qvik_student_llava15
```

The LLaVA-1.5 student scores decoder layer `l` from `hidden_states[l + offset]`
(`--hidden-state-offset`, default 1 = output of layer `l`). The offset is saved
in the checkpoint's `config.json` and read back at inference, so training,
validation and evaluation always use the same index. Checkpoints without the
field are loaded with offset 1. OneVision always uses offset 1.

### 3. Evaluation


Results are written to `results/<model_tag>/<task>/`. Available tasks: `textvqa`, `chartqa`, `docvqa`, `gqa`,
`coco_cap`, `nocaps`, `textcaps`

**lmms-eval (LLaVA-1.5)** 
```bash
python qvik/eval/run_lmms_eval.py \
  --model lmms_llava15_student \
  --model_args pretrained=model/llava-v1.5-7b,student_path=ckpts/v1/student_llava15_orig_vflow_1800_lr1e4_e15,keep_ratio=0.5,device=cuda:0 \
  --tasks textvqa,chartqa,docvqa,gqa,coco_cap,nocaps,textcaps \
  --batch_size 1 \
  --output_path results
```

**lmms-eval (OneVision)** 
```bash
python qvik/eval/run_lmms_eval.py \
  --model lmms_onevision_student \
  --model_args pretrained=model/llava-onevision-qwen2-7b-ov,student_path=ckpts/student_onevision,keep_ratio=0.5,device=cuda:0 \
  --tasks textvqa,chartqa,docvqa,gqa,coco_cap,nocaps,textcaps \
  --batch_size 1 \
  --output_path results
```

**MileBench (OneVision)**
```bash
for ds in ALFRED CLEVR-Change IEdit Spot-the-Diff; do
  python qvik/eval/milebench_onevision_student.py \
    --dataset $ds \
    --pretrained model/llava-onevision-qwen2-7b-ov \
    --student_path ckpts/student_onevision \
    --keep_ratio 0.5
done
```

### Delayed replay (evict before the first answer token)

Q-ViK prefills the full prompt and trims the visual KVs afterwards. Without
extra care, the first generated token still comes from the full-attention
prefill. On short-answer / multiple-choice benchmarks that token is often the
whole answer, so accuracy barely depends on `keep_ratio`.

For these task families both wrappers hold out the final prompt token, prefill
the rest, evict, and then replay the held-out token through the trimmed cache
(`qvik/eval/delayed_replay.py`). The first answer token then sees only the
kept visual KVs. Default families:

`mme, pope, mmstar, vizwiz_vqa, gqa, scienceqa_img, mmbench, vqav2, seedbench`

A family also matches its sub-tasks (`vqav2` → `vqav2_test_s3`, `mmbench` →
`mmbench_en_dev`). Override with `delayed_replay_tasks=pope|gqa` in
`--model_args` (use `|`, since lmms-eval splits model args on commas), with
`delayed_replay_tasks=all` / `none`, or with the env var
`QVIK_DELAYED_REPLAY_TASKS`. The wrapper prints one
`task=... delayed_replay=True/False` line per task, and the stats JSON records
`delayed_replay_samples`.

Note that text tokens after the image were still computed with full image
attention during prefill, so their KVs carry visual information even after
the image KVs are evicted.

### Keep-ratio basis

- **OneVision:** `keep_ratio` is the fraction of image tokens kept.
- **LLaVA-1.5, default `keep_ratio_basis=total`:** `keep_ratio` is the kept
  fraction of the whole prompt, with text always kept. Therefore
  `n_keep = max(1, n_img - (1 - keep_ratio) * prompt_len)`. Once `keep_ratio`
  is at or below the prompt's text fraction (`n_text / prompt_len`), only one
  image token survives. For example, the text fraction is about 0.08 on POPE
  and about 0.19 on ScienceQA-IMG, so keep 0.05 keeps 1 of 576 image tokens on
  both.
- **LLaVA-1.5, `keep_ratio_basis=image`:** keeps `ceil(keep_ratio * n_img)`
  image tokens, the same meaning as OneVision.

### Sequential Q-ViK + text KV eviction

The LLaVA-1.5 and OneVision evaluation wrappers can apply a second, text-only
cache policy after Q-ViK has selected the visual tokens. Visual survivors are
always protected by the second stage.

- `text_eviction_mode=none`: Q-ViK only (control).
- `text_eviction_mode=streamingllm`: keep initial text sinks plus recent text.
- `text_eviction_mode=h2o`: keep per-head attention heavy hitters plus recent
  text. Scores are accumulated from decode queries instead of materializing the
  quadratic prefill attention matrix. The decode-query probabilities are
  recomputed in fp32 by a hook (`DecodeAttentionProbe`), so the model keeps
  running on SDPA. Requesting `output_attentions=True` would fall back to eager
  fp16 attention, which overflows on OneVision/Qwen2 and produces NaN logits.
- `text_cache_size=N`: fixed number of text KV entries (overrides the ratio).
- `text_keep_ratio=0.2`: text budget as a fraction of prompt text when the
  fixed size is zero.
- `h2o_recent_ratio=0.5`: fraction of the H2O text budget reserved for recent
  entries; the remainder is the heavy-hitter budget.
- `streaming_sink_size=4`: number of initial text entries protected by
  StreamingLLM.

H2O with a 20% total text cache (10% heavy hitters + 10% recent):

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python qvik/eval/run_lmms_eval.py \
  --model lmms_llava15_student \
  --model_args pretrained=model/llava-v1.5-7b,student_path=ckpts/v1/student_llava15_orig_vflow_1800_lr1e4_e15,keep_ratio=0.5,keep_ratio_basis=image,text_eviction_mode=h2o,text_keep_ratio=0.2,h2o_recent_ratio=0.5,device=cuda:0 \
  --tasks textvqa,chartqa,docvqa,gqa \
  --batch_size 1 \
  --output_path results/qvik_h2o
```

StreamingLLM with a fixed 512-entry text cache:

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python qvik/eval/run_lmms_eval.py \
  --model lmms_onevision_student \
  --model_args pretrained=model/llava-onevision-qwen2-7b-ov,student_path=ckpts/student_onevision,keep_ratio=0.5,text_eviction_mode=streamingllm,text_cache_size=512,streaming_sink_size=4,device=cuda:0 \
  --tasks textvqa,chartqa,docvqa,gqa \
  --batch_size 1 \
  --output_path results/qvik_streamingllm
```

The keep-ratio stats JSON records both stages, including
`n_image_kept`, `text_cache_budget`,
`avg_n_text_prompt_kept_after_eviction`, and
`avg_n_visual_cache_final`.
