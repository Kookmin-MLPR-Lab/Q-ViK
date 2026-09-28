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

### Prefill mode (how the first answer token is produced)

`prefill_mode` in `--model_args` applies to every task in both wrappers
(`qvik/eval/prefill_mode.py`):

- `prefill_mode=qvik` (default): prefill the prompt without its final token,
  evict visual KVs, then feed the final prompt token through the compressed
  cache. The first answer token therefore sees only the kept visual KVs.
- `prefill_mode=origin`: prefill the full prompt, take the first answer token
  from that full-attention pass, then evict. This is the usual post-prefill
  KV-eviction protocol. On short-answer / multiple-choice benchmarks the first
  token is often the whole answer, so this mode barely depends on `keep_ratio`.

The stats JSON records `prefill_mode`. In both modes, text tokens after the
image were computed with full image attention during prefill, so their KVs
still carry visual information after the image KVs are evicted.

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
