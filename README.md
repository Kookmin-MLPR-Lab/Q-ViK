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

### 2. Student training


```bash
# OneVision
python qvik/train/llava_onevision.py \
  --teacher-root data/train/teacher/llava_onevision \
  --llava-path model/llava-onevision-qwen2-7b-ov \
  --epochs 20 \
  --output-dir ckpts/student_onevision

# LLaVA-1.5
python qvik/train/llava15.py \
  --teacher-root data/train/teacher/llava15 \
  --epochs 20 \
  --output-dir ckpts/student_llava15
```

### 3. Evaluation


Results are written to `results/<model_tag>/<task>/`. Available tasks: `textvqa`, `chartqa`, `docvqa`, `gqa`,
`coco_cap`, `nocaps`, `textcaps`

**lmms-eval (LLaVA-1.5)** 
```bash
python qvik/eval/run_lmms_eval.py \
  --model lmms_llava15_student \
  --model_args pretrained=model/llava-v1.5-7b,student_path=ckpts/student_llava15,keep_ratio=0.5,device=cuda:0 \
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

### Sequential Q-ViK + text KV eviction

The LLaVA-1.5 and OneVision evaluation wrappers can apply a second, text-only
cache policy after Q-ViK has selected the visual tokens. Visual survivors are
always protected by the second stage.

- `text_eviction_mode=none`: Q-ViK only (control).
- `text_eviction_mode=streamingllm`: keep initial text sinks plus recent text.
- `text_eviction_mode=h2o`: keep per-head attention heavy hitters plus recent
  text. Scores are accumulated from decode queries instead of materializing the
  quadratic prefill attention matrix.
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
  --model_args pretrained=model/llava-v1.5-7b,student_path=ckpts/student_llava15,keep_ratio=0.5,keep_ratio_basis=image,text_eviction_mode=h2o,text_keep_ratio=0.2,h2o_recent_ratio=0.5,device=cuda:0 \
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

For the downloaded MM-NIAH text-needle validation split, run the native
LLaVA-1.5 <=2K sanity slice directly:

```bash
CUDA_VISIBLE_DEVICES=0 python qvik/eval/mm_niah_llava15_student.py \
  --task retrieval-text \
  --keep_ratio 0.5 \
  --keep_ratio_basis image \
  --text_eviction_mode h2o \
  --text_keep_ratio 0.2 \
  --max_expanded_tokens 2048
```

Repeat with `counting-text` and `reasoning-text`, and with
`text_eviction_mode=none,streamingllm,h2o`. Each run writes detailed
predictions, cache statistics, and an official-format JSONL under
`official_outputs/`. The latter can be checked with the upstream scorer:

```bash
python ../MM-NIAH/calculate_scores.py \
  --outputs-dir results/mm_niah/<run>/official_outputs
```

Run the original-format LLaVA-OneVision model over every MM-NIAH row that
fits its native 32K window with full visual and text KV caches:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  python qvik/eval/mm_niah_onevision_full.py \
  --pretrained ../models/llava-onevision-qwen2-7b-ov \
  --max_expanded_tokens 32736 \
  --max_new_tokens 32 \
  --output_dir results/mm_niah_onevision_full_cache_32k
```

The runner encodes multi-image AnyRes inputs sequentially, filters on the
exact expanded prompt length, and saves each prediction immediately so an
interrupted run resumes without repeating completed samples.

Apply Q-ViK to 50% of visual KVs and then H2O to a 50% text-KV budget on the
same native-32K MM-NIAH subset:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  python qvik/eval/mm_niah_onevision_full.py \
  --pretrained ../models/llava-onevision-qwen2-7b-ov \
  --student_path ckpts/student_onevision \
  --visual_keep_ratio 0.5 \
  --text_eviction_mode h2o \
  --text_keep_ratio 0.5 \
  --h2o_recent_ratio 0.5 \
  --max_expanded_tokens 32736 \
  --max_new_tokens 32 \
  --output_dir results/mm_niah_onevision_qvik0.5_h2o0.5_32k
```

The visual ratio is applied independently at every layer. The H2O text budget
is 50% of the original prompt-text count and is split equally between heavy
hitters and recent entries; Q-ViK visual survivors are protected from H2O.

See `docs/long_text_eviction.md` for the full long-context benchmark order and
ablation matrix.
