# Q-ViK



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
    ├── textvqa_val/                  # TextVQA val (or TextVQA/data/*.parquet)
    ├── GQA/                          # testdev_balanced_{instructions,images}
    ├── ChartQA/data/
    ├── DocVQA/DocVQA/
    ├── COCO-Caption2017/data/
    ├── NoCaps/data/
    ├── TextCaps/data/
    ├── POPE/data/
    ├── MME/data/
    ├── MMStar/mmstar_lmms.parquet    # (or MMStar/mmstar.parquet)
    ├── VizWiz-VQA/data/
    ├── ScienceQA/ScienceQA-IMG/      # (or ScienceQA-IMG/)
    ├── MMBench/en/                   # (or MMBench-EN/)
    ├── VQAv2/data/
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
  --output-dir ckpts/qvik_student_onevision

# LLaVA-1.5
python qvik/train/llava15.py \
  --teacher-root data/train/teacher/llava15 \
  --llava-path model/llava-v1.5-7b \
  --epochs 15 \
  --output-dir ckpts/qvik_student_llava15
```

### 3. Evaluation


Results are written to `results/<model_tag>/keep<keep_ratio>/<task>/`
(`model_tag` is `llava15_lmms` or `onevision_lmms`). Available tasks: `textvqa`,
`chartqa`, `docvqa`, `gqa`, `coco_cap`, `nocaps`, `textcaps`, `pope`, `mme`,
`mmstar`, `vizwiz_vqa`, `scienceqa_img`, `mmbench_en_dev`, `vqav2_testdev`,
`vqav2_test` (and their shards `vqav2_testdev_s0`-`s11`, `vqav2_test_s0`-`s11`).

**lmms-eval (LLaVA-1.5)** 
```bash
python qvik/eval/run_lmms_eval.py \
  --model lmms_llava15_student \
  --model_args pretrained=model/llava-v1.5-7b,student_path=ckpts/qvik_student_llava15,keep_ratio=0.5,device=cuda:0 \
  --tasks textvqa,chartqa,docvqa,gqa,coco_cap,nocaps,textcaps \
  --batch_size 1 \
  --output_path results
```

**lmms-eval (OneVision)** 
```bash
python qvik/eval/run_lmms_eval.py \
  --model lmms_onevision_student \
  --model_args pretrained=model/llava-onevision-qwen2-7b-ov,student_path=ckpts/qvik_student_onevision,keep_ratio=0.5,device=cuda:0 \
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
    --student_path ckpts/qvik_student_onevision \
    --keep_ratio 0.5
done
```

