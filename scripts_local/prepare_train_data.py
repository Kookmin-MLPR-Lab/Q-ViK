"""Build Q-ViK/data/train/{scienceqa,gqa,textvqa} from local parquet files in
the layout the teacher extractors expect. Only a subset of GQA/TextVQA images is
materialized; the extractors sample n=600 from whatever has an image on disk."""
import io, json, random, glob
from pathlib import Path
import pyarrow.parquet as pq

OUT = Path("/workspace/nips/Q-ViK/data/train")
EVAL = Path("/workspace/nips/data/eval")
SQA = Path("/workspace/nips/data/train_src/scienceqa/train.parquet")

def save(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(img["bytes"])

# ScienceQA: problems.json keyed train_<i>, images/train/train_<i>/image.png
t = pq.read_table(SQA).to_pylist()
problems = {}
for i, r in enumerate(t):
    if not r["image"] or not r["image"].get("bytes"):
        continue
    qid = f"train_{i}"
    problems[qid] = {"question": r["question"], "choices": r["choices"], "answer": r["answer"], "hint": r["hint"]}
    from PIL import Image
    p = OUT / "scienceqa/images/train" / qid / "image.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(p)
(OUT / "scienceqa").mkdir(parents=True, exist_ok=True)
(OUT / "scienceqa/problems.json").write_text(json.dumps(problems))
print("scienceqa", len(problems))

# GQA: images from 2 shards, questions restricted to those images
img_ids = set()
for f in sorted(glob.glob(str(EVAL / "GQA/train_balanced_images/*.parquet")))[:2]:
    for r in pq.read_table(f).to_pylist():
        save(r["image"], OUT / "gqa/images" / f"{r['id']}.jpg")
        img_ids.add(r["id"])
q = pq.read_table(EVAL / "GQA/train_balanced_instructions/train-00000-of-00001.parquet",
                  columns=["id", "imageId", "question", "answer"]).to_pylist()
q = [r for r in q if r["imageId"] in img_ids]
random.Random(0).shuffle(q)
qs = {r["id"]: {"imageId": r["imageId"], "question": r["question"], "answer": r["answer"]} for r in q[:3000]}
(OUT / "gqa/train_balanced_questions_600.json").write_text(json.dumps(qs))
print("gqa", len(img_ids), "images", len(qs), "questions")

# TextVQA: 2 train shards
recs = []
for f in sorted(glob.glob(str(EVAL / "TextVQA/data/train-*.parquet")))[:2]:
    for r in pq.read_table(f).to_pylist():
        rel = f"textvqa/train/images/{r['image_id']}.jpg"
        if not (OUT / rel).exists():
            save(r["image"], OUT / rel)
        recs.append({"question_id": r["question_id"], "question": r["question"], "answers": r["answers"], "image_path": rel})
(OUT / "textvqa/train/data.json").write_text(json.dumps({"data": recs}))
print("textvqa", len(recs))
