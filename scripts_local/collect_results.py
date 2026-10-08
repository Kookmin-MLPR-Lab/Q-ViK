"""Collect lmms-eval metrics from result trees into {task: value-string}."""
import glob
import json
import os
import re

MC = ["mme", "mmbench_en_dev", "mmstar", "pope", "vizwiz_vqa", "scienceqa_img"]
FREE = ["textvqa", "gqa", "docvqa", "chartqa", "coco_cap", "nocaps", "textcaps"]
TASKS = MC + FREE
HEAD = {"mme": "MME", "mmbench_en_dev": "MMB_EN", "mmstar": "MMStar", "pope": "POPE",
        "vizwiz_vqa": "VizWiz", "scienceqa_img": "SQA_I", "textvqa": "TextVQA", "gqa": "GQA",
        "docvqa": "DocVQA", "chartqa": "ChartQA", "coco_cap": "COCO", "nocaps": "NoCaps",
        "textcaps": "TextCaps"}
SCALAR = {"mme": ("mme_percetion_score", 1), "mmbench_en_dev": ("gpt_eval_score", 1),
          "mmstar": ("average", 100), "pope": ("pope_accuracy", 100),
          "vizwiz_vqa": ("exact_match", 100), "scienceqa_img": ("exact_match", 100),
          "textvqa": ("exact_match", 100), "gqa": ("exact_match", 100),
          "docvqa": ("anls", 100), "chartqa": ("relaxed_overall", 100)}
CAP_PREFIX = {"coco_cap": "coco", "nocaps": "nocaps", "textcaps": "textcaps"}


def norm_task(name):
    name = re.sub(r"_(offline|local)$", "", name)
    name = {"vizwiz_vqa_val": "vizwiz_vqa", "textvqa_val": "textvqa", "docvqa_val": "docvqa",
            "coco2017_cap_val": "coco_cap", "nocaps_val": "nocaps", "textcaps_val": "textcaps"}.get(name, name)
    return name


def fmt(task, metrics):
    if task in SCALAR:
        key, scale = SCALAR[task]
        v = metrics.get(key + ",none")
        if not isinstance(v, (int, float)):
            return None
        return f"{v * scale:.1f}"
    p = CAP_PREFIX.get(task)
    r, c = metrics.get(f"{p}_ROUGE_L,none"), metrics.get(f"{p}_CIDEr,none")
    if isinstance(r, (int, float)) and isinstance(c, (int, float)):
        return f"{r * 100:.1f} / {c * 100:.1f}"
    return None


def collect(*roots, tasks=TASKS):
    """Newest result wins when a task appears more than once."""
    found = {}
    for root in roots:
        for f in glob.glob(os.path.join(root, "**", "*results.json"), recursive=True):
            try:
                j = json.load(open(f))
            except Exception:
                continue
            mtime = os.path.getmtime(f)
            # MMBench whose Excel-submission step crashed (no openpyxl) still
            # saved its accuracy as <dir>/mmbench_en_dev_results.json.
            if isinstance(j, dict) and "overall_acc" in j and os.path.basename(f).startswith("mmbench_en_dev"):
                if "mmbench_en_dev" in tasks and "mmbench_en_dev" not in found:
                    found["mmbench_en_dev"] = (0, f"{j['overall_acc'] * 100:.1f}", f)
                continue
            if not isinstance(j, dict) or not isinstance(j.get("results"), dict):
                continue
            for raw, metrics in j["results"].items():
                task = norm_task(raw)
                if task not in tasks:
                    continue
                val = fmt(task, metrics)
                if val is not None and (task not in found or found[task][0] < mtime):
                    found[task] = (mtime, val, f)
    return {t: v[1] for t, v in found.items()}, {t: v[2] for t, v in found.items()}
