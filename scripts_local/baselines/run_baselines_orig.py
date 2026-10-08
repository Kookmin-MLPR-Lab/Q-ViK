#!/usr/bin/env python3
"""Run original-code LOOK-M / PrefixKV wrappers through Q-ViK's lmms-eval task
harness (same local task YAMLs, scoring and results layout as the Q-ViK runs).

Q-ViK's runner imports its own student wrappers at startup; those need a newer
transformers than the 4.33.1 both baselines pin, so they are stubbed out here.
Usage: python run_baselines_orig.py --model lookm_llava15_orig|prefixkv_llava15_orig ...
"""
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
QVIK_ROOT = Path("/workspace/nips/Q-ViK")
LMMS_EVAL_ROOT = Path("/workspace/nips/VFlowOpt/src/lmms_eval-0.2.4")
sys.path.insert(0, str(LMMS_EVAL_ROOT))
sys.path.insert(0, str(QVIK_ROOT))
sys.path.insert(0, str(HERE))

model = sys.argv[sys.argv.index("--model") + 1]
WRAPPERS = {
    "lookm_llava15_orig": ("lookm_llava15_orig", "LookMLlava15Orig"),
    "prefixkv_llava15_orig": ("prefixkv_llava15_orig", "PrefixKVLlava15Orig"),
}
mod_name, cls_name = WRAPPERS[model]
wrapper = __import__(mod_name)

for stub in ("qvik.eval.lmms_llava15_student", "qvik.eval.lmms_onevision_student"):
    sys.modules[stub] = types.ModuleType(stub)
sys.modules[f"lmms_eval.models.{model}"] = wrapper

from lmms_eval.models import AVAILABLE_MODELS  # noqa: E402
AVAILABLE_MODELS[model] = cls_name

import qvik.eval.run_lmms_eval as runner  # noqa: E402
runner.MODEL_TAGS[model] = model

# runner.main() re-registers the stubbed student modules; keep ours in place.
_orig_main = runner.main
def main():
    AVAILABLE_MODELS[model] = cls_name
    _orig_main()
    
if __name__ == "__main__":
    main()
