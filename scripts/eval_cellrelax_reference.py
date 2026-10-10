"""
Run an existing phonon eval with the DFT reference of the 16 cellrelax materials taken from
workflow/FINAL_RESULTS_CELLRELAX/ (ISIF=4-refined cells) instead of workflow/FINAL_RESULTS_HEALTHY/.
All other materials are read from FINAL_RESULTS_HEALTHY as usual. Additive: neither eval script is
modified; only `eval_healthy2d_finetune.load_dft_reference` is wrapped. Once the 16 entries of
FINAL_RESULTS_HEALTHY have been swapped for the FINAL_RESULTS_CELLRELAX ones (cellrelax stage 6),
the routing is a no-op and this wrapper gives exactly the plain scripts' result.

Usage (from nequix/):
  uv run python scripts/eval_cellrelax_reference.py healthy2d <args of scripts/eval_healthy2d_finetune.py>
  uv run python scripts/eval_cellrelax_reference.py explicit  <args of explicit_dispersion/eval_explicit_dispersion.py>
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "explicit_dispersion"))

import eval_healthy2d_finetune as E  # noqa: E402

WORKFLOW_ROOT = E.WORKFLOW_ROOT
HEALTHY = WORKFLOW_ROOT / "FINAL_RESULTS_HEALTHY"
CELLRELAX = WORKFLOW_ROOT / "FINAL_RESULTS_CELLRELAX"
M16 = {l.split()[0] for l in (WORKFLOW_ROOT / "cellrelax" / "materials_16.txt").read_text().splitlines()
       if l.strip() and not l.startswith("#")}
assert len(M16) == 16, M16

_orig_load = E.load_dft_reference


def load_dft_reference(material: str):
    E.FINAL_RESULTS_HEALTHY = CELLRELAX if material in M16 else HEALTHY
    try:
        return _orig_load(material)
    finally:
        E.FINAL_RESULTS_HEALTHY = HEALTHY


E.load_dft_reference = load_dft_reference


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("healthy2d", "explicit"):
        sys.exit(__doc__)
    which, rest = sys.argv[1], sys.argv[2:]
    print(f"[eval_cellrelax_reference] {which}: DFT reference for {len(M16)} cellrelax materials <- {CELLRELAX}", flush=True)
    if which == "healthy2d":
        sys.argv = [str(HERE / "eval_healthy2d_finetune.py")] + rest
        E.main()
    else:
        import eval_explicit_dispersion as X
        X.load_dft_reference = load_dft_reference  # it imported the name at import time
        sys.argv = [str(ROOT / "explicit_dispersion" / "eval_explicit_dispersion.py")] + rest
        X.main()


if __name__ == "__main__":
    main()
