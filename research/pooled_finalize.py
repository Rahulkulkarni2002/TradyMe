"""End pass for the one-model-for-all-coins training (research/pooled.py).

pooled.py only switches a coin when the new model PASSES the 100-situation test and is at least as often right.
When BOTH the current and the new model fail, the better one should still win. Rule here:
  switch if the new model is right ≥ 0.3 points more often (sure arrows, 2025 → now)
  and wins at most 5 fewer situations.
The old model is kept in research/models/replaced/.

    python -m research.pooled_finalize            (shows what it would do)
    python -m research.pooled_finalize --apply
"""
from __future__ import annotations

import argparse
import json
import shutil

from .strategy_league import MODELS, RESULTS

STAGE = MODELS / "pooled"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    for staged in sorted(STAGE.glob("*.joblib")):
        name = staged.stem
        new_p, cur_p = RESULTS / f"{name}_pooled_stress.json", RESULTS / f"{name}_stress.json"
        cur_model = MODELS / f"{name}.joblib"
        if not new_p.exists() or not cur_p.exists() or not cur_model.exists():
            continue
        new, cur = json.loads(new_p.read_text()), json.loads(cur_p.read_text())
        if cur.get("pooled_now"):                        # already switched
            continue
        both_fail = new["verdict"] == "FAIL" and cur["verdict"] == "FAIL"
        gain = new["overall_right"] - cur["overall_right"]
        if both_fail and gain >= 0.003 and new["passed"] >= cur["passed"] - 5:
            print(f"✅ {name}: switch — {cur['overall_right']:.1%} ({cur['passed']}/{cur['counted']}) → "
                  f"{new['overall_right']:.1%} ({new['passed']}/{new['counted']}) · both fail the 100-situation test")
            if a.apply:
                (MODELS / "replaced").mkdir(exist_ok=True)
                shutil.copy2(cur_model, MODELS / "replaced" / cur_model.name)
                shutil.copy2(staged, cur_model)
                cur["pooled_now"] = True
                cur_p.write_text(json.dumps(cur, indent=1, default=float))
        elif both_fail:
            print(f"➖ {name}: keep — new {new['overall_right']:.1%} ({new['passed']}/{new['counted']}) vs current "
                  f"{cur['overall_right']:.1%} ({cur['passed']}/{cur['counted']})")


if __name__ == "__main__":
    main()
