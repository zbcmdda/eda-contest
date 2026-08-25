#!/usr/bin/env python3
"""Compare Python reference short features with an offline C++ CSV dump."""
import argparse
from pathlib import Path
import pandas as pd
from short_features import NAMES, ShortFeatureReference
from train_model import ArchitectureFeatures

p = argparse.ArgumentParser()
p.add_argument("--input", type=Path, required=True)
p.add_argument("--cpp", type=Path, required=True)
p.add_argument("--arch", type=Path, default=Path("arch"))
p.add_argument("--limit", type=int, default=128)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
raw = pd.read_csv(a.input, dtype={"From":"string", "To":"string"}).iloc[:a.limit]
cpp = pd.read_csv(a.cpp, dtype={"From":"string", "To":"string"}).iloc[:a.limit]
if not raw[["From","To"]].equals(cpp[["From","To"]]): raise ValueError("C++ dump endpoint mismatch")
python = ShortFeatureReference(ArchitectureFeatures(a.arch)).build(raw)
diff = (python.reset_index(drop=True) != cpp[NAMES].astype("int32").reset_index(drop=True))
report = {"rows":len(raw), "mismatched_cells":int(diff.to_numpy().sum()),
          "mismatched_columns":diff.sum().loc[lambda x:x>0].to_dict()}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(__import__("json").dumps(report, indent=2, sort_keys=True)+"\n")
print(__import__("json").dumps(report, indent=2, sort_keys=True))
if report["mismatched_cells"]: raise SystemExit(1)
