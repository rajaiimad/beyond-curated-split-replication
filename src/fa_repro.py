"""
fa_repro.py -- revision 3: compares result files re-computed with the released code against the
committed versions, value by value (run-time fields excluded). Writes results/reproducibility.json.

Usage: python fa_repro.py <results dir> <git revision of the committed files> <recomputed file> ...
       (each recomputed file is compared with <git revision>:results/<its name without '.rerun'>)
"""
from __future__ import annotations
import json, math, platform, subprocess, sys
from pathlib import Path

SKIP = {"secs", "versions", "secs_total", "assumptions"}   # run times, versions, and fa_vrlens.py's copy of its own documentation


def compare(old, new):
    n, diffs = 0, []

    def walk(a, b, p):
        nonlocal n
        if isinstance(a, dict):
            for k in a:
                if k in SKIP:
                    continue
                if not isinstance(b, dict) or k not in b:
                    diffs.append([p + "/" + k, "missing"]); continue
                walk(a[k], b[k], f"{p}/{k}")
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                diffs.append([p, "length"]); return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{p}[{i}]")
        else:
            n += 1
            if isinstance(a, float) or isinstance(b, float):
                if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                    diffs.append([p, a, b])
                elif not (math.isnan(a) and math.isnan(b)) and abs(a - b) > 1e-9:
                    diffs.append([p, a, b])
            elif a != b:
                diffs.append([p, a, b])
    walk(old, new, "")
    return n, diffs


if __name__ == "__main__":
    R, rev = Path(sys.argv[1]), sys.argv[2]
    out = {"python": platform.python_version(), "committed_revision": rev, "files": {}}
    for f in map(Path, sys.argv[3:]):
        name = f.name.replace(".rerun", "")
        old = json.loads(subprocess.run(["git", "show", f"{rev}:results/{name}"], capture_output=True, text=True,
                                        check=True).stdout)
        n, d = compare(old, json.load(open(f)))
        out["files"][name] = {"values": n, "differ": len(d), "examples": d[:10]}
        print(f"{name}: {n} values, {len(d)} differ")
    out["total_values"] = sum(v["values"] for v in out["files"].values())
    out["total_differ"] = sum(v["differ"] for v in out["files"].values())
    json.dump(out, open(R / "reproducibility.json", "w"), indent=1)
