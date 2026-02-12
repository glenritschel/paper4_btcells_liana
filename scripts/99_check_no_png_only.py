#!/usr/bin/env python
"""
Fail if any script uses plt.savefig(...) directly.
All figures must use savefig_png_pdf() from _plotting.py.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Files allowed to contain plt.savefig()
ALLOW_FILES = {
    "_plotting.py",
    "99_check_no_png_only.py",  # allow self
}

pattern = re.compile(r"\bplt\.savefig\s*\(")

bad = []
for p in SCRIPTS.glob("*.py"):
    if p.name in ALLOW_FILES:
        continue

    txt = p.read_text(encoding="utf-8", errors="ignore")
    if pattern.search(txt):
        bad.append(str(p.relative_to(ROOT)))

if bad:
    print("ERROR: Direct plt.savefig() found. Use savefig_png_pdf() instead:")
    for b in bad:
        print(" -", b)
    sys.exit(1)

print("OK: No direct plt.savefig() usage found outside approved files.")

