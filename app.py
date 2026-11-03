#!/usr/bin/env python3
"""
B2TF Almanac - league-aware miss classifier (v3.9). Apply AFTER v3.8.

Bug: classify_miss() is basketball-only. It labels a miss "FOUL TROUBLE MISS" when Actual_Fouls >= 5, or when
a minutes heuristic (mins <= 28, low per-minute output) fires. Neither belongs in NFL/NHL/MLB, so a bust like
Zay Flowers could be tagged "foul trouble". Fix: classify_miss takes a league argument; for NFL/NHL/MLB the foul
check and the minutes heuristic are skipped, and the blowout-tier wording is sport-appropriate (script/role/snaps).
Basketball behavior is unchanged. Unknown/blank league keeps the old behavior.

Usage:   python apply_nfl_patch_v3_9.py app.py
Safety:  all-or-nothing; backup (app.py.bak); syntax-checked afterwards.
"""
import re, sys, shutil, py_compile

PATCHES = []
def patch(name, old, new):
    PATCHES.append((name, old, new))

patch("1. classify_miss takes a league",
r'''def classify_miss(proj, line, actual, vote, minutes_played=None, actual_fouls=None):
    try:''',
r'''def classify_miss(proj, line, actual, vote, minutes_played=None, actual_fouls=None, league=None):
    _non_bb = str(league or '').strip().upper() in ('NFL', 'NHL', 'MLB')
    try:''')

patch("2. Foul check is basketball only",
r'''    if actual_fouls is not None and str(actual_fouls).strip() not in ['', 'nan', 'None']:
        try:
            fouls = int(float(actual_fouls))''',
r'''    if (not _non_bb) and actual_fouls is not None and str(actual_fouls).strip() not in ['', 'nan', 'None']:
        try:
            fouls = int(float(actual_fouls))''')

patch("3. Minutes heuristic is basketball only",
r'''        if minutes_played is not None:
            try:
                mins = float(minutes_played)''',
r'''        if (not _non_bb) and minutes_played is not None:
            try:
                mins = float(minutes_played)''')

patch("4. Sport-appropriate blowout-tier wording",
r'''        if vote == "OVER" and line > 0 and actual < line * 0.65:
            likely_cause = "Actual came in far below the line.''',
r'''        if _non_bb and vote == "OVER" and line > 0 and actual < line * 0.65:
            likely_cause = "Actual came in far below the line. Top causes: game script (blowout or abandoned game plan), role/snap-share change, in-game injury, or a negative-script week for this stat. Check snaps and targets/carries before blaming the model."
        elif vote == "OVER" and line > 0 and actual < line * 0.65:
            likely_cause = "Actual came in far below the line.''')

patch("5. Loss Pattern Report passes the league",
r'''lr.get('Vote', ''), lr.get('Actual_Mins', None), lr.get('Actual_Fouls', None))''',
r'''lr.get('Vote', ''), lr.get('Actual_Mins', None), lr.get('Actual_Fouls', None), lr.get('League', None))''')

patch("6. Autopsy card passes the league",
r'''actual_raw, row.get('Vote', ''), row.get('Actual_Mins', None), row.get('Actual_Fouls', None))''',
r'''actual_raw, row.get('Vote', ''), row.get('Actual_Mins', None), row.get('Actual_Fouls', None), row.get('League', None))''')

def build_regex(old):
    lines = old.split("\n")
    return re.compile(r"[ \t]*\r?\n".join(re.escape(l) for l in lines))

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__); sys.exit(1)
    path = args[0]
    src = open(path, encoding="utf-8", newline="").read()
    if "def _consensus_quote(" not in src:
        print("This file does not have v3.8 yet. Apply v2 through v3.8 first (or use app_patched_latest.py).")
        sys.exit(4)
    if "league=None):\n    _non_bb" in src:
        print("v3.9 is already applied to this file. Nothing to do.")
        sys.exit(0)
    out, failures, applied = src, [], []
    for name, old, new in PATCHES:
        hits = list(build_regex(old).finditer(out))
        if len(hits) != 1:
            failures.append((name, len(hits))); continue
        m = hits[0]
        out = out[:m.start()] + new + out[m.end():]
        applied.append(name)
    for n in applied: print(f"  OK       {n}")
    for n, c in failures: print(f"  FAILED   {n}  (matched {c}x, expected exactly 1)")
    if failures:
        print(f"\n{len(failures)} edit(s) did not match. NOTHING was written. Send me the failed names above.")
        sys.exit(2)
    shutil.copyfile(path, path + ".bak")
    open(path, "w", encoding="utf-8", newline="").write(out)
    try:
        py_compile.compile(path, doraise=True)
        print(f"\nSyntax check passed. Wrote {path}  (backup: {path}.bak)")
    except py_compile.PyCompileError as e:
        print(f"\nSYNTAX ERROR after patching: {e}")
        shutil.copyfile(path + ".bak", path); print("Original restored.")
        sys.exit(3)

if __name__ == "__main__":
    main()
