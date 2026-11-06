#!/usr/bin/env python3
"""
B2TF Almanac - NBA calibration + gate + logging fixes (v4.0 Stage 1). Apply AFTER v3.9.

Why: the Feb-May NBA ledger (134 clean legs) showed the raw win probability is badly overconfident.
  Raw 90%+ hit 50% (-25% to -33% ROI); model >=30 pts above the book hit 31% (-47.5% ROI); overs projected
  35%+ over the line hit 36%. Display + gating only. Projections, ensemble and Poisson/normal sims are untouched.

What it does (NBA only, other leagues unchanged except the logging fix):
 1. Calibrated win probability: p = implied + NBA_MARKET_K * (raw - implied), capped at NBA_PROB_CAP.
    Price Edge, EV, Kelly, Setup Score and the VETO rule all use the calibrated number. Raw is shown in grey.
 2. NBA leg gate (same badge as NFL): L5 hits >= 4/5, calibrated prob >= 60%, L5 minutes >= 24 and not falling,
    price not shorter than -300, "too good to be true" (raw model > 25 pts over the book, or an Over
    projected 35%+ over the line) and a Role Risk checkbox. Unders get a warning.
 3. Logging: every result now carries p_over / p_under (calibrated) and raw values. OVERRIDE saves the
    probability of OVER (it used to save the model's own vote probability, often a PASS = 0.50, which is
    the cause of the wrong-side and 0.50 placeholder rows). New ledger column Raw_Prob keeps the raw number
    so you can recalibrate later. Gate_Log now also records NBA legs; the NFL gate report ignores them.

BEFORE your next logged bet: type "Raw_Prob" in cell Y1 of ROI_Ledger.

Usage:   python apply_nba_patch_v4_0.py app.py
Safety:  all-or-nothing; backup (app.py.bak); syntax-checked afterwards.
"""
import re, sys, shutil, py_compile

PATCHES = []
def patch(name, old, new):
    PATCHES.append((name, old, new))

patch("1. NBA constants, calibration and gate (before _nfl_pos)",
r'''def _nfl_pos(archetype):''',
r'''# ===== NBA calibration + leg gate (v4.0 stage 1). Provisional values: tune from Raw_Prob in the ledger. =====
NBA_PROB_CAP      = 0.70    # calibrated win prob is never above this
NBA_MARKET_K      = 0.20    # weight on the model when blending toward the book's implied probability
NBA_GATE_MIN_PROB = 0.60    # calibrated prob needed to be parlay-eligible
NBA_GATE_MIN_L5   = 4
NBA_GATE_MIN_MINS = 24.0    # L5 average minutes for Over legs
NBA_TOO_GOOD_GAP  = 25.0    # raw model prob minus book implied prob, in points
NBA_TOO_GOOD_CUSH = 0.35    # an Over projected this far above the line (lines of 3 or more)
NBA_SHORT_PRICE   = -300    # legs priced at or shorter than this fail the gate


def nba_calibrated_prob(raw, implied):
    # Shrinks the raw simulated probability toward the book's implied probability, then caps it.
    try:
        raw, imp = float(raw), float(implied)
    except Exception:
        return raw
    if raw != raw or imp != imp:
        return raw
    return float(max(0.01, min(imp + NBA_MARKET_K * (raw - imp), NBA_PROB_CAP)))


def nba_leg_gate(df_ml, s_col, line, vote, win_prob, raw_prob=None, implied_prob=None, odds=None, proj=None, role_risk=False):
    # Same return shape as nfl_leg_gate so the badge and the gate log work unchanged. None when not an OVER/UNDER leg.
    if vote not in ("OVER", "UNDER") or s_col not in df_ml.columns:
        return None
    ser = pd.to_numeric(df_ml[s_col], errors='coerce').dropna()
    l5 = ser.tail(5)
    n = int(len(l5))
    reasons, warns = [], []
    hits = int((l5 >= line).sum()) if vote == "OVER" else int((l5 <= line).sum())
    med = float(l5.median()) if n else 0.0

    if n < 5:
        reasons.append(f"only {n} recent games")
    elif hits < NBA_GATE_MIN_L5:
        reasons.append(f"L5 hit rate {hits}/5 (need {NBA_GATE_MIN_L5}/5)")
    if win_prob < NBA_GATE_MIN_PROB:
        reasons.append(f"calibrated prob {win_prob*100:.0f}% (need {NBA_GATE_MIN_PROB*100:.0f}%)")

    mins_avg = None
    if 'MINS' in df_ml.columns:
        mc = pd.to_numeric(df_ml['MINS'], errors='coerce').dropna()
        if len(mc) >= 3:
            mins_avg = float(mc.tail(5).mean())
            if vote == "OVER" and mins_avg < NBA_GATE_MIN_MINS:
                reasons.append(f"low minutes: L5 avg {mins_avg:.1f} (need {NBA_GATE_MIN_MINS:g}+)")
            if len(mc) >= 5:
                l2 = float(mc.tail(2).mean())
                prior = float(mc.iloc[-5:-2].mean())
                if prior >= 20 and l2 < 0.80 * prior:
                    reasons.append(f"minutes falling: L2 {l2:.1f} vs {prior:.1f} before")

    try:
        if raw_prob is not None and implied_prob is not None:
            gap = (float(raw_prob) - float(implied_prob)) * 100.0
            if gap > NBA_TOO_GOOD_GAP:
                reasons.append(f"too good to be true: raw model {float(raw_prob)*100:.0f}% is {gap:.0f} pts over the book ({float(implied_prob)*100:.0f}%). Check role, minutes, injuries")
        if vote == "OVER" and proj is not None and float(line) >= 3 and float(proj) >= float(line) * (1.0 + NBA_TOO_GOOD_CUSH):
            reasons.append(f"too good to be true: projection {float(proj):.1f} is {(float(proj)/float(line)-1)*100:.0f}% over the line")
    except Exception:
        pass

    try:
        if odds is not None and float(odds) <= NBA_SHORT_PRICE:
            reasons.append(f"price {int(float(odds))} is {NBA_SHORT_PRICE} or shorter (ledger: these hit ~61% vs 82% needed)")
    except Exception:
        pass

    if role_risk:
        reasons.append("role risk flagged (returning starter / minutes change)")
    if vote == "UNDER":
        warns.append("Unders hit 33% in the Feb-May ledger (12 legs)")

    return {"ok": len(reasons) == 0, "reasons": reasons, "warns": warns,
            "hits": hits, "n": n, "median": med, "opp_avg": mins_avg, "position": ""}


def nba_gate_safe(*a, **k):
    # Never let a gate problem break the analysis run.
    try:
        return nba_leg_gate(*a, **k)
    except Exception:
        return None


def _nfl_pos(archetype):''')

patch("2. Calibrate the win probability and capture p_over / p_under",
r'''                ev_dollars   = (win_prob * profit) - ((1 - win_prob) * 100)
                edge_pct     = (win_prob - implied_prob) * 100''',
r'''                # v4.0: raw simulated probabilities for both sides, then NBA calibration
                win_prob_raw = float(win_prob)
                _pois = (league_key == "MLB" and s_col in MLB_POISSON_STATS) or (league_key == "NFL" and s_col in NFL_POISSON_STATS)
                p_over_raw  = float(np.mean(sims > line))
                p_under_raw = float(np.mean(sims < line))
                if _pois:
                    p_over_raw, p_under_raw = min(p_over_raw, 0.85), min(p_under_raw, 0.85)
                p_over_v, p_under_v = p_over_raw, p_under_raw
                if league_key == "NBA":
                    p_over_v  = nba_calibrated_prob(p_over_raw, implied_prob)
                    p_under_v = nba_calibrated_prob(p_under_raw, implied_prob)
                    if c_vote in ("OVER", "UNDER"):
                        win_prob = nba_calibrated_prob(win_prob_raw, implied_prob)
                ev_dollars   = (win_prob * profit) - ((1 - win_prob) * 100)
                edge_pct     = (win_prob - implied_prob) * 100''')

patch("3. Result dict carries raw and per-side probabilities",
r'''"setup_score": setup_score, "win_prob": win_prob, "edge_pct": edge_pct,''',
r'''"setup_score": setup_score, "win_prob": win_prob, "edge_pct": edge_pct,
                    "win_prob_raw": win_prob_raw, "p_over": p_over_v, "p_under": p_under_v,
                    "p_over_raw": p_over_raw, "p_under_raw": p_under_raw,''')

patch("4. NBA gate on each result",
r'''                    "gate": (nfl_gate_safe(df_ml, s_col, line, c_vote, win_prob, archetype,
                                           bool(st.session_state.get(f"{lk}.role_risk", False)),
                                           st.session_state.get("nfl.game_total"), st.session_state.get("nfl.game_total_open"))
                             if league_key == "NFL" else None),''',
r'''                    "gate": (nfl_gate_safe(df_ml, s_col, line, c_vote, win_prob, archetype,
                                           bool(st.session_state.get(f"{lk}.role_risk", False)),
                                           st.session_state.get("nfl.game_total"), st.session_state.get("nfl.game_total_open"))
                             if league_key == "NFL" else
                             (nba_gate_safe(df_ml, s_col, line, c_vote, win_prob, win_prob_raw, implied_prob, odds, c_proj,
                                            bool(st.session_state.get(f"{lk}.role_risk", False)))
                              if league_key == "NBA" else None)),''')

patch("5. Role Risk checkbox for NBA",
r'''            if league_key == "NFL":
                st.checkbox("⚠️ Role Risk", key=f"{lk}.role_risk",''',
r'''            if league_key == "NBA":
                st.checkbox("⚠️ Role Risk", key=f"{lk}.role_risk",
                            help="Returning starter, minutes restriction, or a role change since his recent logs. Over legs for this player fail the gate.")
            if league_key == "NFL":
                st.checkbox("⚠️ Role Risk", key=f"{lk}.role_risk",''')

patch("6. Gate badge shows for NBA",
r'''gate_html = nfl_gate_badge_html(result.get('gate')) if league_key == "NFL" else ""''',
r'''gate_html = nfl_gate_badge_html(result.get('gate')) if league_key in ("NFL", "NBA") else ""''')

patch("7. Raw probability text (NBA)",
r'''                _pe_col = "#00c853" if result['edge_pct'] >= 3.0 else ("#f59e0b" if result['edge_pct'] >= 0 else "#ff5252")''',
r'''                _pe_col = "#00c853" if result['edge_pct'] >= 3.0 else ("#f59e0b" if result['edge_pct'] >= 0 else "#ff5252")
                _raw_txt = (f" <span style='color:#64748b;'>(raw {result['win_prob_raw']*100:.0f}%)</span>"
                            if (league_key == "NBA" and result.get('win_prob_raw') is not None) else "")''')

patch("8. Raw probability in the result line",
r'''Win Prob {result['win_prob']*100:.1f}%</div>{gate_html}''',
r'''Win Prob {result['win_prob']*100:.1f}%{_raw_txt}</div>{gate_html}''')

patch("9. OVERRIDE saves the probability of OVER (not the model's own vote)",
r'''save_to_ledger(league_key, target_player, r['stat_type'], r['line'], r['odds'], r['proj'], "OVER", r['win_prob'], False, r['setup_score'], implied, r['line'])''',
r'''save_to_ledger(league_key, target_player, r['stat_type'], r['line'], r['odds'], r['proj'], "OVER", r.get('p_over', r['win_prob']), False, r['setup_score'], implied, r['line'], raw_prob=r.get('p_over_raw'))''')

patch("10. Lock saves the raw probability too",
r'''                                implied, r['line'], min_max_proj, stat_proj_v, contrarian_v, context_v
                            )''',
r'''                                implied, r['line'], min_max_proj, stat_proj_v, contrarian_v, context_v,
                                raw_prob=r.get('win_prob_raw')
                            )''')

patch("11. save_to_ledger takes raw_prob",
r'''contrarian=0.0, context=0.0):''',
r'''contrarian=0.0, context=0.0, raw_prob=None):''')

patch("12. Ledger row has Raw_Prob",
r'''        "Context Proj": float(context)
    }''',
r'''        "Context Proj": float(context),
        "Raw_Prob": "" if raw_prob is None else round(float(raw_prob), 4)
    }''')

patch("13. Ledger save writes Raw_Prob",
r'''    append_to_sheet("ROI_Ledger", row, new_cols + ["Book", "Closing_Odds"])''',
r'''    append_to_sheet("ROI_Ledger", row, new_cols + ["Book", "Closing_Odds", "Raw_Prob"])''')

patch("14. Ledger load reads Raw_Prob",
r'''    df = load_sheet_df("ROI_Ledger", new_cols + ["Book", "Closing_Odds"])''',
r'''    df = load_sheet_df("ROI_Ledger", new_cols + ["Book", "Closing_Odds", "Raw_Prob"])''')

patch("15. Gate log accepts NBA",
r'''    if league != "NFL" or not isinstance(result, dict):''',
r'''    if league not in ("NFL", "NBA") or not isinstance(result, dict):''')

patch("16. Gate log override path handles NBA",
r'''        try:
            prob = nfl_prob_for_vote(result, forced_vote)
            g = nfl_gate_safe(result['df_ml'], result['s_col'], result['line'], forced_vote, prob, result.get('archetype', ''),
                              bool(st.session_state.get("nfl.role_risk", False)),
                              st.session_state.get("nfl.game_total"), st.session_state.get("nfl.game_total_open"))''',
r'''        try:
            if league == "NBA":
                _side = "p_over" if forced_vote == "OVER" else "p_under"
                prob = float(result.get(_side, result.get('win_prob', 0)))
                g = nba_gate_safe(result['df_ml'], result['s_col'], result['line'], forced_vote, prob,
                                  float(result.get(_side + "_raw", prob)), float(result.get('implied_prob', 0.5)),
                                  result.get('odds'), result.get('proj'), bool(st.session_state.get("nba.role_risk", False)))
            else:
                prob = nfl_prob_for_vote(result, forced_vote)
                g = nfl_gate_safe(result['df_ml'], result['s_col'], result['line'], forced_vote, prob, result.get('archetype', ''),
                                  bool(st.session_state.get("nfl.role_risk", False)),
                                  st.session_state.get("nfl.game_total"), st.session_state.get("nfl.game_total_open"))''')

patch("17. Gate log spread/total columns use the right league",
r'''    sp = st.session_state.get("nfl.spread", "")''',
r'''    sp = st.session_state.get("nba.spread" if league == "NBA" else "nfl.spread", "")''')

patch("18. Gate log total columns are NFL only",
r'''"Total": _blank_none(st.session_state.get("nfl.game_total")), "Total_Open": _blank_none(st.session_state.get("nfl.game_total_open")),''',
r'''"Total": _blank_none(st.session_state.get("nfl.game_total")) if league == "NFL" else "", "Total_Open": _blank_none(st.session_state.get("nfl.game_total_open")) if league == "NFL" else "",''')

patch("19. NFL gate report ignores NBA rows",
r'''def gate_report_summary(gate_df, ledger_df):
    # Joins Gate_Log to graded ROI_Ledger rows (Date + Player + Stat). Returns (summary_df, merged_df).
    if gate_df is None or gate_df.empty or ledger_df is None or ledger_df.empty:
        return None, None
    g = gate_df.copy()''',
r'''def gate_report_summary(gate_df, ledger_df, league="NFL"):
    # Joins Gate_Log to graded ROI_Ledger rows (Date + Player + Stat). Returns (summary_df, merged_df).
    if gate_df is None or gate_df.empty or ledger_df is None or ledger_df.empty:
        return None, None
    g = gate_df.copy()
    if 'League' in g.columns:
        g = g[g['League'].astype(str).str.strip().str.upper() == str(league).upper()]
        if g.empty:
            return None, None''')


def build_regex(old):
    lines = old.split("\n")
    return re.compile(r"[ \t]*\r?\n".join(re.escape(l) for l in lines))

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__); sys.exit(1)
    path = args[0]
    src = open(path, encoding="utf-8", newline="").read()
    if "league=None):\n    _non_bb" not in src:
        print("This file does not have v3.9 yet. Apply v2 through v3.9 first (or use app_patched_latest.py).")
        sys.exit(4)
    if "def nba_calibrated_prob(" in src:
        print("v4.0 Stage 1 is already applied to this file. Nothing to do.")
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
