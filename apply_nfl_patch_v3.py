#!/usr/bin/env python3
"""
B2TF Almanac - NFL discipline patch (v3). Apply AFTER apply_nfl_patch_v2.py.

Adds (agreed ruleset, Claude + Gemini):
  1. Leg gate on every NFL OVER/UNDER: model prob >= 62% AND L5 hit rate >= 4/5, opportunity floor
     (targets/carries/attempts), falling-usage check, Role Risk checkbox, low-total warnings.
  2. Spread + game total auto-filled from ESPN (with the open->current line move).
  3. Quantile-based alt lines (from the player's own L10) replace "proj - thresh - 0.5".
  4. Gate_Log worksheet: every NFL pick you lock/override is logged with the gate result and graded
     against the ROI ledger in a new "NFL Gate Report" (ROI tab > Deep Analytics). Gates are shown, NOT enforced.
  5. Parlay Builder: guardrails (1 RB per team, same-team pass-catcher stacking, opposite game scripts,
     low-total yardage Overs, failed-gate legs) and a break-even average-leg-probability check.
  6. NFL Data Check: parsed-games check (parsed games vs team games played).

Usage:   python apply_nfl_patch_v3.py app.py
Safety:  all-or-nothing. Every edit must match EXACTLY ONE place; otherwise nothing is written and the
         failures are listed. A backup (app.py.bak) is made and the result is syntax-checked.
Options: --lenient  skip edits that don't match (testing only)
"""
import re, sys, shutil, py_compile

HELPERS = r'''# ══════════════════════════════════════════════════════════════════════════════
# 🎟️ NFL PARLAY DISCIPLINE (v3): leg gate, game odds, quantile alts, gate log, guardrails
# ══════════════════════════════════════════════════════════════════════════════
NFL_GATE_MIN_PROB = 0.62      # model win prob needed for a leg to be parlay-eligible
NFL_GATE_MIN_L5   = 4         # L5 hit rate needed (4 of 5)
NFL_LOW_TOTAL     = 41.5      # at or below this, yardage Overs are fighting the script
# stat -> (opportunity column, minimum L5 average). Applies to OVER legs only.
NFL_OPP_FLOOR = {
    "REC_YDS": ("TARGETS", 4.5), "REC": ("TARGETS", 4.5),
    "RUSH_YDS": ("CARRIES", 8.0), "CARRIES": ("CARRIES", 8.0),
    "RUSH_REC": ("TOUCHES", 10.0),
    "PASS_YDS": ("PASS_ATT", 25.0), "COMP": ("PASS_ATT", 25.0),
    "PASS_TDS": ("PASS_ATT", 25.0), "INT": ("PASS_ATT", 25.0),
}
NFL_PASS_VOLUME_STATS = {"PASS_YDS", "PASS_TDS", "COMP", "REC_YDS", "REC"}
NFL_YARDAGE_STATS = {"PASS_YDS", "RUSH_YDS", "REC_YDS", "RUSH_REC"}
GATE_LOG_COLS = ["Date", "League", "Player", "Team", "Opp", "Position", "Stat", "Line", "Vote", "Win_Prob",
                 "L5_Hits", "L5_N", "L5_Median", "Opp_Avg", "Gate_OK", "Reasons", "Warnings",
                 "Spread", "Total", "Total_Open"]


def _nfl_pos(archetype):
    import re as _re
    m = _re.search(r'\b(QB|RB|WR|TE)\b', str(archetype or ""))
    return m.group(1) if m else ""


def nfl_leg_gate(df_ml, s_col, line, vote, win_prob, archetype="", role_risk=False, total=None, total_open=None):
    # Returns None when the gate does not apply (not an OVER/UNDER leg).
    if vote not in ("OVER", "UNDER") or s_col not in df_ml.columns:
        return None
    ser = pd.to_numeric(df_ml[s_col], errors='coerce').dropna()
    l5 = ser.tail(5)
    n = int(len(l5))
    reasons, warns = [], []

    if vote == "OVER":
        hits = int((l5 >= line).sum())
    else:
        hits = int((l5 <= line).sum())
    med = float(l5.median()) if n else 0.0

    if n < 5:
        reasons.append(f"only {n} recent games")
    elif hits < NFL_GATE_MIN_L5:
        reasons.append(f"L5 hit rate {hits}/5 (need {NFL_GATE_MIN_L5}/5)")
    if win_prob < NFL_GATE_MIN_PROB:
        reasons.append(f"model prob {win_prob*100:.0f}% (need {NFL_GATE_MIN_PROB*100:.0f}%)")

    opp_avg = None
    if vote == "OVER" and s_col in NFL_OPP_FLOOR:
        col, floor = NFL_OPP_FLOOR[s_col]
        if col in df_ml.columns:
            oc = pd.to_numeric(df_ml[col], errors='coerce').dropna()
            if len(oc) >= 3:
                opp_avg = float(oc.tail(5).mean())
                if opp_avg < floor:
                    reasons.append(f"low volume: L5 {col.lower()} {opp_avg:.1f} (need {floor:g}+)")
                if len(oc) >= 5:
                    l2 = float(oc.tail(2).mean())
                    prior = float(oc.iloc[-5:-2].mean())
                    if prior >= 3 and l2 < 0.70 * prior:
                        reasons.append(f"usage falling: L2 {col.lower()} {l2:.1f} vs {prior:.1f} before")

    if role_risk:
        reasons.append("role risk flagged (committee shift / returning starter)")

    if total is not None:
        try:
            t = float(total)
            if t <= NFL_LOW_TOTAL and vote == "OVER" and s_col in NFL_PASS_VOLUME_STATS | {"RUSH_REC"}:
                warns.append(f"low total {t:g}: this Over fights the script")
            if total_open is not None and float(total_open) - t >= 2.0:
                warns.append(f"total dropped {float(total_open) - t:.1f} pts since open")
        except Exception:
            pass

    return {"ok": len(reasons) == 0, "reasons": reasons, "warns": warns,
            "hits": hits, "n": n, "median": med, "opp_avg": opp_avg,
            "position": _nfl_pos(archetype)}


def nfl_quantile_alts(df, s_col):
    # Alt lines from the player's own last 10 games instead of proj +/- a fixed threshold.
    # Returns (over_1, over_2, under_1, under_2) as display strings with the L10 hit count.
    import math as _math
    ser = pd.to_numeric(df[s_col], errors='coerce').dropna().tail(10)
    if len(ser) < 5:
        raise ValueError("not enough games")
    is_yds = ("YDS" in s_col) or (s_col == "RUSH_REC")
    step = 10 if s_col == "PASS_YDS" else (5 if is_yds else 1)
    n = len(ser)

    def over(q):
        t = _math.floor(float(np.percentile(ser, q)) / step) * step
        t = max(t, step)
        return f"{t - 0.5:.1f} · L{n} {int((ser >= t).sum())}/{n}"

    def under(q):
        t = _math.ceil(float(np.percentile(ser, q)) / step) * step
        return f"{t + 0.5:.1f} · L{n} {int((ser <= t).sum())}/{n}"

    return over(30), over(15), under(70), under(85)


def _first_num(x):
    import re as _re
    m = _re.search(r'[-+]?\d+(?:\.\d+)?', str(x if x is not None else ""))
    return float(m.group(0)) if m else None


@st.cache_data(ttl=300)
def get_nfl_game_odds_map():
    # {TEAM: {spread (that team's perspective), spread_open, total, total_open, opp, home}} from ESPN's scoreboard.
    out = {}
    try:
        r = requests.get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard", timeout=10).json()
        for e in r.get('events', []):
            c = (e.get('competitions') or [{}])[0]
            odds = (c.get('odds') or [None])[0]
            if not odds:
                continue
            comps = c.get('competitors', [])
            home = next((t for t in comps if t.get('homeAway') == 'home'), None)
            away = next((t for t in comps if t.get('homeAway') == 'away'), None)
            if not home or not away:
                continue
            h = str(home.get('team', {}).get('abbreviation', '')).upper()
            a = str(away.get('team', {}).get('abbreviation', '')).upper()

            home_spread = None
            details = str(odds.get('details', '') or '').strip()
            if details.upper() in ("EVEN", "PK", "PICK", "PICK'EM"):
                home_spread = 0.0
            elif details:
                parts = details.split()
                num = _first_num(parts[-1]) if len(parts) >= 2 else None
                if num is not None:
                    home_spread = num if parts[0].upper() == h else (-num if parts[0].upper() == a else None)
            if home_spread is None:
                home_spread = _first_num(odds.get('spread'))
            open_home = _first_num((((odds.get('pointSpread') or {}).get('home') or {}).get('open') or {}).get('line'))
            total = _first_num(odds.get('overUnder'))
            total_open = _first_num(((((odds.get('total') or {}).get('over') or {}).get('open')) or {}).get('line'))

            for team, opp, sign, is_home in ((h, a, 1, True), (a, h, -1, False)):
                out[team] = {
                    "spread": None if home_spread is None else sign * home_spread,
                    "spread_open": None if open_home is None else sign * open_home,
                    "total": total, "total_open": total_open, "opp": opp, "home": is_home,
                }
    except Exception:
        pass
    return out


@st.cache_data(ttl=1800)
def get_nfl_team_games():
    # {TEAM: regular-season games played} from ESPN standings (wins + losses + ties).
    out = {}
    try:
        r = requests.get("https://site.api.espn.com/apis/v2/sports/football/nfl/standings", timeout=8).json()

        def walk(node):
            for ent in (node.get('standings') or {}).get('entries', []):
                s = {x.get('name'): x.get('value') for x in ent.get('stats', []) if x.get('name')}
                abbr = str(ent.get('team', {}).get('abbreviation', '')).upper()
                gp = (s.get('wins') or 0) + (s.get('losses') or 0) + (s.get('ties') or 0)
                if abbr:
                    out[abbr] = int(gp)
            for ch in node.get('children', []):
                walk(ch)
        walk(r)
    except Exception:
        pass
    return out


@st.cache_data(ttl=120)
def load_gate_log():
    gc = get_gc()
    if not gc:
        return pd.DataFrame(columns=GATE_LOG_COLS)
    try:
        ws = gc.open("B2TF_Database").worksheet("Gate_Log")
        df = pd.DataFrame(ws.get_all_records())
        for c in GATE_LOG_COLS:
            if c not in df.columns:
                df[c] = ""
        df = df[GATE_LOG_COLS].fillna("")
        for c in ("Date", "Player", "Stat", "Team", "Opp", "Gate_OK", "Position"):
            df[c] = df[c].astype(str).str.strip()
        return df
    except Exception:
        return pd.DataFrame(columns=GATE_LOG_COLS)


def append_gate_log(row):
    gc = get_gc()
    if not gc:
        return
    try:
        sh = gc.open("B2TF_Database")
        try:
            ws = sh.worksheet("Gate_Log")
        except Exception:
            ws = sh.add_worksheet(title="Gate_Log", rows=1000, cols=len(GATE_LOG_COLS))
            ws.append_row(GATE_LOG_COLS)
        clean = []
        for col in GATE_LOG_COLS:
            v = row.get(col, "")
            clean.append(("TRUE" if v else "FALSE") if isinstance(v, bool) else v)
        ws.append_row(clean, value_input_option='USER_ENTERED')
        load_gate_log.clear()
    except Exception as e:
        st.warning(f"Gate log not saved: {e}")


def _blank_none(v):
    return "" if v is None else v


def log_gate_row(league, player_label, result, opp):
    # Called when a pick is locked/overridden. NFL legs only (other leagues have no gate).
    g = result.get('gate') if isinstance(result, dict) else None
    if not g:
        return
    team = str(player_label).split("(")[1].split(")")[0].strip().upper() if "(" in str(player_label) else ""
    sp = st.session_state.get("nfl.spread", "")
    row = {
        "Date": datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d"),
        "League": league, "Player": str(player_label).split('(')[0].strip(), "Team": team,
        "Opp": result.get('opp', opp), "Position": g.get('position', ''), "Stat": result.get('stat_type', ''),
        "Line": float(result.get('line', 0)), "Vote": result.get('vote', ''),
        "Win_Prob": round(float(result.get('win_prob', 0)), 4),
        "L5_Hits": g.get('hits', ''), "L5_N": g.get('n', ''), "L5_Median": round(g.get('median', 0.0), 2),
        "Opp_Avg": "" if g.get('opp_avg') is None else round(g['opp_avg'], 2),
        "Gate_OK": bool(g.get('ok')), "Reasons": "; ".join(g.get('reasons', [])), "Warnings": "; ".join(g.get('warns', [])),
        "Spread": sp, "Total": _blank_none(st.session_state.get("nfl.game_total")), "Total_Open": _blank_none(st.session_state.get("nfl.game_total_open")),
    }
    append_gate_log(row)


def parlay_guardrail_flags(selected_labels, pick_meta, gate_df):
    # Pure logic (no Streamlit) so it can be tested. Returns a list of (level, message); level in error/warn/info.
    flags = []
    legs = []
    for lab in selected_labels:
        m = pick_meta.get(lab, {}) or {}
        leg = {"label": lab, "league": str(m.get('League', '')), "player": str(m.get('Player', '')).strip(),
               "stat": str(m.get('Stat', '')).strip(), "date": str(m.get('Date', '')).strip(),
               "vote": str(m.get('Vote', '')).strip(), "g": None}
        if leg["league"] == "NFL" and gate_df is not None and not gate_df.empty:
            hit = gate_df[(gate_df['Player'] == leg["player"]) & (gate_df['Stat'] == leg["stat"]) & (gate_df['Date'] == leg["date"])]
            if not hit.empty:
                leg["g"] = hit.iloc[-1].to_dict()
        legs.append(leg)

    nfl = [l for l in legs if l["league"] == "NFL"]
    if not nfl:
        return flags
    sc = {"Passing Yards": "PASS_YDS", "Passing TDs": "PASS_TDS", "Completions": "COMP", "Interceptions": "INT",
          "Rushing Yards": "RUSH_YDS", "Receiving Yards": "REC_YDS", "Receptions": "REC", "Carries": "CARRIES",
          "Rush+Rec Yards": "RUSH_REC"}
    for l in nfl:
        l["col"] = sc.get(l["stat"], "")

    # 1) one RB leg per team
    rb_by_team = {}
    for l in nfl:
        if l["g"] and str(l["g"].get("Position")) == "RB":
            rb_by_team.setdefault(str(l["g"].get("Team")), set()).add(l["player"])
    for team, names in rb_by_team.items():
        if len(names) > 1:
            flags.append(("error", f"Two {team} running backs on one slip ({', '.join(sorted(names))}). They split the same snaps. Keep one RB leg per team."))

    # 2) several pass-catchers from one team compete for the same targets
    rec_by_team = {}
    for l in nfl:
        if l["g"] and l["col"] in ("REC_YDS", "REC", "RUSH_REC") and str(l["g"].get("Position")) in ("WR", "TE", "RB"):
            rec_by_team.setdefault(str(l["g"].get("Team")), set()).add(l["player"])
    for team, names in rec_by_team.items():
        if len(names) >= 3:
            flags.append(("warn", f"{len(names)} {team} pass-catchers ({', '.join(sorted(names))}). Targets are zero-sum; keep one target-alpha per team."))

    # 3) opposite game scripts inside one game
    for i in range(len(nfl)):
        for j in range(i + 1, len(nfl)):
            a, b = nfl[i], nfl[j]
            if not (a["g"] and b["g"]): continue
            if str(a["g"].get("Team")) == str(b["g"].get("Opp")) and str(a["g"].get("Opp")) == str(b["g"].get("Team")):
                if a["vote"] == "OVER" and b["vote"] == "OVER" and a["col"] in NFL_PASS_VOLUME_STATS and b["col"] in NFL_PASS_VOLUME_STATS:
                    try:
                        sa = float(a["g"].get("Spread")); sb = float(b["g"].get("Spread"))
                    except Exception:
                        continue
                    fav = a if sa <= -3 else (b if sb <= -3 else None)
                    if fav is not None:
                        flags.append(("warn", f"Script conflict: {a['player']} and {b['player']} both need pass volume, but one team is a {abs(min(sa, sb)):g}-pt favorite and tends to run when ahead."))

    # 4) low total with several yardage Overs
    low = [l for l in nfl if l["g"] and l["vote"] == "OVER" and l["col"] in NFL_YARDAGE_STATS]
    try:
        tot = [float(l["g"].get("Total")) for l in low if str(l["g"].get("Total")) not in ("", "None")]
    except Exception:
        tot = []
    if tot and min(tot) <= NFL_LOW_TOTAL and len(low) >= 2:
        flags.append(("warn", f"Game total {min(tot):g} (at or below {NFL_LOW_TOTAL:g}) with {len(low)} yardage Overs. Cap it at one high-target Over per team."))

    # 5) legs that failed the gate (or have no gate record)
    for l in nfl:
        if l["g"] is None:
            flags.append(("info", f"{l['player']} {l['stat']}: no gate record (locked before the gate existed)."))
        elif str(l["g"].get("Gate_OK")).upper() != "TRUE":
            flags.append(("warn", f"{l['player']} {l['stat']} failed the gate: {l['g'].get('Reasons')}"))
    return flags


def render_parlay_guardrails(selected_labels, pick_meta):
    if len(selected_labels) < 2:
        return
    flags = parlay_guardrail_flags(selected_labels, pick_meta, load_gate_log())
    with st.expander(f"🛡️ Parlay Guardrails ({len([f for f in flags if f[0] != 'info'])} flag(s))", expanded=any(f[0] in ('error', 'warn') for f in flags)):
        if not flags:
            st.success("No guardrail flags on this ticket.")
        for lvl, msg in flags:
            (st.error if lvl == "error" else st.warning if lvl == "warn" else st.caption)(msg)


def render_breakeven_check(final_odds, selected_labels, prob_map):
    n = len(selected_labels)
    if n < 2:
        return
    try:
        o = float(final_odds)
        implied = (abs(o) / (abs(o) + 100.0)) if o < 0 else (100.0 / (o + 100.0))
        need = implied ** (1.0 / n)
        probs = [max(0.01, min(0.99, float(prob_map.get(l, 0.55)))) for l in selected_labels]
        avg = float(np.exp(np.mean(np.log(probs))))
        msg = (f"⚖️ **Break-even:** at {int(o):+d} over {n} legs, the average leg must hit **{need*100:.0f}%**. "
               f"Your legs average **{avg*100:.0f}%** by the model (independent legs, uncalibrated).")
        (st.success if avg >= need + 0.02 else st.warning)(msg)
    except Exception:
        pass

def gate_report_summary(gate_df, ledger_df):
    # Joins Gate_Log to graded ROI_Ledger rows (Date + Player + Stat). Returns (summary_df, merged_df).
    if gate_df is None or gate_df.empty or ledger_df is None or ledger_df.empty:
        return None, None
    g = gate_df.copy()
    led = ledger_df[ledger_df['Result'].isin(['Win', 'Loss'])][['Date', 'Player', 'Stat', 'Result']].copy()
    for c in ("Date", "Player", "Stat"):
        g[c] = g[c].astype(str).str.strip()
        led[c] = led[c].astype(str).str.strip()
    led = led.drop_duplicates(['Date', 'Player', 'Stat'], keep='last')
    m = g.merge(led, on=['Date', 'Player', 'Stat'], how='left')
    gr = m[m['Result'].isin(['Win', 'Loss'])]
    rows = []
    for label, flag in (("Passed gate", "TRUE"), ("Failed gate", "FALSE")):
        sub = gr[gr['Gate_OK'].astype(str).str.upper() == flag]
        n = len(sub)
        w = int((sub['Result'] == 'Win').sum())
        rows.append({"Group": label, "Graded legs": n, "Wins": w, "Win %": round(100.0 * w / n, 1) if n else None})
    return pd.DataFrame(rows), m

# ══════════════════════════════════════════════════════════════════════════════


def nfl_gate_badge_html(g):
    # Small HTML line for the result card. g is the dict from nfl_leg_gate (or None for PASS/other leagues).
    import html as _html
    if not g:
        return ""
    if g.get('ok'):
        color, head = "#00c853", f"GATE ✓ &nbsp;L5 {g['hits']}/{g['n']}"
    else:
        color, head = "#ff5252", f"GATE ✗ &nbsp;L5 {g['hits']}/{g['n']}"
    bits = list(g.get('reasons', [])) + ["⚠ " + w for w in g.get('warns', [])]
    tail = (" &nbsp;|&nbsp; " + _html.escape("; ".join(bits))) if bits else ""
    return f"<div style='font-size:10px;margin-top:2px;color:{color};font-weight:700;'>{head}{tail}</div>"


def nfl_autofill_game_odds(player_label):
    # Fills the spread box (and stores totals) from ESPN for the player's team. Runs before the spread widget is drawn.
    team = str(player_label).split("(")[1].split(")")[0].strip().upper() if "(" in str(player_label) else ""
    info = get_nfl_game_odds_map().get(team)
    for k in ("nfl.spread_open", "nfl.game_total", "nfl.game_total_open"):
        st.session_state[k] = None
    if not info:
        return
    if info.get('spread') is not None:
        st.session_state["nfl.spread"] = float(max(-30.0, min(30.0, info['spread'])))
    st.session_state["nfl.spread_open"] = info.get('spread_open')
    st.session_state["nfl.game_total"] = info.get('total')
    st.session_state["nfl.game_total_open"] = info.get('total_open')


def nfl_game_odds_caption():
    sp, spo = st.session_state.get("nfl.spread"), st.session_state.get("nfl.spread_open")
    tot, toto = st.session_state.get("nfl.game_total"), st.session_state.get("nfl.game_total_open")
    if tot is None and spo is None:
        return ""
    parts = []
    if tot is not None:
        mv = f" (open {toto:g})" if toto is not None and toto != tot else ""
        parts.append(f"Total {tot:g}{mv}")
    if spo is not None and sp is not None and float(spo) != float(sp):
        parts.append(f"line moved {float(spo):+g} → {float(sp):+g}")
    return " · ".join(parts)


def nfl_parsed_games_check(df, player_label):
    # Sanity check for the Data Check panel: how many of this season's games did the parser find vs the team's games played?
    team = str(player_label).split("(")[1].split(")")[0].strip().upper() if "(" in str(player_label) else ""
    now = datetime.now(pytz.timezone('US/Eastern')).replace(tzinfo=None)
    season_start = pd.Timestamp(year=now.year if now.month >= 8 else now.year - 1, month=8, day=15)
    cur = df[pd.to_datetime(df['ValidDate']) >= season_start]
    got = len(cur)
    tg = get_nfl_team_games().get(team)
    dates = sorted(pd.to_datetime(cur['ValidDate']).tolist())
    gaps = [(dates[i - 1], dates[i]) for i in range(1, len(dates)) if (dates[i] - dates[i - 1]).days > 15]
    if tg is None:
        return "info", f"Parsed {got} game(s) this season. Could not load {team or 'team'} games played from ESPN standings to compare."
    if got >= tg:
        return "ok", f"Parsed {got} game(s) this season vs {team} games played {tg}. Looks complete."
    msg = (f"Parsed {got} game(s) this season but {team} has played {tg}. If he was active in the missing week(s), "
           f"the parser dropped a game - send me the player and week.")
    if gaps:
        msg += " Gap(s) over 2 weeks: " + ", ".join(f"{a:%b %d} → {b:%b %d}" for a, b in gaps) + "."
    return "warn", msg


def nfl_gate_safe(*a, **k):
    # Never let a gate problem break the analysis run.
    try:
        return nfl_leg_gate(*a, **k)
    except Exception:
        return None'''

PATCHES = []
def patch(name, old, new):
    PATCHES.append((name, old, new))

# 1 ---------------------------------------------------------------- helper block
patch("v3 helpers (gate, ESPN odds, alts, gate log, guardrails)",
r'''def render_scoreboard(sd, league_name):''',
HELPERS + r'''


def render_scoreboard(sd, league_name):''')

# 2 ---------------------------------------------------------------- spread auto-fill
patch("Spread box: give it a session default (so ESPN can fill it)",
r'''spread_input = st.number_input("Spread", min_value=-30.0, max_value=30.0, value=0.0, step=0.5, key=f"{lk}.spread", format="%.1f", label_visibility="collapsed")''',
r'''init_state(f"{lk}.spread", 0.0)
            spread_input = st.number_input("Spread", min_value=-30.0, max_value=30.0, step=0.5, key=f"{lk}.spread", format="%.1f", label_visibility="collapsed")''')

patch("Spread box: show total + line move under it",
r'''st.markdown(f"<div style='font-size:11px;font-weight:700;color:{sh_color};text-align:center;margin-top:2px;'>{sh_text}</div>", unsafe_allow_html=True)''',
r'''st.markdown(f"<div style='font-size:11px;font-weight:700;color:{sh_color};text-align:center;margin-top:2px;'>{sh_text}</div>", unsafe_allow_html=True)
            if league_key == "NFL":
                _oc = nfl_game_odds_caption()
                if _oc: st.caption(_oc)''')

patch("Auto-fill spread/total when the player changes",
r'''# 💥 THE BUG FIX: Indented so it ONLY updates the toggle when the player actually changes!
            if player_name and player_name != st.session_state.get(f"{lk}.last_player"):
                st.session_state[f"{lk}.last_player"] = player_name
                if auto_opp and auto_opp in teams:
                    st.session_state[f"{lk}.opp"]     = auto_opp
                    st.session_state[f"{lk}.is_home"] = auto_is_home''',
r'''# 💥 THE BUG FIX: Indented so it ONLY updates the toggle when the player actually changes!
            if player_name and player_name != st.session_state.get(f"{lk}.last_player"):
                st.session_state[f"{lk}.last_player"] = player_name
                if auto_opp and auto_opp in teams:
                    st.session_state[f"{lk}.opp"]     = auto_opp
                    st.session_state[f"{lk}.is_home"] = auto_is_home
                if league_key == "NFL":
                    nfl_autofill_game_odds(player_name)''')

# 3 ---------------------------------------------------------------- role risk
patch("Role Risk checkbox (NFL)",
r'''st.session_state[f"{lk}.injury_boost"] = teammate_out''',
r'''st.session_state[f"{lk}.injury_boost"] = teammate_out
            if league_key == "NFL":
                st.checkbox("⚠️ Role Risk", key=f"{lk}.role_risk",
                            help="Returning from injury, snap share falling, or a new committee. Over legs for this player fail the gate.")''')

# 4 ---------------------------------------------------------------- gate on results
patch("Compute the leg gate for each NFL result",
r'''"l5_avg_val": round(float(df_ml.tail(5)[s_col].mean()), 1),
                })''',
r'''"l5_avg_val": round(float(df_ml.tail(5)[s_col].mean()), 1),
                    "gate": (nfl_gate_safe(df_ml, s_col, line, c_vote, win_prob, archetype,
                                           bool(st.session_state.get(f"{lk}.role_risk", False)),
                                           st.session_state.get("nfl.game_total"), st.session_state.get("nfl.game_total_open"))
                             if league_key == "NFL" else None),
                })''')

patch("Gate badge on the result card: build",
r'''top_badge = " 🏆 TOP PICK" if is_top else ""''',
r'''top_badge = " 🏆 TOP PICK" if is_top else ""
                gate_html = nfl_gate_badge_html(result.get('gate')) if league_key == "NFL" else ""''')

patch("Gate badge on the result card: show",
r'''Win Prob {result['win_prob']*100:.1f}%</div>''',
r'''Win Prob {result['win_prob']*100:.1f}%</div>
                    {gate_html}''')

# 5 ---------------------------------------------------------------- quantile alts
patch("Quantile alt lines for NFL",
r'''alt_under_2 = round(proj + thresh + 1.0, 1)''',
r'''alt_under_2 = round(proj + thresh + 1.0, 1)
                if league_key == "NFL":
                    try:
                        alt_over_1, alt_over_2, alt_under_1, alt_under_2 = nfl_quantile_alts(result['df_ml'], s_col)
                    except Exception:
                        pass''')

# 6 ---------------------------------------------------------------- gate log on lock / override
patch("Gate log: on Lock Selected",
r'''log_prediction_receipt(target_player, r['stat_type'], r['proj'], datetime.now().strftime("%Y-%m-%d"))
                            locked += 1''',
r'''log_prediction_receipt(target_player, r['stat_type'], r['proj'], datetime.now().strftime("%Y-%m-%d"))
                            log_gate_row(league_key, target_player, r, opp)
                            locked += 1''')

patch("Gate log: on Override",
r'''log_prediction_receipt(target_player, r['stat_type'], r['proj'], datetime.now().strftime("%Y-%m-%d"), is_override=True)''',
r'''log_prediction_receipt(target_player, r['stat_type'], r['proj'], datetime.now().strftime("%Y-%m-%d"), is_override=True)
                        log_gate_row(league_key, target_player, r, opp)''')

# 7 ---------------------------------------------------------------- parlay builder
patch("Parlay builder: pick metadata (init)",
r'''pick_options, pick_odds_map, pick_prob_map = [], {}, {}''',
r'''pick_options, pick_odds_map, pick_prob_map = [], {}, {}
                pick_meta = {}''')

patch("Parlay builder: pick metadata (fill)",
r'''pick_prob_map[label] = prob''',
r'''pick_prob_map[label] = prob
                    pick_meta[label] = {k: str(r.get(k, "")) for k in ("League", "Player", "Stat", "Date", "Vote")}''')

patch("Parlay builder: guardrails + break-even",
r'''st.info(f"💸 **Projected Payout:** ${(proj_profit if p_free else p_risk + proj_profit):.2f} (Profit: ${proj_profit:.2f})")''',
r'''st.info(f"💸 **Projected Payout:** ${(proj_profit if p_free else p_risk + proj_profit):.2f} (Profit: ${proj_profit:.2f})")
                render_breakeven_check(p_odds, selected_picks, pick_prob_map)
                render_parlay_guardrails(selected_picks, pick_meta)''')

# 8 ---------------------------------------------------------------- data check
patch("NFL Data Check: parsed-games check",
r'''st.dataframe(_dfc[_cols].tail(8), hide_index=True, use_container_width=True)''',
r'''st.dataframe(_dfc[_cols].tail(8), hide_index=True, use_container_width=True)
                _lvl, _msg = nfl_parsed_games_check(_dfc, _tp)
                (st.success if _lvl == "ok" else st.warning if _lvl == "warn" else st.caption)(_msg)''')

# 9 ---------------------------------------------------------------- gate report
patch("ROI tab: NFL Gate Report",
r'''with roi_tab_data:
                # CHARTS & LEAK FINDER''',
r'''with roi_tab_data:
                with st.expander("🛡️ NFL Gate Report (does the gate actually predict wins?)", expanded=False):
                    _gsum, _gmerged = gate_report_summary(load_gate_log(), ledger_df)
                    if _gsum is None:
                        st.caption("No gate-logged NFL picks yet. Lock NFL picks from the board and let them grade; this fills in over a few weeks.")
                    else:
                        st.dataframe(_gsum, hide_index=True, use_container_width=True)
                        st.caption("Rule 8: don't trust the gate with real money until 'Passed gate' clearly beats 'Failed gate' over several weeks.")
                        _gcols = [c for c in ["Date", "Player", "Stat", "Line", "Vote", "Win_Prob", "L5_Hits", "Gate_OK", "Reasons", "Result"] if c in _gmerged.columns]
                        st.dataframe(_gmerged[_gcols].tail(40), hide_index=True, use_container_width=True)
                # CHARTS & LEAK FINDER''')


# ======================================================================== engine
def build_regex(old):
    """Match `old` exactly, but tolerate trailing spaces on lines and CRLF line endings."""
    lines = old.split("\n")
    return re.compile(r"[ \t]*\r?\n".join(re.escape(l) for l in lines))

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    lenient = "--lenient" in sys.argv
    if not args:
        print(__doc__); sys.exit(1)
    path = args[0]
    src = open(path, encoding="utf-8", newline="").read()

    if "NFL_ELITE_DEF_FALLBACK" not in src:
        print("This file does not have the v2 patch yet (no NFL_ELITE_DEF_FALLBACK). Run apply_nfl_patch_v2.py first.")
        sys.exit(4)
    if "NFL PARLAY DISCIPLINE (v3)" in src:
        print("v3 is already applied to this file. Nothing to do.")
        sys.exit(0)

    out, failures, applied = src, [], []
    for name, old, new in PATCHES:
        rx = build_regex(old)
        hits = list(rx.finditer(out))
        if len(hits) != 1:
            failures.append((name, len(hits)))
            continue
        m = hits[0]
        out = out[:m.start()] + new + out[m.end():]
        applied.append(name)

    for n in applied:
        print(f"  OK       {n}")
    for n, c in failures:
        print(f"  {'SKIPPED' if lenient else 'FAILED '}  {n}  (matched {c}x, expected exactly 1)")

    if failures and not lenient:
        print(f"\n{len(failures)} edit(s) did not match. NOTHING was written. Send me the failed names above.")
        sys.exit(2)

    if lenient:
        dest = path + ".patched"
    else:
        shutil.copyfile(path, path + ".bak")
        dest = path
    open(dest, "w", encoding="utf-8", newline="").write(out)
    try:
        py_compile.compile(dest, doraise=True)
        print(f"\nSyntax check passed. Wrote {dest}" + ("" if lenient else f"  (backup: {path}.bak)"))
    except py_compile.PyCompileError as e:
        print(f"\nSYNTAX ERROR after patching: {e}")
        if not lenient:
            shutil.copyfile(path + ".bak", path)
            print("Original restored.")
        sys.exit(3)

if __name__ == "__main__":
    main()
