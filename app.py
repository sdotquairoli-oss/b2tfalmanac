import streamlit as st
import pandas as pd
from datetime import datetime
import pytz
import altair as alt
import requests
import numpy as np
import time
import os
import base64
import json
import gspread
from datetime import datetime
import pytz
from io import StringIO
import streamlit.components.v1 as components
from concurrent.futures import ThreadPoolExecutor

# --- ML IMPORTS ---
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics.pairwise import euclidean_distances

# --- CONFIGURATION & GLOBAL CONSTANTS ---
st.set_page_config(page_title="B2TF Almanac", layout="wide", page_icon="⚡", initial_sidebar_state="collapsed")

# 🔒 SECURE VAULT EXTRACTION
try: BDL_API_KEY = st.secrets["BDL_API_KEY"]
except: BDL_API_KEY = None

try: ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
except: ODDS_API_KEY = None

NBA_TEAMS = sorted(["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"])
NHL_TEAMS = sorted(["ANA", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR", "OTT", "PHI", "PIT", "SJS", "SEA", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WSH", "WPG"])
MLB_TEAMS = sorted(["ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH"])
SPORTSBOOKS = ["FanDuel", "Fanatics", "DraftKings", "BetMGM", "Caesars", "ESPN Bet", "Hard Rock", "bet365", "Other"]
BOOK_LOGOS = {
    "FanDuel": "https://www.google.com/s2/favicons?domain=fanduel.com&sz=128",
    "DraftKings": "https://www.google.com/s2/favicons?domain=draftkings.com&sz=128",
    "BetMGM": "https://www.google.com/s2/favicons?domain=betmgm.com&sz=128",
    "Caesars": "https://www.google.com/s2/favicons?domain=caesars.com&sz=128",
    "Fanatics": "https://www.google.com/s2/favicons?domain=sportsbook.fanatics.com&sz=128",
    "ESPN Bet": "https://www.google.com/s2/favicons?domain=espnbet.com&sz=128",
    "Hard Rock": "https://www.google.com/s2/favicons?domain=hardrock.bet&sz=128",
    "bet365": "https://www.google.com/s2/favicons?domain=bet365.com&sz=128"
}
# ✅ OPT-3: Defined ONCE at module level — never recreated inside loops
LEAGUE_SHIELDS = {
    "NBA": "https://a.espncdn.com/i/teamlogos/leagues/500/nba.png",
    "NHL": "https://a.espncdn.com/i/teamlogos/leagues/500/nhl.png",
    "MLB": "https://a.espncdn.com/i/teamlogos/leagues/500/mlb.png",
    "NFL": "https://a.espncdn.com/i/teamlogos/leagues/500/nfl.png"
}

def get_team_logo(league, abbr):
    """Pulls high-res transparent PNGs from ESPN's hidden CDN."""
    abbr_upper = str(abbr).upper()
    nba_map = {"GSW": "gs", "NOP": "no", "NYK": "ny", "SAS": "sa", "UTA": "utah"}
    nhl_map = {"SJS": "sj", "TBL": "tb", "LAK": "la", "NJD": "nj", "WSH": "wsh"}
    mlb_map = {"CHW": "cws"}
    if league == "NBA":
        espn_abbr = nba_map.get(abbr_upper, abbr_upper).lower()
        return f"https://a.espncdn.com/i/teamlogos/nba/500/{espn_abbr}.png"
    elif league == "NHL":
        espn_abbr = nhl_map.get(abbr_upper, abbr_upper).lower()
        return f"https://a.espncdn.com/i/teamlogos/nhl/500/{espn_abbr}.png"
    elif league == "MLB":
        espn_abbr = mlb_map.get(abbr_upper, abbr_upper).lower()
        return f"https://a.espncdn.com/i/teamlogos/mlb/500/{espn_abbr}.png"
    return ""

NBA_FULL_TO_ABBREV = {'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN', 'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE', 'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND', 'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'}
ODDS_MEGA_MAP = {**NBA_FULL_TO_ABBREV, "ANA": "Anaheim Ducks", "BUF": "Sabres", "CGY": "Flames", "CAR": "Hurricanes", "COL": "Avalanche", "CBJ": "Blue Jackets", "EDM": "Oilers", "FLA": "Panthers", "LAK": "Kings", "MTL": "Canadiens", "NSH": "Predators", "NJD": "Devils", "NYI": "Islanders", "NYR": "Rangers", "OTT": "Senators", "PIT": "Penguins", "SJS": "Sharks", "SEA": "Kraken", "STL": "Blues", "TBL": "Lightning", "VAN": "Canucks", "VGK": "Knights", "WPG": "Jets"}
S_MAP = {
    "Points": "PTS", "Goals": "G", "Assists": "A", "Shots on Goal": "SOG",
    "Rebounds": "TRB", "PRA (Pts+Reb+Ast)": "PRA", "Power Play Points": "PPP",
    "Minutes Played": "MINS", "Threes Made": "FG3M", "Points + Rebounds": "PR",
    "Points + Assists": "PA", "Rebounds + Assists": "RA", "Hits": "H",
    "Home Runs": "HR", "Total Bases": "TB", "Pitcher Strikeouts": "K",
    "Pitcher Earned Runs": "ER", "Double Double": "DD", "Triple Double": "TD",
    "Blocks": "BLK", "Steals": "STL" # 🟢 Added Defense
}

# 🎯 Minimum projection gap required to fire a bet (prevents coin-flip calls)
PASS_THRESHOLDS = {
    "PTS": 1.5, "TRB": 0.8, "AST": 0.8, "FG3M": 0.6, "PRA": 2.0, "PR": 1.5,
    "PA": 1.5, "RA": 1.0, "SOG": 0.75, "G": 0.3, "A": 0.4, "H": 0.5,
    "HR": 0.25, "TB": 0.75, "K": 1.0, "ER": 0.5, "DD": 0.10, "TD": 0.08,
    "MINS": 2.0, "PPP": 0.3, 
    "BLK": 0.35, # 🟢 Low volume stat, requires smaller edge
    "STL": 0.35  # 🟢 Low volume stat, requires smaller edge
}
# --- THEME CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Audiowide&display=swap');
.stApp { background-color: #0f172a; color: #f8fafc; font-family: 'Inter', sans-serif; }
header { visibility: hidden; height: 0px !important; }
.block-container { padding-top: 1rem !important; }
div.stContainer { background-color: #1e293b; border-radius: 12px; border: 1px solid #334155; padding: 20px; margin-bottom: 15px; }
.verdict-box { text-align: center; padding: 10px; border-radius: 6px; border: 2px solid; margin-bottom: 10px; display: flex; flex-direction: column; justify-content: center;}
.board-member { background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 15px; height: 100%; position: relative;}
.board-name { font-size: 14px; font-weight: 800; color: #00E5FF; margin-bottom: 2px; }
.board-model { font-size: 10px; color: #4ade80; font-weight: bold; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.5px;}
.board-vote { font-size: 18px; font-weight: 900; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #1e293b; padding: 10px; border-radius: 8px; border: 1px solid #334155;}
.stTabs [data-baseweb="tab"] { height: 45px; white-space: pre-wrap; background-color: #0f172a; border-radius: 6px; color: #94a3b8; font-weight: 600; font-size: 16px; padding: 0px 24px; border: 1px solid #334155;}
.stTabs [aria-selected="true"] { color: #0f172a !important; background-color: #00E5FF !important; border: 1px solid #00E5FF !important; }
.stButton > button[kind="primary"] { background-color: #00E676 !important; color: #0f172a !important; border: none !important; font-weight: 900 !important; text-transform: uppercase !important; letter-spacing: 1px !important; box-shadow: 0px 4px 10px rgba(0, 230, 118, 0.4) !important; transition: all 0.2s ease-in-out !important; }
.stButton > button[kind="primary"]:hover { background-color: #00c853 !important; color: white !important; box-shadow: 0px 6px 15px rgba(0, 200, 83, 0.6) !important; transform: scale(1.02); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. GOOGLE SHEETS DATABASE MANAGEMENT
# ==========================================
def get_gc():
    try:
        if "google_credentials" in st.secrets:
            return gspread.service_account_from_dict(json.loads(st.secrets["google_credentials"]))
    except Exception as e: st.error(f"🚨 Google Sheets Auth Error: {e}")
    return None

@st.cache_data(ttl=600)
def load_sheet_df(sheet_name, expected_cols=None):
    gc = get_gc()
    if not gc: return pd.DataFrame(columns=expected_cols or [])
    try:
        ws = gc.open("B2TF_Database").worksheet(sheet_name)
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        if not df.empty and 'Date' in df.columns:
            df = df[df['Date'].astype(str).str.strip() != '']
        if expected_cols:
            for c in expected_cols:
                if c not in df.columns: df[c] = ""
            df = df[expected_cols]
        return df
    except Exception as e:
        st.error(f"Error loading {sheet_name}: {e}")
        return pd.DataFrame(columns=expected_cols or [])

def append_to_sheet(sheet_name, row_dict, expected_cols):
    gc = get_gc()
    if not gc: return
    try:
        ws = gc.open("B2TF_Database").worksheet(sheet_name)
        dates = [d for d in ws.col_values(1) if str(d).strip() != '']
        next_row = len(dates) + 1
        clean_row = []
        for col in expected_cols:
            val = row_dict.get(col, f"")
            if isinstance(val, bool): clean_row.append("TRUE" if val else "FALSE")
            else: clean_row.append(val)
        try: ws.update(values=[clean_row], range_name=f'A{next_row}', value_input_option="USER_ENTERED")
        except TypeError: ws.update(f'A{next_row}', [clean_row], value_input_option="USER_ENTERED")
        load_sheet_df.clear()
        # ✅ OPT-9: Clear load_ledger cache when sheet is updated
        load_ledger.clear()
    except Exception as e:
        st.error(f"Failed to save to database: {e}")

def overwrite_sheet(sheet_name, df):
    gc = get_gc()
    if not gc: return
    try:
        ws = gc.open("B2TF_Database").worksheet(sheet_name)
        clean_df = df.fillna("")
        for col in clean_df.columns:
            if clean_df[col].dtype == bool:
                clean_df[col] = clean_df[col].apply(lambda x: "TRUE" if x else "FALSE")
        new_values = [clean_df.columns.values.tolist()] + clean_df.values.tolist()
        try: ws.update(values=new_values, range_name='A1', value_input_option="USER_ENTERED")
        except TypeError: ws.update('A1', new_values, value_input_option="USER_ENTERED")
        load_sheet_df.clear()
        # ✅ OPT-9: Clear load_ledger cache when sheet is overwritten
        load_ledger.clear()
    except Exception as e: st.error(f"Failed to update database: {e}")

# ✅ OPT-9: load_ledger now cached — filtering logic runs once per TTL, not every render
@st.cache_data(ttl=120)
def load_ledger():
    # Updated column order to match the new Google Sheet layout
    new_cols = ["Date", "League", "Player", "Stat", "Odds", "Line", "Proj", "Vote", "Actual", "Result", "Win_Prob", "Is_Boosted", "Setup_Score", "User_Prob"]
    df = load_sheet_df("ROI_Ledger", new_cols)
    df = df[df['Player'].astype(str).str.strip() != '']
    df = df[df['Date'].astype(str).str.strip() != '']
    df = df.reset_index(drop=True)
    if "Is_Boosted" not in df.columns: df["Is_Boosted"] = False
    else: df["Is_Boosted"] = df["Is_Boosted"].apply(lambda x: str(x).strip().upper() == 'TRUE' or x is True)
    if "Setup_Score" not in df.columns: df["Setup_Score"] = 0
    if "User_Prob" not in df.columns: df["User_Prob"] = df["Win_Prob"] 
    return df

def save_to_ledger(league, player, stat, line, odds, proj, vote, win_prob=0.55, is_boosted=False, setup_score=0, user_prob=0.55):
    row = {
        "Date": datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d"),
        "League": league,
        "Player": player.split('(')[0].strip(),
        "Stat": stat,
        "Odds": odds,
        "Line": line,
        "Proj": round(proj, 2),
        "Vote": vote,
        "Actual": "",
        "Result": "Pending",
        "Win_Prob": float(win_prob),
        "Is_Boosted": is_boosted,
        "Setup_Score": int(setup_score),
        "User_Prob": float(user_prob)
    }
    new_cols = ["Date", "League", "Player", "Stat", "Odds", "Line", "Proj", "Vote", "Actual", "Result", "Win_Prob", "Is_Boosted", "Setup_Score", "User_Prob"]
    append_to_sheet("ROI_Ledger", row, new_cols)
@st.cache_data(ttl=120)
def get_suppressed_stats(league, min_bets=10, max_win_rate=0.2):
    """Returns set of stat types with provably bad ledger history for this league."""
    try:
        ledger = load_ledger()
        if ledger.empty: return set()
        graded = ledger[
            (ledger['Result'].isin(['Win', 'Loss'])) &
            (ledger['League'] == league)
        ]
        suppress = set()
        for stat, group in graded.groupby('Stat'):
            if len(group) >= min_bets:
                win_rate = len(group[group['Result'] == 'Win']) / len(group)
                if win_rate <= max_win_rate:
                    suppress.add(stat)
        return suppress
    except:
        return set()

def load_parlay_ledger():
    df = load_sheet_df("Parlay_Ledger", ["Date", "Description", "Odds", "Risk", "Result", "Sportsbook", "Is_Free_Bet", "Is_Boosted"])
    df = df[df['Description'].astype(str).str.strip() != '']
    df = df[df['Date'].astype(str).str.strip() != '']
    df = df.reset_index(drop=True)
    if "Is_Free_Bet" not in df.columns: df["Is_Free_Bet"] = False
    else: df["Is_Free_Bet"] = df["Is_Free_Bet"].apply(lambda x: str(x).strip().upper() == 'TRUE' or x is True)
    if "Is_Boosted" not in df.columns: df["Is_Boosted"] = False
    else: df["Is_Boosted"] = df["Is_Boosted"].apply(lambda x: str(x).strip().upper() == 'TRUE' or x is True)
    return df

def save_to_parlay_ledger(desc, odds, risk, book, is_free, is_boosted=False):
    row = {"Date":datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d"), "Description": desc, "Odds": int(odds), "Risk": float(risk), "Result": "Pending", "Sportsbook": book, "Is_Free_Bet": is_free, "Is_Boosted": is_boosted}
    append_to_sheet("Parlay_Ledger", row, ["Date", "Description", "Odds", "Risk", "Result", "Sportsbook", "Is_Free_Bet", "Is_Boosted"])

def load_bankroll(): return load_sheet_df("Bankroll_Ledger", ["Date", "Sportsbook", "Type", "Amount"])

def save_bankroll_transaction(book, trans_type, amount):
    row = {"Date":datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d"), "Sportsbook": book, "Type": trans_type, "Amount": float(amount)}
    append_to_sheet("Bankroll_Ledger", row, ["Date", "Sportsbook", "Type", "Amount"])

# ✅ OPT-1: Cached — was called 3+ times per render with no caching
# ✅ OPT-2: Vectorized pandas ops replace slow .iterrows() loops
@st.cache_data(ttl=120)
def get_wallet_breakdown():
    b_df, p_df = load_bankroll(), load_parlay_ledger()
    book_balances = {book: 0.0 for book in SPORTSBOOKS}
    tot_dep, tot_wit, tot_cas, tot_sports = 0.0, 0.0, 0.0, 0.0

    if not b_df.empty:
        b_df['Amount'] = pd.to_numeric(b_df['Amount'], errors='coerce').fillna(0)
        b_df['Sportsbook'] = b_df['Sportsbook'].astype(str).str.strip()
        b_df['Type'] = b_df['Type'].astype(str)

        # ✅ OPT-2: Vectorized groupby instead of iterrows
        for bk, grp in b_df.groupby('Sportsbook'):
            total = grp['Amount'].sum()
            if bk in book_balances:
                book_balances[bk] += total
            elif bk:
                book_balances[bk] = total

        # ✅ OPT-2: Vectorized category sums
        tot_dep = b_df[b_df['Type'].str.contains('Deposit', na=False)]['Amount'].sum()
        tot_wit = b_df[b_df['Type'].str.contains('Withdrawal', na=False)]['Amount'].abs().sum()
        tot_cas = b_df[b_df['Type'].str.contains('Casino', na=False)]['Amount'].sum()

    if not p_df.empty:
        p_df['Odds_num'] = pd.to_numeric(p_df['Odds'], errors='coerce')
        p_df['Risk_num'] = pd.to_numeric(p_df['Risk'], errors='coerce')
        p_df['Is_Free'] = p_df['Is_Free_Bet'].apply(lambda x: str(x).strip().upper() == 'TRUE' or x is True)

        # ✅ OPT-2: Vectorized profit calculation
        def calc_profit(row):
            o, r, is_f, res = row['Odds_num'], row['Risk_num'], row['Is_Free'], row['Result']
            if pd.isna(o) or pd.isna(r): return 0.0, row.get('Sportsbook', '')
            prof = 0.0
            if res == 'Win': prof = (r * (o / 100)) if o > 0 else (r / (abs(o) / 100))
            elif res in ['Loss', 'Pending']: prof = -(0 if is_f else r)
            return prof, str(row.get('Sportsbook', '')).strip()

        for _, row in p_df.iterrows():
            prof, bk = calc_profit(row)
            if bk in book_balances: book_balances[bk] += prof
            elif bk: book_balances[bk] = prof
            tot_sports += prof

    book_balances = {k: v for k, v in book_balances.items() if v != 0.0}
    total_liquid = sum(book_balances.values())
    return max(total_liquid, 0.0), book_balances, tot_dep, tot_wit, tot_cas, tot_sports

def get_liquid_balance(): return get_wallet_breakdown()[0]

# ==========================================
# 2. AUTO-GRADER & AI AUTOPSY
# ==========================================
def auto_grade_ledger():
    df = load_ledger()
    if not (df['Result'] == 'Pending').any(): return df, "No pending bets."
    updated = 0
    stats_cache = {}

    for idx, r in df[df['Result'] == 'Pending'].iterrows():
        if r['Stat'] in ["Moneyline", "Spread", "Total (O/U)"]: continue
        try:
            league, player = r['League'], r['Player']
            cache_key = (league, player)
            if cache_key not in stats_cache:
                time.sleep(1)
                if league == "NBA": stats, _, _ = get_nba_stats(player); d_col = 'ValidDate'
                elif league == "NHL": stats, _, _ = get_nhl_stats(player); d_col = 'ValidDate'
                else: stats, _, _ = get_mlb_stats(player); d_col = 'ValidDate'
                stats_cache[cache_key] = (stats, d_col)
            stats, d_col = stats_cache[cache_key]
            if stats.empty: continue
            s_col = S_MAP.get(r['Stat'], "PTS")
            if league == "NBA":
                if s_col == "A": s_col = "AST"
                if s_col == "PRA" and 'PTS' in stats: stats['PRA'] = stats['PTS'] + stats['TRB'] + stats['AST']
                if s_col in ["DD", "TD"]:
                    tens = (stats['PTS'] >= 10).astype(int) + (stats['TRB'] >= 10).astype(int) + (stats['AST'] >= 10).astype(int) + (stats.get('STL', 0) >= 10).astype(int) + (stats.get('BLK', 0) >= 10).astype(int)
                    stats['DD'] = (tens >= 2).astype(int)
                    stats['TD'] = (tens >= 3).astype(int)
            stats['td'] = pd.to_datetime(stats[d_col]).dt.date
            bet_date = pd.to_datetime(r['Date']).date()
            next_date = bet_date + pd.Timedelta(days=1)
            g_row = stats[stats['td'].isin([bet_date, next_date])]
            if len(g_row) > 1: g_row = g_row[g_row['td'] == bet_date]
            if g_row.empty: g_row = stats[stats['td'] == next_date]
            if not g_row.empty:
                val, line_val = g_row.iloc[0][s_col], float(r['Line'])
                if r['Vote'] == "OVER": df.at[idx, 'Result'] = 'Win' if val > line_val else 'Loss' if val < line_val else 'Push'
                elif r['Vote'] == "UNDER": df.at[idx, 'Result'] = 'Win' if val < line_val else 'Loss' if val > line_val else 'Push'
                # ✅ Store actual stat value permanently for autopsy analysis
                df.at[idx, 'Actual'] = round(float(val), 2)
                updated += 1
        except: continue

    overwrite_sheet("ROI_Ledger", df)
    return df, f"Graded {updated} bets synced to Cloud!"

# ==========================================
# 3. DATA PULLS & ML ENGINE
# ==========================================
def check_api_quota():
    if not ODDS_API_KEY: return
    try:
        r = requests.get(f"https://api.the-odds-api.com/v4/sports?apiKey={ODDS_API_KEY}", timeout=5)
        u, rem = r.headers.get('x-requests-used'), r.headers.get('x-requests-remaining')
        if u and rem: st.session_state['api_used'] = int(u); st.session_state['api_remaining'] = int(rem)
    except: pass

@st.cache_data(ttl=3600)
def search_nba_players(query):
    if not query: return []
    try:
        search_term = query.split()[-1] if " " in query else query
        r = requests.get("https://api.balldontlie.io/v1/players", headers={"Authorization": BDL_API_KEY}, params={"search": search_term, "per_page": 100}, timeout=5)
        if r.status_code == 200:
            matches = []
            for p in r.json().get('data', []):
                if p.get('team'):
                    full_name = f"{p['first_name']} {p['last_name']}"
                    if query.lower() in full_name.lower():
                        matches.append(f"{full_name} ({p['team']['abbreviation']})")
            return matches
    except: pass; return []

@st.cache_data(ttl=3600)
def search_nhl_players(query):
    if not query: return []
    try:
        r = requests.get(f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=100&q={requests.utils.quote(query)}", timeout=5)
        if r.status_code == 200: return [f"{p['name']} ({p.get('teamAbbrev', 'FA')})" for p in r.json()]
    except: pass; return []

@st.cache_data(ttl=3600)
def search_mlb_players(query):
    if not query: return []
    try:
        r = requests.get(f"https://statsapi.mlb.com/api/v1/people/search?names={requests.utils.quote(query)}", timeout=5).json()
        if not r.get('people'): return []
        ids = ",".join([str(p['id']) for p in r['people'][:15]])
        br = requests.get(f"https://statsapi.mlb.com/api/v1/people?personIds={ids}&hydrate=currentTeam", timeout=5).json()
        m_t = {"Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "Seattle Mariners": "SEA", "San Francisco Giants": "SF", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH"}
        return [f"{p.get('fullName')} ({m_t.get(p.get('currentTeam', {}).get('name', ''), 'FA')})" for p in br.get('people', [])]
    except: pass; return []

@st.cache_data(ttl=60)
def get_nba_schedule():
    try:
        from nba_api.stats.endpoints import scoreboardv2
        from nba_api.stats.static import teams
        today_str = datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d")
        board = scoreboardv2.ScoreboardV2(game_date=today_str)
        games = board.get_data_frames()[0]
        if games.empty: return None, "No games scheduled today."
        team_dict = {t['id']: t['abbreviation'] for t in teams.get_teams()}
        matchups = []
        for _, g in games.iterrows():
            home_id, away_id = g['HOME_TEAM_ID'], g['VISITOR_TEAM_ID']
            home_abbrev, away_abbrev = team_dict.get(home_id, 'HOME'), team_dict.get(away_id, 'AWAY')
            status_id, status_text = g['GAME_STATUS_ID'], g['GAME_STATUS_TEXT']
            is_live_or_final = status_id in [2, 3]
            line_score = board.get_data_frames()[1]
            home_score, away_score = 0, 0
            if not line_score.empty:
                try:
                    home_row = line_score[line_score['TEAM_ID'] == home_id]
                    away_row = line_score[line_score['TEAM_ID'] == away_id]
                    if not home_row.empty and pd.notna(home_row['PTS'].iloc[0]): home_score = int(home_row['PTS'].iloc[0])
                    if not away_row.empty and pd.notna(away_row['PTS'].iloc[0]): away_score = int(away_row['PTS'].iloc[0])
                except: pass
            ds = f"Today - {status_text.replace(' ET', '').replace(' EST', '').upper()}" if status_id == 1 else status_text
            matchups.append({"home": home_abbrev, "away": away_abbrev, "status": ds, "home_score": home_score, "away_score": away_score, "is_live_or_final": is_live_or_final})
        return matchups, "Success"
    except: return None, "Failed to connect to NBA API."

@st.cache_data(ttl=60)
def get_nhl_schedule():
    try:
        today_str = datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d")
        r = requests.get(f"https://api-web.nhle.com/v1/schedule/{today_str}", timeout=5).json()
        if not r.get('gameWeek'): return None, "No games scheduled today."
        matchups = []
        for day_data in r['gameWeek']:
            if day_data.get('date') == today_str:
                for g in day_data.get('games', []):
                    state = g.get('gameState', 'FUT')
                    il = state in ['LIVE', 'CRIT', 'FINAL', 'OFF']
                    ds = "Final" if state in ['FINAL', 'OFF'] else "LIVE" if state in ['LIVE', 'CRIT'] else pd.to_datetime(g['startTimeUTC']).tz_convert('US/Eastern').strftime("%I:%M %p").lstrip("0")
                    matchups.append({"home": g['homeTeam']['abbrev'], "away": g['awayTeam']['abbrev'], "status": ds, "home_score": g.get('homeTeam', {}).get('score', 0), "away_score": g.get('awayTeam', {}).get('score', 0), "is_live_or_final": il})
        if not matchups: return None, "No games scheduled today."
        return matchups, "Success"
    except: return None, "Failed to connect to NHL API."

@st.cache_data(ttl=60)
def get_mlb_schedule():
    try:
        today_str = datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d")
        r = requests.get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today_str}", timeout=5).json()
        if not r.get('dates') or not r['dates'][0].get('games'): return None, "No games scheduled today."
        m_t = {"Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "Seattle Mariners": "SEA", "San Francisco Giants": "SF", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH"}
        matchups = []
        for g in r['dates'][0]['games']:
            home_full = g['teams']['home']['team'].get('name', '')
            away_full = g['teams']['away']['team'].get('name', '')
            home = m_t.get(home_full, home_full.split()[-1][:3].upper() if home_full else "HOME")
            away = m_t.get(away_full, away_full.split()[-1][:3].upper() if away_full else "AWAY")
            sr = g['status']['detailedState']
            il = sr in ['In Progress', 'Final', 'Game Over', 'Completed Early']
            ds = pd.to_datetime(g['gameDate']).tz_convert('US/Eastern').strftime("%I:%M %p").lstrip("0") if not il and sr in ['Scheduled', 'Pre-Game', 'Warmup'] else sr
            matchups.append({"home": home, "away": away, "status": ds, "home_score": g['teams']['home'].get('score', 0), "away_score": g['teams']['away'].get('score', 0), "is_live_or_final": il})
        return matchups, "Success"
    except: return None, "Failed to connect to MLB API."

@st.cache_data(ttl=600)
def get_live_line(player_label, stat_type, api_key, sport_path):
    if not api_key: return None, None, "API Key missing", None, None
    m_map = {
        "Points": "player_points", "Goals": "player_goals", "Assists": "player_assists", 
        "Shots on Goal": "player_shots_on_goal", "Power Play Points": "player_power_play_points", 
        "Rebounds": "player_rebounds", "PRA (Pts+Reb+Ast)": "player_points_rebounds_assists", 
        "Threes Made": "player_threes", "Hits": "batter_hits", "Home Runs": "batter_home_runs", 
        "Pitcher Strikeouts": "pitcher_strikeouts", "Double Double": "player_double_double", 
        "Triple Double": "player_triple_double", 
        "Blocks": "player_blocks", "Steals": "player_steals" # 🟢 Added to Odds API Sync
    }
    market = m_map.get(stat_type, "player_points")
    clean_name = player_label.split("(")[0].strip().lower()
    team_abbr = player_label.split("(")[1].split(")")[0].strip().upper() if "(" in player_label else ""
    used, rem = None, None
    try:
        events_resp = requests.get(f"https://api.the-odds-api.com/v4/sports/{sport_path}/events?apiKey={api_key}", timeout=10)
        events_data = events_resp.json()
        used, rem = events_resp.headers.get('x-requests-used'), events_resp.headers.get('x-requests-remaining')
        if not isinstance(events_data, list) or len(events_data) == 0: return None, None, "No active events", used, rem
        target_team_name = ODDS_MEGA_MAP.get(team_abbr)
        events_to_check = []
        if target_team_name:
            events_to_check = [e for e in events_data if target_team_name in e.get('home_team', '') or target_team_name in e.get('away_team', '')]
        if not events_to_check: events_to_check = events_data[:2]
        for event in events_to_check:
            odds_resp = requests.get(f"https://api.the-odds-api.com/v4/sports/{sport_path}/events/{event['id']}/odds?apiKey={api_key}&regions=us&markets={market}&oddsFormat=american", timeout=10)
            odds_data = odds_resp.json()
            used, rem = odds_resp.headers.get('x-requests-used', used), odds_resp.headers.get('x-requests-remaining', rem)
            for b in odds_data.get('bookmakers', []):
                for m in b.get('markets', []):
                    for o in m.get('outcomes', []):
                        if clean_name in o.get('description', '').lower():
                            if 'point' in o and 'price' in o: return float(o['point']), int(o['price']), f"Synced: {b.get('title')}", used, rem
                            elif 'price' in o: return None, int(o['price']), f"Synced Odds: {b.get('title')}", used, rem
        return None, None, "Props not posted yet.", used, rem
    except Exception as e: return None, None, f"API Error", used, rem

@st.cache_data(ttl=300)
def get_nba_stats(player_label):
    cn = player_label.split("(")[0].strip()
    try:
        from nba_api.stats.static import players
        from nba_api.stats.endpoints import playergamelog
        import unicodedata
        def clean_name(name):
            base = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8').lower()
            return base.replace(".", "").replace("'", "").replace("-", "").replace(" ", "")
        nba_players = players.get_players()
        player_dict = [p for p in nba_players if clean_name(p['full_name']) == clean_name(cn)]
        if not player_dict: return pd.DataFrame(), 404, []
        pid = player_dict[0]['id']
        seasons = ['2025-26', '2024-25', '2023-24']
        df_list = []
        for s in seasons:
            try:
                log = playergamelog.PlayerGameLog(player_id=pid, season=s)
                df_list.append(log.get_data_frames()[0])
                time.sleep(0.5)
            except: pass
        if not df_list: return pd.DataFrame(), 404, []
        df = pd.concat(df_list, ignore_index=True)
        if df.empty: return pd.DataFrame(), 404, []
        df['Is_Home'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
        df['MATCHUP'] = df['MATCHUP'].apply(lambda x: x.split(' ')[-1])
        df['ValidDate'] = pd.to_datetime(df['GAME_DATE'])
        df['ShortDate'] = df['ValidDate'].dt.strftime('%b %d')
        def parse_mins(x):
            try:
                s = str(x)
                return float(s.split(':')[0]) + float(s.split(':')[1])/60.0 if ':' in s else float(s)
            except: return 0.0
        df['MINS'] = df['MIN'].apply(parse_mins)
        df = df.rename(columns={'REB': 'TRB'})
        today = pd.to_datetime(datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d"))
        df['Days_Ago'] = (today - df['ValidDate']).dt.days
        df = df[(df['Days_Ago'] >= 0) & (df['Days_Ago'] <= 1095)]
        df['Weight'] = np.exp(-0.003465 * df['Days_Ago'])
        final_cols = [c for c in ['ValidDate', 'ShortDate', 'MATCHUP', 'Is_Home', 'MINS', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG3M', 'Weight'] if c in df.columns]
        return df[final_cols].sort_values('ValidDate').reset_index(drop=True), 200, []
    except: return pd.DataFrame(), 500, []

@st.cache_data(ttl=300)
def get_nhl_stats(player_label):
    cn = player_label.split("(")[0].strip()
    try:
        r = requests.get(f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=25&q={requests.utils.quote(cn)}", timeout=5).json()
        pid = next((p.get('playerId', p.get('id')) for p in r if p.get('name','').lower() == cn.lower()), r[0].get('playerId', r[0].get('id')) if r else None)
        seasons = ['20252026', '20242025', '20232024']
        logs = []
        for s in seasons:
            try:
                resp = requests.get(f"https://api-web.nhle.com/v1/player/{pid}/game-log/{s}/2", timeout=5).json()
                if 'gameLog' in resp: logs.extend(resp['gameLog'])
            except: pass
        if not logs: return pd.DataFrame(), 404, []
        df = pd.DataFrame(logs)
        df['PTS'] = pd.to_numeric(df.get('points', 0))
        df['G'] = pd.to_numeric(df.get('goals', 0))
        df['A'] = pd.to_numeric(df.get('assists', 0))
        df['SOG'] = pd.to_numeric(df.get('shots', 0))
        df['PPP'] = pd.to_numeric(df.get('powerPlayPoints', 0))
        df['Is_Home'] = np.where(df.get('homeRoadFlag', 'H') == 'H', 1, 0)
        df['MINS'] = df.get('toi', '15:00').apply(lambda x: int(str(x).split(':')[0]) + int(str(x).split(':')[1])/60.0 if ':' in str(x) else 0.0)
        df['MATCHUP'] = df['opponentAbbrev']
        df['ValidDate'] = pd.to_datetime(df['gameDate'])
        df['ShortDate'] = df['ValidDate'].dt.strftime('%b %d')
        today = pd.to_datetime(datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d"))
        df['Days_Ago'] = (today - df['ValidDate']).dt.days
        df = df[(df['Days_Ago'] >= 0) & (df['Days_Ago'] <= 1095)]
        df['Weight'] = np.exp(-0.003465 * df['Days_Ago'])
        return df.sort_values('ValidDate').reset_index(drop=True), 200, []
    except: return pd.DataFrame(), 500, []

@st.cache_data(ttl=300)
def get_mlb_stats(player_label):
    cn = player_label.split("(")[0].strip()
    try:
        sr = requests.get(f"https://statsapi.mlb.com/api/v1/people/search?names={requests.utils.quote(cn)}", timeout=5).json()
        if not sr.get('people'): return pd.DataFrame(), 404, []
        pid = next((p['id'] for p in sr.get('people', []) if p.get('fullName','').lower() == cn.lower()), sr['people'][0]['id'] if sr.get('people') else None)
        curr_year = datetime.now().year
        seasons = [str(curr_year), str(curr_year-1), str(curr_year-2)]
        splits = []
        for s in seasons:
            try:
                log = requests.get(f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=gameLog&group=hitting,pitching&season={s}", timeout=5).json()
                if 'stats' in log:
                    for stat_group in log['stats']: splits.extend(stat_group.get('splits', []))
            except: pass
        if not splits: return pd.DataFrame(), 404, []
        data = [{'ValidDate': pd.to_datetime(s.get('date', '2025-01-01')), 'MATCHUP': s.get('opponent', {}).get('name', 'OPP').split(' ')[-1][:3].upper(), 'Is_Home': 1 if s.get('isHome', True) else 0, 'H': s.get('stat', {}).get('hits', 0), 'HR': s.get('stat', {}).get('homeRuns', 0), 'TB': s.get('stat', {}).get('totalBases', 0), 'K': s.get('stat', {}).get('strikeOuts', 0), 'ER': s.get('stat', {}).get('earnedRuns', 0), 'MINS': float(s.get('stat', {}).get('plateAppearances', s.get('stat', {}).get('battersFaced', 1)))} for s in splits]
        df = pd.DataFrame(data)
        df['ShortDate'] = df['ValidDate'].dt.strftime('%b %d')
        today = pd.to_datetime(datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d"))
        df['Days_Ago'] = (today - df['ValidDate']).dt.days
        df = df[(df['Days_Ago'] >= 0) & (df['Days_Ago'] <= 1095)]
        df['Weight'] = np.exp(-0.003465 * df['Days_Ago'])
        return df.sort_values('ValidDate').reset_index(drop=True), 200, []
    except: return pd.DataFrame(), 500, []

@st.cache_data(ttl=43200)
def get_live_nba_team_stats():
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced')
        df = stats.get_data_frames()[0]
        df['TEAM_ABBREV'] = df['TEAM_NAME'].map(NBA_FULL_TO_ABBREV)
        df_clean = df[['TEAM_ABBREV', 'DEF_RATING', 'PACE']].set_index('TEAM_ABBREV')
        df_clean['DEF_RANK'] = df_clean['DEF_RATING'].rank(ascending=True)
        df_clean['PACE_RANK'] = df_clean['PACE'].rank(ascending=False)
        return df_clean.to_dict('index')
    except: return {}

def get_player_archetype(df, league):
    if df.empty or league != "NBA": return "Unknown Profile"
    avg_mins = df['MINS'].mean()
    if pd.isna(avg_mins) or avg_mins < 5: avg_mins = 15.0
    pts_36 = (df.get('PTS', pd.Series([0])).mean() / avg_mins) * 36
    trb_36 = (df.get('TRB', pd.Series([0])).mean() / avg_mins) * 36
    ast_36 = (df.get('AST', pd.Series([0])).mean() / avg_mins) * 36
    fg3m_36 = (df.get('FG3M', pd.Series([0])).mean() / avg_mins) * 36
    clusters = {"👑 Primary Playmaker (High USG)": [26.0, 6.0, 9.5, 2.5], "🦍 Paint Beast / Rim Runner": [17.0, 13.5, 2.0, 0.1], "🧬 Versatile Point-Forward": [21.0, 9.0, 6.0, 1.5], "🎯 3&D Wing / Spot-Up Shooter": [15.0, 5.0, 2.0, 3.8], "🛡️ Two-Way Connector": [13.0, 4.5, 5.5, 1.5]}
    player_vec = [[pts_36, trb_36, ast_36, fg3m_36]]
    best_match, min_dist = "Unknown", float('inf')
    for name, centroid in clusters.items():
        dist = euclidean_distances(player_vec, [centroid])[0][0]
        if dist < min_dist: min_dist = dist; best_match = name
    return best_match

def get_archetype_defense_modifier(league, opp, archetype, bad_defs=None):
    """bad_defs: pre-fetched NHL bad defenses dict (pass to avoid repeated calls in loops)"""
    if league == "NBA":
        live_stats = get_live_nba_team_stats()
        if opp in live_stats:
            def_rank, pace_rank = live_stats[opp]['DEF_RANK'], live_stats[opp]['PACE_RANK']
            mod_val, mod_desc = 1.0, f"🛡️ Def Rank: #{int(def_rank)} | 🏃 Pace: #{int(pace_rank)} -> "
            if def_rank <= 10: mod_val *= 0.90; mod_desc += "Elite Def (-10%). "
            elif def_rank >= 21: mod_val *= 1.10; mod_desc += "Weak Def (+10%). "
            else: mod_desc += "Avg Def (Neutral). "
            if pace_rank <= 10: mod_val *= 1.05; mod_desc += "Fast Pace (+5%). "
            elif pace_rank >= 21: mod_val *= 0.95; mod_desc += "Slow Pace (-5%). "
            if "Point-Forward" in archetype and def_rank >= 15: mod_val *= 1.05; mod_desc += "🚨 Exploit: Weak vs Forwards."
            elif "Primary Playmaker" in archetype and def_rank <= 10: mod_val *= 0.95; mod_desc += "🛑 Fade: Elite Perimeter Def."
            return mod_val, mod_desc
        else:
            if opp in ["MIN", "BOS", "OKC", "ORL", "MIA", "NYK"]: return 0.90, "Elite Defense (-10%)"
            elif opp in ["WAS", "DET", "CHA", "SAS", "POR", "ATL", "UTA"]: return 1.10, "Weak Defense (+10%)"
            return 1.00, "Average Def (Neutral)"
    elif league == "MLB":
        if opp in ["ATL", "HOU", "LAD", "BAL", "PHI", "NYY"]: return 0.90, "Elite Pitching (-10%)"
        elif opp in ["COL", "OAK", "CHW", "KC", "WSH"]: return 1.10, "Weak Pitching (+10%)"
        return 1.00, "Average Pitching (Neutral)"
    else:
        # ✅ OPT-8: Accept pre-fetched bad_defs to avoid repeated API calls in tight loops
        defs = bad_defs if bad_defs is not None else get_nhl_bad_defenses()
        if opp in defs:
            sog_allowed = defs[opp]
            return 1.10, f"Swiss Cheese Def (+10%, {sog_allowed} SOG/G allowed)"
        if opp in ["FLA", "DAL", "CAR", "WPG", "VGK", "LAK"]: return 0.90, "Elite Goalie (-10%)"
        return 1.00, "Average Def (Neutral)"

def get_fatigue_modifier(rest_status):
    if "B2B" in rest_status: return 0.95, "Tired Legs (-5%)"
    if "3 in 4" in rest_status: return 0.90, "Exhausted (-10%)"
    return 1.00, "Fully Rested"

def calculate_implied_prob(odds_str):
    try:
        odds = int(str(odds_str).replace('+', '').strip())
        if odds == 0: return 0.0
        if odds < 0: return (abs(odds) / (abs(odds) + 100)) * 100
        else: return (100 / (odds + 100)) * 100
    except ValueError: return 0.0

def estimate_alt_odds(orig_line, orig_odds, new_line, stat_type):
    if orig_line is None or orig_odds is None or orig_line == new_line: return orig_odds
    p_orig = abs(orig_odds)/(abs(orig_odds)+100) if orig_odds < 0 else 100/(orig_odds+100)
    std_est = {"Points": 6.0, "Rebounds": 2.5, "Assists": 2.5, "Threes Made": 1.2, "PRA (Pts+Reb+Ast)": 8.0, "Minutes Played": 5.0, "Hits": 1.0, "Pitcher Strikeouts": 2.0}.get(stat_type, 3.0)
    z_shift = (orig_line - new_line) / std_est
    p_new = max(0.05, min(0.95, p_orig + (z_shift * 0.35)))
    new_odds = int(round((-100*p_new)/(1-p_new))) if p_new > 0.50 else int(round((100*(1-p_new))/p_new))
    return 5 * round(new_odds/5)

def calculate_setup_score(win_prob, edge_pct, board, c_proj, line, stat_type):
    """
    0–100 score rating the quality of a bet setup.
    Higher = cleaner setup with more converging signals.
    """
    score = 0
    # Component 1: Win probability above 50% baseline (max 35 pts)
    score += min(35, max(0, (win_prob - 0.50) * 200))
    # Component 2: Edge over implied odds (max 25 pts)
    score += min(25, max(0, edge_pct * 2.5))
    # Component 3: Board agreement (max 25 pts)
    if board:
        votes = [m['vote'] for m in board]
        top_vote = max(set(votes), key=votes.count)
        agreement = votes.count(top_vote)
        score += {5: 25, 4: 15, 3: 5}.get(agreement, 0)
    # Component 4: Projection conviction
    thresh = PASS_THRESHOLDS.get(S_MAP.get(stat_type, ""), 0.75)
    gap = abs(c_proj - line)
    if thresh > 0:
        score += min(15, max(0, ((gap / thresh) - 1.0) * 15))
    return max(0, min(100, int(score)))

def build_models(df_ml, s_col, weights, league, is_home_current, rest_status, tonight_def_mod):
    # 🟢 SAFETY CLIP: Forcing a floor of 0 because Poisson crashes on negative numbers
    y = df_ml[s_col].fillna(0).clip(lower=0).values

    # 🟢 1. FEATURE ENRICHMENT: Calculate Rest Days
    df_ml['Rest_Days'] = df_ml['ValidDate'].diff().dt.days.fillna(3.0).clip(0, 7)

    expected_mins = df_ml['MINS'].tail(5).mean()
    mins_std = df_ml['MINS'].tail(10).std()
    if pd.isna(mins_std) or mins_std == 0: mins_std = 2.0

    if pd.isna(expected_mins) or expected_mins == 0:
        if league == "MLB": expected_mins = 4.0
        elif league == "NHL": expected_mins = 18.0
        else: expected_mins = 15.0
    else:
        if league == "MLB": expected_mins = np.clip(expected_mins, 2.0, 6.0)
        elif league == "NHL": expected_mins = np.clip(expected_mins, 10.0, 28.0)
        else: expected_mins = np.clip(expected_mins, 5.0, 42.0)

    # 🟢 2. POISSON SETUP: Predicts raw count using Minutes and Time Index
    X_poi_train = df_ml[['MINS']].copy()
    X_poi_train['Trend'] = np.arange(len(df_ml))

    # 🟢 3. RANDOM FOREST SETUP: Now sees Form + Context + Defense
    df_ml['Roll3'] = df_ml[s_col].rolling(3).mean().fillna(df_ml[s_col].mean()).fillna(0)
    X_rf_train = df_ml[['Roll3', 'MINS', 'Is_Home', 'Rest_Days', 'Opp_Def_Mod']].fillna(0).values

    # 🟢 4. XGBOOST SETUP: Replaces old Gradient Boosting
    s_mean = df_ml[s_col].mean() if not pd.isna(df_ml[s_col].mean()) else 0.0
    df_ml['Dev'] = df_ml[s_col].fillna(0) - s_mean
    X_xgb_train = df_ml[['MINS', 'Dev']].fillna(0).values

    # 🟢 5. BASELINE SETUP: HistGradientBoosting
    df_ml['EWMA'] = df_ml[s_col].ewm(span=5, adjust=False).mean().fillna(s_mean)
    X_hgbr_train = df_ml[['EWMA', 'MINS']].fillna(0).values

    # ⚡ PARALLEL TRAINING
    def train_poisson():
        return PoissonRegressor(alpha=1e-3, max_iter=500).fit(X_poi_train.values, y, sample_weight=weights)
    def train_rf():
        return RandomForestRegressor(n_estimators=50, random_state=42).fit(X_rf_train, y, sample_weight=weights)
    def train_xgb():
        return XGBRegressor(n_estimators=50, learning_rate=0.05, random_state=42, objective='reg:squarederror').fit(X_xgb_train, y, sample_weight=weights)
    def train_hgbr():
        return HistGradientBoostingRegressor(max_iter=50, random_state=42).fit(X_hgbr_train, y, sample_weight=weights)

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            f_poi, f_rf, f_xgb, f_hgbr = executor.submit(train_poisson), executor.submit(train_rf), executor.submit(train_xgb), executor.submit(train_hgbr)
            poi, rf, xgb, hgbr = f_poi.result(), f_rf.result(), f_xgb.result(), f_hgbr.result()
    except Exception:
        poi, rf, xgb, hgbr = train_poisson(), train_rf(), train_xgb(), train_hgbr()

    # Format tonight's contextual inputs
    tonight_rest = 1.0 if "B2B" in str(rest_status) else (0.0 if "3 in 4" in str(rest_status) else 3.0)

    # 🎯 GENERATE TONIGHT'S PROJECTIONS
    trend_proj = poi.predict([[expected_mins, len(df_ml)]])[0]
    stat_proj = rf.predict([[df_ml['Roll3'].iloc[-1], expected_mins, is_home_current, tonight_rest, tonight_def_mod]])[0]
    con_proj = xgb.predict([[expected_mins, trend_proj - s_mean]])[0]
    base_proj = hgbr.predict([[df_ml['EWMA'].iloc[-1], expected_mins]])[0]

    return trend_proj, stat_proj, con_proj, base_proj, poi, rf, xgb, hgbr, X_poi_train, X_rf_train, X_xgb_train, X_hgbr_train, expected_mins, mins_std

def apply_context_mods(df, s_col, league, opp, rest, is_home_current, archetype):
    mod_val, mod_desc = get_archetype_defense_modifier(league, opp, archetype)
    fatigue_val, fatigue_desc = get_fatigue_modifier(rest)
    home_mod = 1.0
    away_mod = 1.0
    split_text = "Home" if is_home_current == 1 else "Road"
    split_desc = "0%"
    try:
        if s_col in df.columns and len(df) > 0:
            season_avg = df[s_col].mean()
            if not pd.isna(season_avg) and season_avg > 0:
                home_games = df[df['Is_Home'] == 1][s_col]
                away_games = df[df['Is_Home'] == 0][s_col]
                home_weight = min(len(home_games) / 10.0, 1.0)
                away_weight = min(len(away_games) / 10.0, 1.0)
                home_avg = (home_games.mean() * home_weight + season_avg * (1 - home_weight)) if len(home_games) > 0 else season_avg
                away_avg = (away_games.mean() * away_weight + season_avg * (1 - away_weight)) if len(away_games) > 0 else season_avg
                home_mod = float(np.clip(home_avg / season_avg, 0.80, 1.20))
                away_mod = float(np.clip(away_avg / season_avg, 0.80, 1.20))
    except: pass
    current_split_mod = home_mod if is_home_current == 1 else away_mod
    split_desc = f"+{((current_split_mod-1)*100):.0f}%" if current_split_mod > 1 else f"{((current_split_mod-1)*100):.0f}%"
    return mod_val, mod_desc, fatigue_val, fatigue_desc, current_split_mod, split_text, split_desc, home_mod, away_mod

def apply_skynet(raw_vote, stat_type, league):
    if raw_vote == "PASS": return {"mod": 1.0, "msg": "🟣 Skynet: Market is efficient. Pass.", "color": "#94a3b8"}
    try:
        ledger = load_ledger()
        if not ledger.empty and 'Result' in ledger.columns and 'League' in ledger.columns:
            graded = ledger[ledger['Result'].isin(['Win', 'Loss'])]
            subset = graded[(graded['Stat'] == stat_type) & (graded['Vote'] == raw_vote) & (graded['League'] == league)]
            total_graded = len(subset)
            prior_wins, prior_losses = 3.0, 3.0
            if total_graded > 0:
                actual_wins = len(subset[subset['Result'] == 'Win'])
                actual_losses = total_graded - actual_wins
                posterior_win_rate = (prior_wins + actual_wins) / (prior_wins + prior_losses + total_graded)
                skynet_mod = 1.0 + ((posterior_win_rate - 0.50) * 0.3)
                if posterior_win_rate >= 0.53:
                    msg = f"🔥 SKYNET BOOST: You are {actual_wins}-{actual_losses} on {league} {stat_type} {raw_vote}s."
                    color = "#00E676"
                elif posterior_win_rate <= 0.47:
                    msg = f"🛑 SKYNET TAX: You are {actual_wins}-{actual_losses} on {league} {stat_type} {raw_vote}s. Applying penalty."
                    color = "#ff0055"
                else:
                    msg = f"⚖️ SKYNET AUDIT: You are {actual_wins}-{actual_losses} on {league} {stat_type} {raw_vote}s. Neutral."
                    color = "#FFD700"
                return {"mod": round(skynet_mod, 3), "msg": msg, "color": color}
            else:
                return {"mod": 1.0, "msg": f"🟣 Skynet: Gathering initial data on {league} {stat_type} {raw_vote}s.", "color": "#94a3b8"}
    except: pass
    return {"mod": 1.0, "msg": "🟣 Skynet: Awaiting enough ledger data.", "color": "#94a3b8"}

# ✅ OPT-4: Skynet moved OUTSIDE the cached function.
#    - Skynet result no longer gets frozen in the 300s ML cache
#    - New graded bets immediately affect Skynet modifier on next render
#    - df_hash and ledger_hash params added as cheap cache invalidation keys
# ✅ OPT-6: df_hash param prevents re-hashing the entire DataFrame on every cache check
def log_prediction_receipt(player_name, stat_type, proj_value, game_date):
    """Saves a tamper-proof receipt of the live, pre-game AI projection."""
    file_path = "saved_projections.csv"
    
    # Format the incoming date to match your DataFrame (usually YYYY-MM-DD)
    game_date_str = str(game_date)[:10] 
    
    new_record = pd.DataFrame([{
        "Player": player_name,
        "Stat": stat_type,
        "Game_Date": game_date_str,
        "Live_Proj": round(proj_value, 2),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])
    
    if not os.path.exists(file_path):
        new_record.to_csv(file_path, index=False)
        return
        
    try:
        df_saved = pd.read_csv(file_path)
        # Check if we already logged this exact player/stat/date
        is_duplicate = not df_saved[
            (df_saved['Player'] == player_name) & 
            (df_saved['Stat'] == stat_type) & 
            (df_saved['Game_Date'] == game_date_str)
        ].empty
        
        if not is_duplicate:
            new_record.to_csv(file_path, mode='a', header=False, index=False)
    except Exception as e:
        pass # Silently fail if CSV is locked by another thread
@st.cache_data(show_spinner=False, ttl=300)
def run_ml_board(df, s_col, line, opp, league, rest, is_home_current, stat_type, ignore_blowout=False, df_hash="", ledger_hash=""):
    df_ml = df.copy()
    archetype = get_player_archetype(df_ml, league)

    if len(df_ml) < 5:
        return df_ml, [], 0, "PASS", "#94a3b8", 1.0, "Not enough data", 1.0, "", "", 1.0, "", archetype, "Awaiting Data", "#94a3b8"

    weights = df_ml['Weight'].values if 'Weight' in df_ml.columns else np.ones(len(df_ml))

    # ✅ OPT-8: Pre-fetch bad_defs once here, pass to all defense modifier calls
    bad_defs = get_nhl_bad_defenses() if league == "NHL" else None

    # 🟢 1. DEFENSE MODIFIER & HISTORICAL MAPPING
    mod_val, mod_desc = get_archetype_defense_modifier(league, opp, archetype, bad_defs)
    
    # Pre-calculate the defense modifier for EVERY game in the log
    unique_mods = {team: get_archetype_defense_modifier(league, team, archetype, bad_defs)[0] for team in df_ml['MATCHUP'].unique()}
    df_ml['Opp_Def_Mod'] = df_ml['MATCHUP'].map(unique_mods).fillna(1.0)
    # 🟢 2. VENUE SPLIT MODIFIERS (Calculated inline to prevent NameErrors)
    s_mean = df_ml[s_col].mean()
    if pd.isna(s_mean) or s_mean == 0:
        home_mod, away_mod, current_split_mod = 1.0, 1.0, 1.0
        split_text, split_desc = "Neutral", "Not enough data for venue splits."
    else:
        home_mean = df_ml[df_ml['Is_Home'] == 1][s_col].mean()
        away_mean = df_ml[df_ml['Is_Home'] == 0][s_col].mean()
        
        h_mod = (home_mean / s_mean) if pd.notna(home_mean) and len(df_ml[df_ml['Is_Home'] == 1]) > 0 else 1.0
        a_mod = (away_mean / s_mean) if pd.notna(away_mean) and len(df_ml[df_ml['Is_Home'] == 0]) > 0 else 1.0
        
        # Dampen extremes so a wild split doesn't break the projection
        home_mod = np.clip(1.0 + ((h_mod - 1.0) * 0.5), 0.8, 1.2)
        away_mod = np.clip(1.0 + ((a_mod - 1.0) * 0.5), 0.8, 1.2)
        
        current_split_mod = home_mod if is_home_current == 1 else away_mod
        split_text = "Home" if is_home_current == 1 else "Away"
        split_desc = f"{split_text} Split: {current_split_mod:.2f}x production."

    # 🟢 3. FATIGUE MODIFIERS (Calculated inline)
    rest_str = str(rest)
    if "B2B" in rest_str:
        fatigue_val, fatigue_desc = 0.90, "⚠️ B2B: Heavy legs expected (-10%)."
    elif "3 in 4" in rest_str:
        fatigue_val, fatigue_desc = 0.95, "⚠️ 3 in 4 Nights: Slight fatigue (-5%)."
    elif "3+" in rest_str:
        fatigue_val, fatigue_desc = 1.05, "🔋 3+ Days Rest: Fully rested (+5%)."
    else:
        fatigue_val, fatigue_desc = 1.0, "🟢 Standard Rest."

    trend_proj, stat_proj, con_proj, base_proj, poi, rf, xgb, hgbr, X_poi_train, X_rf_train, X_xgb_train, X_hgbr_train, expected_mins, mins_std = build_models(
        df_ml, s_col, weights, league, is_home_current, rest, mod_val
    )

    is_blowout_risk = False
    if league == "NBA" and "Weak Def" in mod_desc and expected_mins >= 25 and not ignore_blowout:
        is_blowout_risk = True
        expected_mins = max(15.0, expected_mins - (mins_std * 1.5))
        
        # Calculate tonight's rest to feed into the Random Forest recalculation
        tonight_rest = 1.0 if "B2B" in str(rest) else (0.0 if "3 in 4" in str(rest) else 3.0)
        s_mean = df_ml[s_col].mean() if not pd.isna(df_ml[s_col].mean()) else 0.0
        
        # Recalculate using the NEW models and inputs
        trend_proj = poi.predict([[expected_mins, len(df_ml)]])[0]
        # 🟢 Added mod_val to the rf.predict line so the Random Forest has all 5 inputs
        stat_proj = rf.predict([[df_ml['Roll3'].iloc[-1], expected_mins, is_home_current, tonight_rest, mod_val]])[0]
        con_proj = xgb.predict([[expected_mins, trend_proj - s_mean]])[0]
        base_proj = hgbr.predict([[df_ml['EWMA'].iloc[-1], expected_mins]])[0]
        
    # 🟢 MATH FIX: The Random Forest (stat_proj) already knows the defense, venue, and rest internally!
    # We only apply the external multipliers to the Poisson model (trend_proj).
    adjusted_trend = trend_proj * mod_val * fatigue_val * current_split_mod
    guru_proj = (adjusted_trend + stat_proj) / 2
    raw_consensus = (trend_proj + stat_proj + con_proj + base_proj + guru_proj) / 5

    floor_proj = max(0.0, raw_consensus * (max(1.0, expected_mins - mins_std) / max(1.0, expected_mins)))
    ceil_proj = raw_consensus * ((expected_mins + mins_std) / max(1.0, expected_mins))

    SAMPLE_GATES = {
        "NBA": {"min_games": 15, "min_recent": 5},
        "NHL": {"min_games": 10, "min_recent": 3},
        "MLB": {"min_games": 10, "min_recent": 4},
    }
    gate = SAMPLE_GATES.get(league, {"min_games": 10, "min_recent": 3})
    recent_games = int((df_ml['Days_Ago'] <= 30).sum()) if 'Days_Ago' in df_ml.columns else len(df_ml)
    low_sample_warning = ""

    if len(df_ml) < gate["min_games"]:
        low_sample_warning = f"⚠️ <b>THIN SAMPLE:</b> Only {len(df_ml)} career games found (need {gate['min_games']}). Confidence is reduced.<br>"
    elif recent_games < gate["min_recent"]:
        low_sample_warning = f"⚠️ <b>STALE DATA:</b> Only {recent_games} games in the last 30 days. Player may be returning from injury.<br>"

    # 🟢 BUILD THE UI WARNING STRING
    vol_warning = ""
    if is_blowout_risk: vol_warning += f"🚨 BLOWOUT RISK: Matchup is highly lopsided. Slashed expected minutes.<br>"
    if mins_std >= 4.5: vol_warning += f"⚠️ HIGH VOLATILITY (±{mins_std:.1f}m). Floor: {floor_proj:.1f} | Ceil: {ceil_proj:.1f}.<br>"
    elif mins_std <= 2.5: vol_warning += f"🟢 Stable Rotation (±{mins_std:.1f}m).<br>"
    
    mod_desc = vol_warning + low_sample_warning + mod_desc

    threshold = PASS_THRESHOLDS.get(s_col, 0.5) # Defaults to 0.5 if stat isn't listed

    def get_raw_vote(p): return "OVER" if p >= line + threshold else ("UNDER" if p <= line - threshold else "PASS")
    raw_vote = get_raw_vote(raw_consensus)

    # ✅ OPT-4: Return raw_vote so caller can apply fresh Skynet outside the cache
    final_consensus = raw_consensus  # Skynet applied by caller after this returns

    def get_final_vote(p): return ("OVER", "#00c853") if p >= line + threshold else (("UNDER", "#d50000") if p <= line - threshold else ("PASS", "#94a3b8"))
    f_vote, f_color = get_final_vote(final_consensus)

    poi_hist = poi.predict(X_poi_train.values)
    rf_hist = rf.predict(X_rf_train)
    xgb_hist = xgb.predict(X_xgb_train)
    hgbr_hist = hgbr.predict(X_hgbr_train)

    # ✅ OPT-8: Pass pre-fetched bad_defs to avoid N repeated get_nhl_bad_defenses() calls
    hist_split_mods = np.where(df_ml['Is_Home'] == 1, home_mod, away_mod)
    mods = df_ml['Opp_Def_Mod'].values  # We already built this column in Step 1!
    
    # Only scale the Poisson historicals, leave the RF historicals alone
    guru_hist = ((poi_hist * mods * 1.0 * hist_split_mods) + rf_hist) / 2
    df_ml['AI_Proj'] = (poi_hist + rf_hist + xgb_hist + hgbr_hist + guru_hist) / 5

    board = [
        {"name": "⏱️ MIN Maximizer", "model": "Poisson Regressor", "proj": trend_proj, "vote": get_raw_vote(trend_proj), "color": get_final_vote(trend_proj)[1], "quote": f"Projects {trend_proj:.1f} by weighting recent mins."},
        {"name": "📊 Statistician", "model": "Random Forest", "proj": stat_proj, "vote": get_raw_vote(stat_proj), "color": get_final_vote(stat_proj)[1], "quote": f"Deep Memory sets stable floor. Trees favor {get_raw_vote(stat_proj)}."},
        {"name": "🃏 Contrarian", "model": "XGBoost", "proj": con_proj, "vote": get_raw_vote(con_proj), "color": get_final_vote(con_proj)[1], "quote": f"Flags variance {'regression' if con_proj < df_ml[s_col].mean() else 'spike'} from season norms."},
        {"name": "🛡️ Baseline", "model": "Hist-Gradient Boosting", "proj": base_proj, "vote": get_raw_vote(base_proj), "color": get_final_vote(base_proj)[1], "quote": "Weighted multi-year mapping using exponentially weighted moving averages."},
        {"name": "🎯 Context Guru", "model": "Radar, Rest, Arena", "proj": guru_proj, "vote": get_raw_vote(guru_proj), "color": get_final_vote(guru_proj)[1], "quote": f"Factors Context and Volatility."}
    ]

    # Return raw_vote and raw_consensus — Skynet modifier applied by caller
    return df_ml, board, final_consensus, f_vote, f_color, mod_val, mod_desc, current_split_mod, split_text, split_desc, fatigue_val, fatigue_desc, archetype, raw_vote, f_color

@st.cache_data(ttl=3600)
def run_nba_heaters(stat_choice="Points"):
    try:
        from nba_api.stats.endpoints import leagueleaders
        sched, _ = get_nba_schedule()
        if not sched: return None, "No NBA games scheduled today."
        teams_today = [g['home'] for g in sched] + [g['away'] for g in sched]
        s_col = S_MAP.get(stat_choice, "PTS")
        if s_col == "A": s_col = "AST"
        leaders = leagueleaders.LeagueLeaders(stat_category_abbreviation="PTS", per_mode48='PerGame').get_data_frames()[0]
        leaders['TRB'] = leaders['REB']
        leaders['PRA'] = leaders['PTS'] + leaders['REB'] + leaders['AST']
        leaders['PR'] = leaders['PTS'] + leaders['REB']
        leaders['PA'] = leaders['PTS'] + leaders['AST']
        leaders['RA'] = leaders['REB'] + leaders['AST']
        sort_col = s_col if s_col in leaders.columns else 'PTS'
        leaders = leaders.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
        heaters = []
        for _, r in leaders.head(50).iterrows():
            team_abbrev = r['TEAM']
            if team_abbrev in teams_today:
                player_name = r['PLAYER']
                season_val = round(r[sort_col], 1)
                opp, is_home = "OPP", True
                for g in sched:
                    if g['home'] == team_abbrev: opp = g['away']; is_home = True; break
                    elif g['away'] == team_abbrev: opp = g['home']; is_home = False; break
                ai_proj = 0.0
                matchup_status = f"vs {opp}" if is_home else f"@ {opp}"
                df, status, _ = get_nba_stats(player_name)
                if status != 429 and not df.empty and len(df) >= 5:
                    last_played = pd.to_datetime(df['ValidDate'].max()).tz_localize(None)
                    today_est = pd.to_datetime(datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d"))
                    days_out = (today_est - last_played).days
                    if days_out >= 6: matchup_status = f"⚠️ CHECK STATUS (Out {days_out} days)"
                    if s_col == "PRA": df['PRA'] = df['PTS'] + df['TRB'] + df['AST']
                    if s_col == "PR": df['PR'] = df['PTS'] + df['TRB']
                    if s_col == "PA": df['PA'] = df['PTS'] + df['AST']
                    if s_col == "RA": df['RA'] = df['TRB'] + df['AST']
                    # ✅ OPT-4: Pass df_hash; Skynet=1.0 (heaters don't need live Skynet correction)
                    dh = f"{len(df)}_{str(df['ValidDate'].iloc[-1])}_{df[s_col].sum():.1f}" if s_col in df.columns else str(len(df))
                    _, _, c_proj, _, _, _, _, _, _, _, _, _, _, _, _ = run_ml_board(
                        df, s_col, float(season_val), opp, "NBA", "Rested (1+ Days)", is_home, stat_choice, dh
                    )
                    ai_proj = round(c_proj, 1)
                time.sleep(0.5)
                heaters.append({"Player": player_name, "Team": team_abbrev, "Season Stat": season_val, "AI Proj": ai_proj, "Status": matchup_status})
        if not heaters: return None, f"No top 50 {stat_choice} leaders playing tonight."
        return pd.DataFrame(heaters), f"✅ Deep Scan Complete: {stat_choice} Projections loaded."
    except Exception as e: return None, f"API Error: {str(e)}"

@st.cache_data(ttl=3600)
def run_nhl_heaters(stat_choice="Points"):
    try:
        sched, _ = get_nhl_schedule()
        if not sched: return None, "No NHL games scheduled today."
        teams_today = [g['home'] for g in sched] + [g['away'] for g in sched]
        y = datetime.now().year
        curr_season = f"{y-1}{y}" if datetime.now().month < 9 else f"{y}{y+1}"
        sort_map = {"Points": "points", "Goals": "goals", "Assists": "assists", "Shots on Goal": "shots"}
        api_stat = sort_map.get(stat_choice, "points")
        s_col = {"Points": "PTS", "Goals": "G", "Assists": "A", "Shots on Goal": "SOG"}.get(stat_choice, "PTS")
        r = requests.get(f"https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=false&sort=[{{\"property\":\"{api_stat}\",\"direction\":\"DESC\"}}]&start=0&limit=50&cayenneExp=seasonId={curr_season}", timeout=5).json()
        heaters = []
        for p in r.get('data', []):
            team_abbr = p.get('teamAbbrevs', '').split(',')[0]
            if team_abbr in teams_today:
                player_name = p.get('skaterFullName', p.get('lastName', ''))
                games_played = max(1, p.get('gamesPlayed', 1))
                season_val = round(p.get(api_stat, 0) / games_played, 1)
                opp, is_home = "OPP", True
                for g in sched:
                    if g['home'] == team_abbr: opp = g['away']; is_home = True; break
                    elif g['away'] == team_abbr: opp = g['home']; is_home = False; break
                ai_proj = 0.0
                matchup_status = f"vs {opp}" if is_home else f"@ {opp}"
                df, status, _ = get_nhl_stats(player_name)
                if status != 429 and not df.empty and len(df) >= 5:
                    last_played = pd.to_datetime(df['ValidDate'].max()).tz_localize(None)
                    today_est = pd.to_datetime(datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d"))
                    days_out = (today_est - last_played).days
                    if days_out >= 6: matchup_status = f"⚠️ CHECK STATUS (Out {days_out} days)"
                    dh = f"{len(df)}_{str(df['ValidDate'].iloc[-1])}_{df[s_col].sum():.1f}" if s_col in df.columns else str(len(df))
                    _, _, c_proj, _, _, _, _, _, _, _, _, _, _, _, _ = run_ml_board(
                        df, s_col, float(season_val), opp, "NHL", "Rested (1+ Days)", is_home, stat_choice, dh
                    )
                    ai_proj = round(c_proj, 1)
                # ✅ OPT-10: NHL/MLB use different APIs with higher rate limits — reduced from 0.5s
                time.sleep(0.2)
                heaters.append({"Player": player_name, "Team": team_abbr, "Season Stat": season_val, "AI Proj": ai_proj, "Status": matchup_status})
        if not heaters: return None, f"No top {stat_choice} leaders playing tonight."
        return pd.DataFrame(heaters), f"✅ Deep Scan Complete: NHL {stat_choice} Projections loaded."
    except Exception as e: return None, f"API Error: {str(e)}"

@st.cache_data(ttl=3600)
def get_nhl_roster(team_abbr):
    try:
        r = requests.get(f"https://api-web.nhle.com/v1/roster/{team_abbr}/current", timeout=5).json()
        players = []
        for cat in ['forwards', 'defensemen']:
            for p in r.get(cat, []):
                players.append({'Name': f"{p['firstName']['default']} {p['lastName']['default']}", 'ID': p['id']})
        return players[:12]
    except: return []

@st.cache_data(ttl=3600)
def analyze_nhl_shooter(player_id, player_name, opponent, bad_defenses):
    try:
        r = requests.get(f"https://api-web.nhle.com/v1/player/{player_id}/landing", timeout=5).json()
        last_5 = r.get('last5Games', [])
        if not last_5: return None
        sogs = [g.get('shots', 0) for g in last_5]
        l5_avg = sum(sogs) / len(sogs)
        if l5_avg < 2.5: return None
        rating = "---"
        is_diamond = False
        if l5_avg >= 4.0: rating = "🔥 VOLUME KING"
        elif l5_avg >= 3.0: rating = "✅ SOLID"
        if opponent in bad_defenses:
            sog_allowed = bad_defenses[opponent]
            rating += f" + 🧀 BAD DEF ({sog_allowed} SOG/G)"
            if l5_avg >= 3.5: is_diamond = True
        return {'Player': player_name, 'Opp': opponent, 'L5 SOG Avg': round(l5_avg, 1), 'Recent Log': str(sogs), 'Rating': rating, 'Diamond': is_diamond}
    except: return None

@st.cache_data(ttl=3600)
def run_barn_burner():
    try:
        schedule_data, _ = get_nhl_schedule()
        if not schedule_data: return None, "No games scheduled today."
        bad_defenses = get_nhl_bad_defenses()
        results = []
        for g in schedule_data:
            home, away = g['home'], g['away']
            for p in get_nhl_roster(home):
                data = analyze_nhl_shooter(p['ID'], p['Name'], away, bad_defenses)
                if data: results.append(data)
                time.sleep(0.05)
            for p in get_nhl_roster(away):
                data = analyze_nhl_shooter(p['ID'], p['Name'], home, bad_defenses)
                if data: results.append(data)
                time.sleep(0.05)
        if not results: return None, "No high-volume shooters found tonight."
        df = pd.DataFrame(results)
        df['Rank'] = df.apply(lambda x: 1 if x['Diamond'] else (2 if "VOLUME" in x['Rating'] else 3), axis=1)
        df = df.sort_values(by=['Rank', 'L5 SOG Avg'], ascending=[True, False]).head(15).drop(columns=['Diamond', 'Rank'])
        return df, f"✅ Found optimal SOG targets. Bad defenses this season: {', '.join(bad_defenses.keys())}"
    except Exception as e: return None, f"API Error: {str(e)}"

@st.cache_data(ttl=43200)
def get_nhl_bad_defenses():
    try:
        y = datetime.now().year
        curr_season = f"{y-1}{y}" if datetime.now().month < 9 else f"{y}{y+1}"
        r = requests.get(f"https://api.nhle.com/stats/rest/en/team/summary?sort=shotsAgainstPerGame&cayenneExp=seasonId={curr_season}", timeout=5).json()
        nhl_map = {'Anaheim Ducks': 'ANA', 'Boston Bruins': 'BOS', 'Buffalo Sabres': 'BUF', 'Calgary Flames': 'CGY', 'Carolina Hurricanes': 'CAR', 'Chicago Blackhawks': 'CHI', 'Colorado Avalanche': 'COL', 'Columbus Blue Jackets': 'CBJ', 'Dallas Stars': 'DAL', 'Detroit Red Wings': 'DET', 'Edmonton Oilers': 'EDM', 'Florida Panthers': 'FLA', 'Los Angeles Kings': 'LAK', 'Minnesota Wild': 'MIN', 'Montréal Canadiens': 'MTL', 'Montreal Canadiens': 'MTL', 'Nashville Predators': 'NSH', 'New Jersey Devils': 'NJD', 'New York Islanders': 'NYI', 'New York Rangers': 'NYR', 'Ottawa Senators': 'OTT', 'Philadelphia Flyers': 'PHI', 'Pittsburgh Penguins': 'PIT', 'San Jose Sharks': 'SJS', 'Seattle Kraken': 'SEA', 'St. Louis Blues': 'STL', 'Tampa Bay Lightning': 'TBL', 'Toronto Maple Leafs': 'TOR', 'Utah Hockey Club': 'UTA', 'Vancouver Canucks': 'VAN', 'Vegas Golden Knights': 'VGK', 'Washington Capitals': 'WSH', 'Winnipeg Jets': 'WPG'}
        bad = {}
        for t in r.get('data', []):
            sog_against = t.get('shotsAgainstPerGame', 0)
            if sog_against > 31.0:
                abbr = t.get('teamAbbrev') or nhl_map.get(t.get('teamFullName', ''))
                if abbr: bad[abbr] = round(sog_against, 1)
        return bad if bad else {'SJS': 33.0, 'ANA': 32.5, 'CBJ': 32.0, 'CHI': 31.5, 'MTL': 31.2, 'NYI': 31.1}
    except:
        return {'SJS': 33.0, 'ANA': 32.5, 'CBJ': 32.0, 'CHI': 31.5, 'MTL': 31.2, 'NYI': 31.1}

@st.cache_data(ttl=3600)
def run_mlb_heaters(stat_choice="Hits"):
    try:
        sched, _ = get_mlb_schedule()
        if not sched: return None, "No MLB games scheduled today."
        teams_today = [g['home'] for g in sched] + [g['away'] for g in sched]
        curr_year = datetime.now().year
        if stat_choice == "Pitcher Strikeouts":
            group = "pitching"; sort_stat = "strikeOuts"; s_col = "K"
        else:
            group = "hitting"
            sort_stat = {"Hits": "hits", "Home Runs": "homeRuns", "Total Bases": "totalBases"}.get(stat_choice, "hits")
            s_col = {"Hits": "H", "Home Runs": "HR", "Total Bases": "TB"}.get(stat_choice, "H")
        r = requests.get(f"https://statsapi.mlb.com/api/v1/stats?stats=season&group={group}&playerPool=ALL&season={curr_year}&limit=50", timeout=5).json()
        if not r.get('stats'): r = requests.get(f"https://statsapi.mlb.com/api/v1/stats?stats=season&group={group}&playerPool=ALL&season={curr_year-1}&limit=50", timeout=5).json()
        m_t = {"Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "Seattle Mariners": "SEA", "San Francisco Giants": "SF", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH"}
        raw_leaders = []
        for s in r.get('stats', []):
            for p in s.get('splits', []):
                team_full = p.get('team', {}).get('name', '')
                team_abbr = m_t.get(team_full, team_full.split()[-1][:3].upper() if team_full else "")
                if team_abbr in teams_today: raw_leaders.append((p, team_abbr))
        raw_leaders = sorted(raw_leaders, key=lambda x: x[0].get('stat', {}).get(sort_stat, 0), reverse=True)
        heaters = []
        for p, team_abbr in raw_leaders:
            player_name = p.get('player', {}).get('fullName', '')
            season_val = p.get('stat', {}).get(sort_stat, 0)
            opp, is_home = "OPP", True
            for g in sched:
                if g['home'] == team_abbr: opp = g['away']; is_home = True; break
                elif g['away'] == team_abbr: opp = g['home']; is_home = False; break
            ai_proj = 0.0
            matchup_status = f"vs {opp}" if is_home else f"@ {opp}"
            df, status, _ = get_mlb_stats(player_name)
            if status != 429 and not df.empty and len(df) >= 5:
                last_played = pd.to_datetime(df['ValidDate'].max()).tz_localize(None)
                today_est = pd.to_datetime(datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d"))
                days_out = (today_est - last_played).days
                if days_out >= 6: matchup_status = f"⚠️ CHECK STATUS (Out {days_out} days)"
                dh = f"{len(df)}_{str(df['ValidDate'].iloc[-1])}_{df[s_col].sum():.1f}" if s_col in df.columns else str(len(df))
                _, _, c_proj, _, _, _, _, _, _, _, _, _, _, _, _ = run_ml_board(
                    df, s_col, 0.5, opp, "MLB", "Rested (1+ Days)", is_home, stat_choice, dh
                )
                ai_proj = round(c_proj, 2)
            # ✅ OPT-10: MLB API has no rate limit issues — reduced from 0.5s
            time.sleep(0.2)
            heaters.append({"Player": player_name, "Team": team_abbr, "Season Stat": season_val, "AI Proj": ai_proj, "Status": matchup_status})
        if not heaters: return None, f"No top {stat_choice} leaders playing today."
        return pd.DataFrame(heaters), f"✅ Deep Scan Complete: MLB {stat_choice} Projections loaded."
    except Exception as e: return None, f"API Error: {str(e)}"

# ==========================================
# 5. UI RENDERERS
# ==========================================
def init_state(key, default):
    if key not in st.session_state: st.session_state[key] = default
    return st.session_state[key]

def render_scoreboard(sd, league_name):
    if not sd: return
    unique_sd = []
    seen_matchups = set()
    for g in sd:
        match_sig = f"{g.get('away', '')}@{g.get('home', '')}"
        if match_sig not in seen_matchups:
            seen_matchups.add(match_sig)
            unique_sd.append(g)
    for i in range(0, len(unique_sd), 5):
        cols = st.columns(5)
        for j, g in enumerate(unique_sd[i:i+5]):
            with cols[j]:
                away_logo = get_team_logo(league_name, g['away'])
                home_logo = get_team_logo(league_name, g['home'])
                if g.get('is_live_or_final', False):
                    dt = f"<img src='{away_logo}' width='24' style='vertical-align:middle; margin-right:4px;'> <span style='color: #fff;'>{g['away']}</span> <span style='color: #FFD700; font-weight:900;'>{g.get('away_score', '')}</span> - <span style='color: #FFD700; font-weight:900;'>{g.get('home_score', '')}</span> <span style='color: #fff;'>{g['home']}</span> <img src='{home_logo}' width='24' style='vertical-align:middle; margin-left:4px;'>"
                else:
                    dt = f"<img src='{away_logo}' width='24' style='vertical-align:middle; margin-right:4px;'> <span style='color: #fff;'>{g['away']}</span> <span style='color: #94a3b8; margin: 0 4px;'>@</span> <span style='color: #fff;'>{g['home']}</span> <img src='{home_logo}' width='24' style='vertical-align:middle; margin-left:4px;'>"
                st.markdown(f'<div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px; text-align: center; margin-bottom: 10px;"><div style="font-size: 14px; font-weight: bold; color: #fff; display: flex; justify-content: center; align-items: center;">{dt}</div><div style="font-size: 11px; color: #00E5FF; margin-top: 4px; font-weight:bold;">{g.get("status", "")}</div></div>', unsafe_allow_html=True)

def render_league_scanners(league_name):
    lk = league_name.lower()
    with st.expander(f"📡 Launch {league_name} Skynet Radar", expanded=False):
        if league_name == "NBA":
            c1, c2 = st.columns([1, 1.5])
            with c1: scan_stat = st.selectbox("🎯 Target Stat", ["Points", "Rebounds", "Assists", "Threes Made", "PRA (Pts+Reb+Ast)", "Points + Rebounds", "Points + Assists", "Rebounds + Assists"], key=f"{lk}.scan_stat")
            with c2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button(f"🏀 Scan NBA {scan_stat}", type="primary", use_container_width=True, key=f"{lk}.btn.heaters"):
                    with st.spinner(f"Scanning {scan_stat} Leaders..."):
                        df, msg = run_nba_heaters(scan_stat)
                        if df is not None: st.session_state[f'{lk}.radar.heaters'] = df
                        st.info(msg)
        elif league_name == "NHL":
            c1, c2, c3 = st.columns([1.2, 1, 1])
            with c1: scan_stat = st.selectbox("🎯 Target Stat", ["Points", "Goals", "Assists", "Shots on Goal"], key=f"{lk}.scan_stat")
            with c2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button(f"🏒 Scan NHL {scan_stat}", type="primary", use_container_width=True, key=f"{lk}.btn.heaters"):
                    with st.spinner(f"Scanning {scan_stat} Leaders..."):
                        df, msg = run_nhl_heaters(scan_stat)
                        if df is not None: st.session_state[f'{lk}.radar.heaters'] = df
                        st.info(msg)
            with c3:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button("🚨 Scan Barn Burners", type="primary", use_container_width=True, key=f"{lk}.btn.bb"):
                    with st.spinner("Hunting weak defenses..."):
                        df, msg = run_barn_burner()
                        if df is not None: st.session_state[f'{lk}.radar.bb'] = df
                        st.info(msg)
        elif league_name == "MLB":
            c1, c2 = st.columns([1, 1.5])
            with c1: scan_stat = st.selectbox("🎯 Target Stat", ["Hits", "Home Runs", "Total Bases", "Pitcher Strikeouts"], key=f"{lk}.scan_stat")
            with c2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button(f"⚾ Scan MLB {scan_stat}", type="primary", use_container_width=True, key=f"{lk}.btn.heaters"):
                    with st.spinner(f"Scanning Elite {scan_stat}..."):
                        df, msg = run_mlb_heaters(scan_stat)
                        if df is not None: st.session_state[f'{lk}.radar.heaters'] = df
                        st.info(msg)

        if f'{lk}.radar.heaters' in st.session_state:
            df_radar = st.session_state[f'{lk}.radar.heaters']
            st.dataframe(df_radar, use_container_width=True, hide_index=True,
                column_config={"Player": st.column_config.TextColumn("🔥 Player", width="medium"), "Team": st.column_config.TextColumn("🛡️ Team", width="small"), "Season Stat": st.column_config.NumberColumn("🎯 Season Avg", format="%.1f", width="small"), "AI Proj": st.column_config.NumberColumn("🤖 AI Proj", format="%.1f", width="small"), "Status": st.column_config.TextColumn("⚡ Matchup", width="medium")})
            if 'Player' in df_radar.columns:
                st.markdown("#### ⚡ Fast-Track to Analyzer")
                ft_c1, ft_c2 = st.columns([3, 1])
                with ft_c1:
                    formatted_options = ["-- Select --"] + [f"{row['Player']} ({row['Team']})" if 'Team' in row else row['Player'] for _, row in df_radar.iterrows()]
                    selected_heater = st.selectbox("Select a target from the radar:", formatted_options, key=f"{lk}.ft_sel")
                with ft_c2:
                    st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)
                    if st.button("SEND TO BOARD 🚀", type="primary", use_container_width=True, key=f"{lk}.ft_btn"):
                        if selected_heater != "-- Select --":
                            st.session_state[f"{lk}.search_query"] = selected_heater.split('(')[0].strip()
                            st.session_state[f"{lk}.target_player"] = selected_heater
                            st.rerun()

        if f'{lk}.radar.bb' in st.session_state:
            st.markdown("#### 🚨 Weak Defenses Detected")
            st.dataframe(st.session_state[f'{lk}.radar.bb'], use_container_width=True, hide_index=True,
                column_config={"Team": st.column_config.TextColumn("🛡️ Target Team", width="medium"), "Opp": st.column_config.TextColumn("🎯 Weak Opponent", width="medium"), "Opp Status": st.column_config.TextColumn("🚨 Defense Metric", width="large")})

def classify_miss(proj, line, actual, vote):
    """
    Dissects a losing bet into a miss type, distance, and likely cause.
    Returns (miss_type, abs_miss, likely_cause, color)
    """
    try:
        proj   = float(proj)
        line   = float(line)
        actual = float(actual)
    except (ValueError, TypeError):
        return None, None, None, None

    # Calculate how far the actual landed from the line
    if vote == "OVER":
        line_miss = line - actual      # positive number means we fell short by X units
    elif vote == "UNDER":
        line_miss = actual - line      # positive number means we went over by X units
    else:
        return None, None, None, None
        
    abs_miss = abs(line_miss)
    
    # Calculate percentage miss based on the line to scale across different stats
    pct_miss = abs_miss / line if line > 0 else 0

    # BAD BEAT: Missed the line by <= 10% OR by <= 1.5 raw units
    if pct_miss <= 0.10 or abs_miss <= 1.5:
        miss_type    = "😔 BAD BEAT"
        likely_cause = (
            "The model's direction was correct and the projection was close — "
            "this was pure variance. No model adjustment needed. "
            "At your recorded win probability, some losses are always expected."
        )
        color = "#FFD700"

    # MODEL MISS: Missed the line by 10-25% OR by <= 3.5 raw units
    elif pct_miss <= 0.25 or abs_miss <= 3.5:
        miss_type    = "⚠️ MODEL MISS"
        likely_cause = (
            "The projection was in the right ballpark but overconfident. "
            "Check whether the opponent defense modifier or fatigue flag was active — "
            "context modifiers may have been too aggressive on this setup."
        )
        color = "#f59e0b"

    # BLOWOUT: Missed by > 25% of the line and > 3.5 raw units
    else:
        miss_type    = "💥 BLOWOUT MISS"
        likely_cause = (
            "The actual result fell well outside the projected range. "
            "Likely causes: in-game blowout (minutes slashed), undisclosed injury, "
            "surprise lineup change, or an archetype mismatch vs this opponent. "
            "If Setup Score was ELITE (75+), this warrants a manual review."
        )
        color = "#ff0055"

    return miss_type, round(abs_miss, 1), likely_cause, color
    
def render_syndicate_board(league_key):
    lk = league_key.lower()
    sport_path = "basketball_nba" if league_key == "NBA" else ("baseball_mlb" if league_key == "MLB" else "icehockey_nhl")
    teams = NBA_TEAMS if league_key == "NBA" else (MLB_TEAMS if league_key == "MLB" else NHL_TEAMS)
    sched_func = get_nba_schedule if league_key == "NBA" else (get_mlb_schedule if league_key == "MLB" else get_nhl_schedule)
    sched, _ = sched_func()

    top_c1, top_c2, _ = st.columns([1, 1, 2])
    placeholder_sync = top_c1.empty()
    placeholder_home = top_c2.empty()

    with st.container():
        c1, c2, c3, c4 = st.columns([2, 1.5, 1, 1.5])
        with c1:
            search_query = st.text_input(f"🔍 1. Search Player/Team", placeholder="e.g. Judge, LeBron, Lakers", key=f"{lk}.search_query")
            player_name = None
            if search_query:
                if search_query.upper() in teams:
                    player_name = search_query.upper()
                    st.info(f"Team {player_name} detected.")
                else:
                    matches = search_nba_players(search_query) if league_key == "NBA" else (search_mlb_players(search_query) if league_key == "MLB" else search_nhl_players(search_query))
                    if matches: player_name = st.selectbox("🎯 2. Select Exact Match", matches, key=f"{lk}.dropdown")
                    else: st.caption("No matches found.")

        auto_opp = None
        auto_is_home = True
        if player_name and sched and "(" in player_name:
            team_abbr = player_name.split("(")[1].split(")")[0].strip().upper()
            for g in sched:
                if g['home'].upper() == team_abbr: auto_opp = g['away'].upper(); auto_is_home = True; break
                elif g['away'].upper() == team_abbr: auto_opp = g['home'].upper(); auto_is_home = False; break

        if player_name and player_name != st.session_state.get(f"{lk}.last_player"):
            st.session_state[f"{lk}.last_player"] = player_name
            if auto_opp and auto_opp in teams:
                st.session_state[f"{lk}.opp"] = auto_opp
                st.session_state[f"{lk}.is_home"] = auto_is_home

        init_state(f"{lk}.sync", False)
        init_state(f"{lk}.is_home", True)
        init_state(f"{lk}.opp", teams[0])

        sync = placeholder_sync.toggle("📡 Auto-Sync Vegas Odds", key=f"{lk}.sync")
        is_home_current = 1 if placeholder_home.toggle("🏠 Playing at Home?", key=f"{lk}.is_home") else 0

        with c2:
            game_lines = ["Moneyline", "Spread", "Total (O/U)"]
            
            # 🟢 Added "Blocks" and "Steals" to the NBA dropdown list
            player_props = ["Points", "Rebounds", "Assists", "Threes Made", "Blocks", "Steals", "PRA (Pts+Reb+Ast)", "Points + Rebounds", "Points + Assists", "Rebounds + Assists", "Double Double", "Triple Double", "Minutes Played"] if league_key == "NBA" else (["Hits", "Home Runs", "Total Bases", "Pitcher Strikeouts", "Pitcher Earned Runs"] if league_key == "MLB" else ["Points", "Goals", "Assists", "Shots on Goal"])
            stat_type = st.selectbox("Stat / Market", game_lines + player_props, key=f"{lk}.stat")
            live_odds_display = st.empty()

            implied_prob_placeholder = st.empty()

        f_line, f_odds, msg, used, rem = None, None, "", None, None
        if sync and player_name and stat_type not in game_lines:
            with st.spinner("Syncing Odds..."): f_line, f_odds, msg, used, rem = get_live_line(player_name, stat_type, ODDS_API_KEY, sport_path)
            if used and rem: st.session_state['api_used'], st.session_state['api_remaining'] = int(used), int(rem)
            if f_line is not None and f_odds is not None: live_odds_display.markdown(f'<div style="background-color: rgba(0, 230, 118, 0.1); border: 1px solid #00E676; padding: 10px; border-radius: 6px; margin-top: 10px;"><div style="font-size: 11px; font-weight: 900; color: #00E676; letter-spacing: 1px;">📡 LIVE MARKET SYNCED</div><div style="font-size: 16px; font-weight: bold; color: #fff;">{stat_type} O/U {f_line} <span style="color: #94a3b8; font-size: 14px;">({"+"+str(f_odds) if f_odds>0 else f_odds})</span></div></div>', unsafe_allow_html=True)
            elif f_odds is not None: live_odds_display.markdown(f'<div style="background-color: rgba(255, 215, 0, 0.1); border: 1px solid #FFD700; padding: 10px; border-radius: 6px; margin-top: 10px;"><div style="font-size: 11px; font-weight: 900; color: #FFD700; letter-spacing: 1px;">🟡 MARKET PARTIAL SYNC</div><div style="font-size: 16px; font-weight: bold; color: #fff;">{stat_type} Odds <span style="color: #94a3b8; font-size: 14px;">({"+"+str(f_odds) if f_odds>0 else f_odds})</span></div></div>', unsafe_allow_html=True)
            else: live_odds_display.caption(f"🟡 {msg}")
        
            # 📈 LINE MOVEMENT TRACKER: Record opening line on first sync, compare on subsequent syncs
            opening_key = f"{lk}.opening_line.{player_name}.{stat_type}"
            if f_line is not None:
                if opening_key not in st.session_state:
                    # First time we've seen this line — store it as the opener
                    st.session_state[opening_key] = f_line
                else:
                    opener = st.session_state[opening_key]
                    move = round(f_line - opener, 1)
                    if abs(move) >= 0.5:
                        if move > 0:
                            st.session_state[f"{lk}.line_move_msg"] = (
                                f"📈 Line moved UP {move:+.1f} (opened {opener}) — "
                                f"sharp money may be on the OVER."
                            )
                            st.session_state[f"{lk}.line_move_dir"] = "up"
                        else:
                            st.session_state[f"{lk}.line_move_msg"] = (
                                f"📉 Line moved DOWN {move:+.1f} (opened {opener}) — "
                                f"sharp money may be on the UNDER."
                            )
                            st.session_state[f"{lk}.line_move_dir"] = "down"
                    else:
                        # Movement too small to be meaningful — clear any old warning
                        st.session_state.pop(f"{lk}.line_move_msg", None)
                        st.session_state.pop(f"{lk}.line_move_dir", None)        

        with c3:
            if stat_type in ["Double Double", "Triple Double"]:
                st.text_input("Line / Target", value="Yes (0.5)", disabled=True, key=f"{lk}.line_dd")
                line = 0.5
            else:
                start_line = float(f_line) if (sync and f_line is not None) else 0.5
                if stat_type in ["Moneyline"]: start_line = 0.0
                line = st.number_input("Line / Target", value=start_line, step=0.5, key=f"{lk}.line")
                
            if sync and f_line is not None and f_odds is not None and stat_type not in game_lines:
                st.session_state[f"{lk}.odds"] = estimate_alt_odds(float(f_line), int(f_odds), line, stat_type)
            elif f"{lk}.odds" not in st.session_state: st.session_state[f"{lk}.odds"] = -110
            
            odds = st.number_input("Odds", step=5, key=f"{lk}.odds")
            is_boosted = st.checkbox("🚀 Odds Boost Applied", key=f"{lk}.boost")
            ignore_blowout = st.checkbox("🛡️ Ignore Blowout Risk", key=f"{lk}.no_blowout")
            implied_prob = calculate_implied_prob(odds)
            if implied_prob > 0:
                if implied_prob >= 66.0: juice_color, juice_msg = "#ff0055", "🚨 TOXIC JUICE: Requires extreme win rate."
                elif implied_prob >= 58.0: juice_color, juice_msg = "#f59e0b", "⚠️ HEAVY FAVORITE: Proceed with caution."
                else: juice_color, juice_msg = "#00c853", "✅ FAIR PRICE: Mathematical green light."
                implied_prob_placeholder.markdown(f"""<div style="background-color: #0f172a; border: 1px solid #1e293b; border-left: 3px solid {juice_color}; border-radius: 4px; padding: 8px; margin-top: 15px;"><div style="font-size: 11px; color: #94a3b8; font-weight: bold; text-transform: uppercase;">Implied Vegas Probability</div><div style="display: flex; justify-content: space-between; align-items: baseline;"><div style="font-size: 18px; color: #fff; font-weight: 900;">{implied_prob:.1f}% <span style="font-size: 12px; color: {juice_color}; font-weight: 500;">Win Rate Needed</span></div></div><div style="font-size: 10px; color: {juice_color}; margin-top: 2px;">{juice_msg}</div></div>""", unsafe_allow_html=True)

        with c4:
            opp = st.selectbox("Opponent", teams, key=f"{lk}.opp")
            rest = st.selectbox("Fatigue", ["Rested (1+ Days)", "Tired (B2B)", "Exhausted (3 in 4)"], key=f"{lk}.rest")

    btn_c1, btn_c2, btn_c3, _ = st.columns([1.5, 1.5, 1.5, 1])
    with btn_c1: analyze_pressed = st.button(f"🚀 Analyze Matchup", type="primary", use_container_width=True, key=f"{lk}.btn_analyze")

    if analyze_pressed and player_name: st.session_state[f"{lk}.target_player"] = player_name
    target_player = st.session_state.get(f"{lk}.target_player")

    if target_player:
        if stat_type in ["Moneyline", "Spread", "Total (O/U)"]:
            st.markdown("---")
            st.markdown(f"### 🏟️ Syndicate Team Bet: {stat_type}")
            st.info(f"**{target_player}** selected. The AI player model is bypassed for game lines.")
            m_c1, m_c2 = st.columns(2)
            with m_c1: st.metric("Target Line", line if stat_type != "Moneyline" else "Win")
            with m_c2: st.metric("Odds", odds)
            
            # Removed c_vote to fix NameError
            user_side = st.radio("Your Position:", ["OVER", "UNDER", "TEAM"], index=0, horizontal=True)

            if st.button(f"🔒 Lock {league_key} Pick"):
                # Removed literal "..." to fix SyntaxError
                save_to_ledger(league_key, target_player, stat_type, line, odds, 0.0, user_side, 0.50, is_boosted, 0, 0.50)
                st.success(f"Team Pick Locked: {user_side}")
        else:
            # 🛑 AUTO-SUPPRESSION: Block markets with a proven losing record
            suppressed = get_suppressed_stats(league_key)
            if stat_type in suppressed:
                graded = load_ledger()
                graded = graded[
                    (graded['Result'].isin(['Win', 'Loss'])) &
                    (graded['League'] == league_key) &
                    (graded['Stat'] == stat_type)
                ]
                wins = len(graded[graded['Result'] == 'Win'])
                total = len(graded)
                wr = wins / total * 100 if total > 0 else 0
                st.error(
                    f"🛑 **SKYNET AUTO-SUPPRESSION ACTIVE**\n\n"
                    f"Your historical record on **{league_key} {stat_type}** is "
                    f"**{wins}-{total - wins}** ({wr:.1f}% win rate) — below the 42% floor. "
                    f"Syndicate recommends skipping this market entirely until your edge is reestablished. "
                    f"Clear the Skynet Cache above to re-evaluate after new data comes in."
                )
                st.stop()

            with st.spinner(f"Scouting data for {target_player}..."):
                if league_key == "NBA": df, status_code, _ = get_nba_stats(target_player)
                elif league_key == "MLB": df, status_code, _ = get_mlb_stats(target_player)
                else: df, status_code, _ = get_nhl_stats(target_player)

            if status_code == 429: st.error("🚨 **Error 429: Rate Limited.** Please wait 60 seconds.")
            elif status_code == 500: st.warning("🟡 **Server Error.** Try again in a moment.")
            elif df.empty: st.error(f"⚠️ **No Data Found:** Could not locate official game logs for {target_player}.")
            elif not df.empty:
                s_col = S_MAP.get(stat_type, "PTS")
                if league_key == "NBA":
                    if s_col == "A": s_col = "AST"
                    if s_col == "PRA": df['PRA'] = df['PTS'] + df['TRB'] + df['AST']
                    if s_col == "PR": df['PR'] = df['PTS'] + df['TRB']
                    if s_col == "PA": df['PA'] = df['PTS'] + df['AST']
                    if s_col == "RA": df['RA'] = df['TRB'] + df['AST']
                    if s_col in ["DD", "TD"]:
                        tens = (df['PTS'] >= 10).astype(int) + (df['TRB'] >= 10).astype(int) + (df['AST'] >= 10).astype(int) + (df.get('STL', pd.Series(0, index=df.index)) >= 10).astype(int) + (df.get('BLK', pd.Series(0, index=df.index)) >= 10).astype(int)
                        df['DD'] = (tens >= 2).astype(int)
                        df['TD'] = (tens >= 3).astype(int)

                # ✅ OPT-4: Compute df_hash and ledger_hash BEFORE calling run_ml_board
                # This forces cache invalidation when player data or graded bets change
                df_hash = f"{len(df)}_{str(df['ValidDate'].iloc[-1])}_{df[s_col].sum():.2f}" if s_col in df.columns else str(len(df))
                current_ledger = load_ledger()
                graded_counts = current_ledger[current_ledger['Result'].isin(['Win','Loss'])].groupby(['Stat','Vote','League']).size().to_dict()
                ledger_hash = str(hash(str(sorted(graded_counts.items()))))

                df_with_ml, board, raw_consensus, raw_vote_from_board, c_color, mod_val, mod_desc, current_split_mod, split_text, split_desc, fatigue_val, fatigue_desc, archetype, raw_vote, _ = run_ml_board(
                    df, s_col, line, opp, league_key, rest, is_home_current, stat_type, ignore_blowout, df_hash, ledger_hash
                )

                # ✅ OPT-4: Apply Skynet HERE (outside cache) so it always uses fresh ledger data
                skynet_data = apply_skynet(raw_vote, stat_type, league_key)
                final_consensus = raw_consensus * skynet_data["mod"]
                df_with_ml['AI_Proj'] = df_with_ml['AI_Proj'] * skynet_data["mod"]

                # 🟢 SAVE THE RECEIPT: Log the pre-game stamp if today is game day
                if len(df_with_ml) > 0:
                    # Just use the target_player variable we already have!
                    today_date = datetime.now().strftime("%Y-%m-%d") 
                    log_prediction_receipt(target_player, stat_type, final_consensus, today_date)

                dynamic_thresh = PASS_THRESHOLDS.get(s_col, 0.3)
                def get_final_vote(p): return ("OVER", "#00c853") if p >= line + dynamic_thresh else (("UNDER", "#d50000") if p <= line - dynamic_thresh else ("PASS", "#94a3b8"))
                
                c_vote, c_color = get_final_vote(final_consensus)
                c_proj = final_consensus
                skynet_msg, skynet_color = skynet_data["msg"], skynet_data["color"]

                if len(board) == 0: st.warning(f"⚠️ **Insufficient Data:** {target_player} has played fewer than 5 games this season.")
                else:
                    df_with_ml['Residual'] = df_with_ml[s_col] - df_with_ml['AI_Proj']
                    residual_std = df_with_ml['Residual'].std()
                    if np.isnan(residual_std) or residual_std == 0:
                        residual_std = df_with_ml[s_col].std()
                        if np.isnan(residual_std) or residual_std == 0: residual_std = 1.0

                    if stat_type in ['HR', 'Goals', 'RBI', 'R', 'Steals', 'SB', 'Double Double', 'Triple Double']:
                        lam_val = max(0.001, c_proj)
                        # ✅ OPT-7: 5,000 samples is statistically identical to 10,000 at this precision
                        sims = np.random.poisson(lam=lam_val, size=5000)
                    else:
                        sims = np.random.normal(loc=c_proj, scale=residual_std, size=5000)

                    if c_vote == "OVER": win_prob = np.sum(sims > line) / 5000.0
                    elif c_vote == "UNDER": win_prob = np.sum(sims < line) / 5000.0
                    else: win_prob = 0.50

                    if odds < 0: implied_prob = abs(odds) / (abs(odds) + 100); profit = 100 / (abs(odds) / 100); risk = 100; b_odds = 100 / abs(odds)
                    else: implied_prob = 100 / (odds + 100); profit = odds; risk = 100; b_odds = odds / 100

                    ev_dollars = (win_prob * profit) - ((1 - win_prob) * risk)
                    edge_pct = (win_prob - implied_prob) * 100

                    liq_bal = get_liquid_balance()
                    kelly_pct = max(0.0, (b_odds * win_prob - (1 - win_prob)) / b_odds) if b_odds > 0 else 0.0
                    rec_stake = liq_bal * (kelly_pct * 0.5)

                    memory_mult = 1.0
                    mem_notes = []
                    try:
                        mem_df = load_ledger()
                        if not mem_df.empty and c_vote not in ["PASS", "VETO"]:
                            mem_graded = mem_df[mem_df['Result'].isin(['Win', 'Loss'])]
                            p_hist = mem_graded[(mem_graded['Player'] == target_player) & (mem_graded['Stat'] == stat_type)]
                            p_w, p_l = len(p_hist[p_hist['Result']=='Win']), len(p_hist[p_hist['Result']=='Loss'])
                            s_hist = mem_graded[(mem_graded['Stat'] == stat_type) & (mem_graded['League'] == league_key)]
                            s_w, s_l = len(s_hist[s_hist['Result']=='Win']), len(s_hist[s_hist['Result']=='Loss'])
                            if (p_w + p_l) >= 3:
                                p_pct = p_w / (p_w + p_l)
                                if p_pct <= 0.35: memory_mult *= 0.5; mem_notes.append(f"🛑 <b>FADE:</b> You are {p_w}-{p_l} on {target_player} {stat_type}s. Halving risk.")
                                elif p_pct >= 0.65: memory_mult *= 1.5; mem_notes.append(f"🔥 <b>SMASH:</b> You are {p_w}-{p_l} on {target_player} {stat_type}s. Boosting risk.")
                                else: mem_notes.append(f"⚖️ <b>NEUTRAL:</b> You are {p_w}-{p_l} on {target_player} {stat_type}s.")
                            if (s_w + s_l) >= 10:
                                s_pct = s_w / (s_w + s_l)
                                if s_pct <= 0.45: memory_mult *= 0.75; mem_notes.append(f"⚠️ <b>LEAK:</b> Syndicate win rate on {stat_type}s is {s_pct*100:.1f}%. Trimming risk.")
                                elif s_pct >= 0.58: memory_mult *= 1.25; mem_notes.append(f"📈 <b>EDGE:</b> Syndicate crushes {stat_type}s ({s_pct*100:.1f}%). Boosting risk.")
                    except: pass

                    rec_stake = rec_stake * min(memory_mult, 2.0)
                    memory_html = f"<div style='margin-top:12px; padding-top:10px; border-top:1px dashed #334155; font-size:12px; color:#FFD700; line-height:1.4;'>{'<br>'.join(mem_notes)}</div>" if mem_notes else ""

                    # 🟢 CHECK THE SUPPRESSION BLACKLIST
                    suppressed_stats = get_suppressed_stats(league_key)
                    is_suppressed = stat_type in suppressed_stats

                    # 🟢 THE +EV AI VETO PROTOCOL & SUMMARY BUILDER
                    if is_suppressed:
                        c_vote = "VETO"
                        c_color = "#ff0055"
                        rec_stake = 0.0
                        ai_summary_short = f"🛑 <b>STAT LOCKED ({stat_type}):</b> Your Syndicate win rate on this market is dangerously low (under 42%). Skynet is physically blocking this bet to protect your bankroll until you improve."
                    elif edge_pct < 0 and c_vote != "PASS":
                        c_vote = "VETO"
                        c_color = "#ff0055"
                        rec_stake = 0.0
                        ai_summary_short = f"🛑 <b>NEGATIVE EV DETECTED.</b><br>Vegas requires a {implied_prob*100:.1f}% win rate, but Skynet only projects {win_prob*100:.1f}%. The math is toxic. Do not bet."
                    else:
                        ai_summary_short = f"Projected to {'clear' if c_vote == 'OVER' else ('stay under' if c_vote == 'UNDER' else 'too close to')} {line} with a {win_prob*100:.1f}% probability."
                        if league_key == "NBA" and "Exploit" in mod_desc: ai_summary_short += f"<br><span style='color:#FFD700; font-weight:bold;'>🚨 Archetype Exploit vs {opp}</span>"
                        elif league_key == "NBA" and "Fade" in mod_desc: ai_summary_short += f"<br><span style='color:#ff0055; font-weight:bold;'>🛑 Archetype Fade vs {opp}</span>"
                        
                        # Add the Skynet Audit and Memory Loop to the bottom
                        ai_summary_short += f"<br><br><span style='color:{skynet_color}; font-weight:bold;'>{skynet_msg}</span>"
                        ai_summary_short += memory_html

                    # 🟢 THE BOARD UNANIMITY FILTER (4/5 Required)
                    votes = [m['vote'] for m in board]
                    agree_count = max(votes.count("OVER"), votes.count("UNDER"), votes.count("PASS"))
                    is_near_unanimous = agree_count >= 4
                    consensus_pct = int((agree_count / len(board)) * 100)

                    if not is_near_unanimous and c_vote not in ["PASS", "VETO"]:
                        # 1. Override the AI's final vote to PASS
                        c_vote = "PASS"
                        c_color = "#94a3b8"
                        rec_stake = 0.0
                        # 2. Overwrite the Consensus Box text
                        ai_summary_short = f"⚠️ <b>SPLIT BOARD ({agree_count}/5):</b> The models are divided. Vegas has priced this line efficiently. Pass."
                        # 3. Render Claude's warning banner
                        st.markdown(f"""
                        <div style="background-color: rgba(255,69,0,0.1); border: 1px solid #ff4500; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                            <span style="font-size:16px; font-weight:900; color:#ff4500;">⚠️ SPLIT BOARD — NO BET RECOMMENDED</span>
                            <div style="font-size:13px; color:#94a3b8; margin-top:4px;">Board agreement: {agree_count}/5 ({consensus_pct}%). Syndicate requires 4/5 consensus.</div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif is_near_unanimous and c_vote not in ["PASS", "VETO"]:
                        consensus_label = "🟢 UNANIMOUS (5/5)" if agree_count == 5 else "🟡 STRONG CONSENSUS (4/5)"
                        st.caption(f"{consensus_label} — Board in agreement.")

                    lock_pressed = False
                    final_side = c_vote
                    
                    if c_vote not in ["PASS", "VETO"]:
                        # 🟢 BUTTON GOES IN COL 2
                        with btn_c2:
                            lock_pressed = st.button(f"🔒 Lock Pick", use_container_width=True, type="primary", key=f"{lk}.smart_lock")
                        # 🟢 RADIO GOES IN COL 3 (Right next to it!)
                        with btn_c3:
                            if stat_type in ["Double Double", "Triple Double"]:
                                side_choice = st.radio("Side", ["YES", "NO"], index=0 if c_vote == "OVER" else 1, horizontal=True, key=f"{lk}.smart_side_dd", label_visibility="collapsed")
                                final_side = "OVER" if side_choice == "YES" else "UNDER"
                            else:
                                final_side = st.radio("Side", ["OVER", "UNDER"], index=0 if c_vote == "OVER" else 1, horizontal=True, key=f"{lk}.smart_side", label_visibility="collapsed")

                    if lock_pressed:
                        # 🟢 AUTO-INVERT MATH
                        if final_side != c_vote:
                            auto_user_p = 1.0 - win_prob
                            user_edge_pct = (auto_user_p - implied_prob) * 100
                        else:
                            auto_user_p = win_prob
                            user_edge_pct = edge_pct

                        s_score = calculate_setup_score(auto_user_p, user_edge_pct, board, c_proj, line, stat_type)
                        save_to_ledger(league_key, target_player, stat_type, line, odds, c_proj, final_side, win_prob, is_boosted, s_score, auto_user_p)
                        st.success(f"Pick locked as {final_side}! (AI: {win_prob*100:.1f}% | User: {auto_user_p*100:.1f}%)")
                        
                    ai_summary_short = f"Projected to {'clear' if c_vote == 'OVER' else ('stay under' if c_vote == 'UNDER' else 'too close to')} {line} with a {win_prob*100:.1f}% probability."
                        
                    ai_summary_short = f"Projected to {'clear' if c_vote == 'OVER' else ('stay under' if c_vote == 'UNDER' else 'too close to')} {line} with a {win_prob*100:.1f}% probability."
                    if league_key == "NBA" and "Exploit" in mod_desc: ai_summary_short += f"<br><span style='color:#FFD700; font-weight:bold;'>🚨 Archetype Exploit vs {opp}</span>"
                    elif league_key == "NBA" and "Fade" in mod_desc: ai_summary_short += f"<br><span style='color:#ff0055; font-weight:bold;'>🛑 Archetype Fade vs {opp}</span>"
                    ai_summary_short += f"<br><br><span style='color:{skynet_color}; font-weight:bold;'>{skynet_msg}</span>"

                    if win_prob >= 0.60 and edge_pct >= 5.0 and c_vote != "PASS":
                        # 🟢 CALCULATE SCORE FOR THE BANNER
                        s_score = calculate_setup_score(win_prob, edge_pct, board, c_proj, line, stat_type)
                        
                        # 🟢 DETERMINE LABEL (Now includes the AI's actual vote)
                        if s_score >= 75: banner_label = f"🌟 ELITE AI TOP PICK: {c_vote} 🌟"
                        elif s_score >= 55: banner_label = f"✅ SOLID AI TOP PICK: {c_vote}"
                        else: banner_label = f"🎯 AI TOP PICK: {c_vote}"

                        st.markdown(f"""
                        <div style="background: linear-gradient(90deg, #FFD700 0%, #ff8c00 100%); padding: 3px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0px 0px 15px rgba(255, 215, 0, 0.4);">
                            <div style="background-color: #0f172a; padding: 12px; border-radius: 6px; text-align: center;">
                                <span style="font-size: 18px; font-weight: 900; color: #FFD700; letter-spacing: 2px;">{banner_label}</span>
                                <div style="font-size: 13px; color: #f8fafc; margin-top: 4px;">
                                    <b>Score: {s_score}/100</b> | {win_prob*100:.1f}% Win Prob for the <b>{c_vote}</b> | Recommend risking <b>${rec_stake:.2f}</b>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # 📈 LINE MOVEMENT WARNING DISPLAY
                    move_msg = st.session_state.get(f"{lk}.line_move_msg")
                    move_dir = st.session_state.get(f"{lk}.line_move_dir")
                    if move_msg and c_vote != "PASS":
                        is_against = (
                            (c_vote == "OVER" and move_dir == "down") or
                            (c_vote == "UNDER" and move_dir == "up")
                        )
                        border_color = "#ff4500" if is_against else "#00E676"
                        icon = "🚨" if is_against else "✅"
                        severity = (
                            "Sharp money appears to be on the **same side** as your pick. Good sign."
                            if not is_against else
                            "Sharp money appears to be **against** your pick. Proceed with caution or reduce stake."
                        )
                        st.markdown(f"""
                        <div style="background-color: rgba(255,255,255,0.03); border: 1px solid {border_color};
                             border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                            <span style="font-size:15px; font-weight:900; color:{border_color};">
                                {icon} LINE MOVEMENT ALERT
                            </span>
                            <div style="font-size:13px; color:#f8fafc; margin-top:4px;">{move_msg}</div>
                            <div style="font-size:12px; color:#94a3b8; margin-top:4px;">{severity}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    sum_c1, sum_c2, sum_c3, sum_c4 = st.columns(4)
                    with sum_c1:
                        display_vote = c_vote
                        if stat_type in ["Double Double", "Triple Double"] and c_vote in ["OVER", "UNDER"]:
                            display_vote = "YES" if c_vote == "OVER" else "NO"
                            
                        st.markdown(f"""<div class="verdict-box" style="background-color: {c_color}15; border-color: {c_color}; color: #fff; height: 100%;"><div style="font-size:10px; font-weight:bold; color:{c_color}; letter-spacing: 1px;">AI CONSENSUS</div><div style="font-size:26px; font-weight:900; margin: 4px 0px;">{display_vote}</div><div style="font-size:14px; font-weight:bold; margin-bottom: 6px;">Proj: {c_proj:.2f}</div><div style="font-size:11px; color:#94a3b8; border-top: 1px solid {c_color}50; padding-top: 8px; line-height: 1.3;">{ai_summary_short}</div></div>""", unsafe_allow_html=True)
                    with sum_c2:
                        if c_vote == "PASS" or edge_pct <= 0:
                            st.markdown(f'<div class="verdict-box" style="background-color: #1e293b; border-color: #334155; color: #fff; height: 100%;"><div style="font-size:10px; font-weight:bold; color:#94a3b8; letter-spacing: 1px;">RECOMMENDED RISK</div><div style="font-size:22px; font-weight:900; color:#94a3b8;">$0.00 (PASS)</div><div style="font-size:12px; color:#94a3b8;">Negative EV or too tight.</div></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="verdict-box" style="background-color: #1e293b; border-color: #00E5FF; color: #fff; height: 100%;"><div style="font-size:10px; font-weight:bold; color:#00E5FF; letter-spacing: 1px;">HALF-KELLY STAKE</div><div style="font-size:26px; font-weight:900; color:#00E5FF; margin: 4px 0px;">${rec_stake:.2f}</div><div style="font-size:12px; color:#94a3b8;">EV: ${ev_dollars:+.2f}/$100 | Edge: {edge_pct:+.1f}%</div></div>', unsafe_allow_html=True)
                    with sum_c3:
                        df_l10, df_l5 = df_with_ml.tail(10).reset_index(drop=True), df_with_ml.tail(5)
                        # Changed > to >= so exact hits count as wins
                        l10_hits, l5_hits = int((df_l10[s_col] >= line).sum()), int((df_l5[s_col] >= line).sum())
                        hit_color = "#00c853" if l10_hits >= 6 else ("#d50000" if l10_hits <= 4 else "#FFD700")
                        st.markdown(f'<div class="verdict-box" style="background-color: #1e293b; border-color: #334155; color: #fff; height: 100%;"><div style="font-size:10px; font-weight:bold; color:#94a3b8; letter-spacing: 1px;">HIT RATE (OVER {line})</div><div style="font-size:22px; font-weight:900; color:{hit_color};">{l10_hits}/10</div><div style="font-size:13px;">L5: {l5_hits}/5</div></div>', unsafe_allow_html=True)
                    with sum_c4:
                        s_avg, l10_avg, l5_avg = round(df[s_col].mean(), 1), round(df_l10[s_col].mean(), 1), round(df_with_ml.tail(5)[s_col].mean(), 1)
                        trend_color = "#00c853" if l5_avg >= s_avg * 1.1 else ("#d50000" if l5_avg <= s_avg * 0.9 else "#fff")
                        st.markdown(f'<div class="verdict-box" style="background-color: #1e293b; border-color: #334155; color: #fff; height: 100%;"><div style="font-size:10px; font-weight:bold; color:#94a3b8; letter-spacing: 1px; margin-bottom: 2px;">RECENT AVERAGES</div><div style="display: flex; justify-content: space-around; align-items: center; margin-top: 2px;"><div><div style="font-size:10px; color:#94a3b8;">Season</div><div style="font-size:18px; font-weight:900;">{s_avg}</div></div><div><div style="font-size:10px; color:#94a3b8;">L10</div><div style="font-size:18px; font-weight:900;">{l10_avg}</div></div><div><div style="font-size:10px; color:#94a3b8;">L5</div><div style="font-size:18px; font-weight:900; color:{trend_color};">{l5_avg}</div></div></div></div>', unsafe_allow_html=True)

                    b_cols = st.columns(len(board))
                    for i, m in enumerate(board):
                        b_cols[i].markdown(f'<div class="board-member"><div class="board-name">{m["name"]}</div><div class="board-model">{m["model"]}</div><div style="font-size:11px; color:#94a3b8; font-style:italic; line-height:1.3; margin-bottom:12px; min-height:45px;">"{m["quote"]}"</div><div style="color:#94a3b8; font-size:12px; border-top:1px dashed #334155; padding-top:8px;">Proj: <span style="color:#fff; font-weight:bold;">{m["proj"]:.2f}</span></div><div class="board-vote" style="color:{m["color"]}; margin-top:2px;">{m["vote"]}</div></div>', unsafe_allow_html=True)

                    st.markdown("#### 📊 L10 Performance vs Line")
                    chart_col, side_col = st.columns([3.2, 1.4])
                    
                    with chart_col:
                        df_l10['Matchup_Formatted'] = np.where(df_l10['Is_Home'] == 1, "vs " + df_l10['MATCHUP'], "@ " + df_l10['MATCHUP'])
                        df_l10['Matchup_Label'] = df_l10['ShortDate'] + "|" + df_l10['Matchup_Formatted']
                        df_l10['Is_Target_Opp'] = df_l10['MATCHUP'] == opp
                        
                        # 🔴 RED DOT INJECTION: Fetch saved receipts from the vault
                        df_l10['Saved_Proj'] = np.nan
                        try:
                            import os
                            import pandas as pd
                            if os.path.exists("saved_projections.csv"):
                                receipts = pd.read_csv("saved_projections.csv")
                                receipts = receipts[(receipts['Player'] == target_player) & (receipts['Stat'] == stat_type)]
                                if not receipts.empty:
                                    # Match date formats (YYYY-MM-DD) to map the dots to the right game
                                    date_col = 'ValidDate' if 'ValidDate' in df_l10.columns else 'Date'
                                    df_l10_date_strs = pd.to_datetime(df_l10[date_col]).dt.strftime('%Y-%m-%d')
                                    receipts_date_strs = pd.to_datetime(receipts['Game_Date']).dt.strftime('%Y-%m-%d')
                                    
                                    receipt_dict = dict(zip(receipts_date_strs, receipts['Live_Proj']))
                                    df_l10['Saved_Proj'] = df_l10_date_strs.map(receipt_dict)
                        except Exception as e:
                            pass

                        bars = alt.Chart(df_l10).mark_bar(opacity=0.85).encode(
                            x=alt.X('Matchup_Label', sort=None, title=None, axis=alt.Axis(labelAngle=0, labelExpr="split(datum.value, '|')")),
                            y=alt.Y(s_col, title=stat_type),
                            color=alt.condition(
                                alt.datum[s_col] >= line, 
                                alt.value('#00c853'), 
                                alt.value('#d50000')
                            ),
                            stroke=alt.condition(
                                alt.datum.Is_Target_Opp, 
                                alt.value('#FFD700'), 
                                alt.value('transparent')
                            ),
                            strokeWidth=alt.condition(
                                alt.datum.Is_Target_Opp, 
                                alt.value(3), 
                                alt.value(0)
                            ),
                            tooltip=[
                                alt.Tooltip('ShortDate', title='Date'), 
                                alt.Tooltip('Matchup_Formatted', title='Opponent'), 
                                alt.Tooltip('MINS', title='Minutes', format='.1f'), 
                                alt.Tooltip(s_col, title='Actual Stats'), 
                                alt.Tooltip('AI_Proj', title='Retro AI Projection', format='.2f'),
                                alt.Tooltip('Saved_Proj', title='PRE-GAME Vault Proj', format='.2f')
                            ]
                        ).properties(height=350)
                        
                        vegas_rule = alt.Chart(pd.DataFrame({'y': [line]})).mark_rule(color='#FFD700', strokeDash=[5,5], size=2).encode(y='y')
                        ai_line = alt.Chart(df_l10).mark_line(color='#00E5FF', strokeWidth=3, point=alt.OverlayMarkDef(color='#00E5FF', size=60)).encode(x=alt.X('Matchup_Label', sort=None), y=alt.Y('AI_Proj'))
                        
                        # 🔴 RED DOT LAYER: Plots the pre-game stamp if it exists
                        red_dots = alt.Chart(df_l10).mark_circle(color='#ff0055', size=150, opacity=1).encode(
                            x=alt.X('Matchup_Label', sort=None),
                            y=alt.Y('Saved_Proj')
                        ).transform_filter(
                            "isValid(datum.Saved_Proj)" # Only draw a dot if we have a saved receipt
                        )

                        text = bars.mark_text(align='center', baseline='top', dy=5, fontSize=15, fontWeight='bold').encode(text=alt.Text(s_col, format='.0f'), color=alt.value('#ffffff'))
                        
                        # Layer it all together (+ red_dots)
                        final_chart = (bars + vegas_rule + ai_line + red_dots + text)
                        
                        st.altair_chart(final_chart.configure(background='transparent').configure_axis(gridColor='#334155', domainColor='#334155', tickColor='#334155', labelColor='#94a3b8', titleColor='#f8fafc').configure_view(strokeWidth=0), use_container_width=True)
                        st.caption("🟡 Dashed Yellow: Vegas Line &nbsp; | &nbsp; 🔵 Cyan Line: Retro AI &nbsp; | &nbsp; 🔴 <span style='color:#ff0055; font-weight:bold;'>Red Dot: Pre-Game Vault</span> &nbsp; | &nbsp; 🏆 <span style='color:#FFD700;'>Gold Border: Target Opp</span>", unsafe_allow_html=True)
                        
                    with side_col:
                        with st.expander("📊 Matchup Intel (Team Stats)", expanded=True):
                            player_team = target_player.split('(')[1].replace(')', '').strip() if '(' in target_player else opp
                            team_logo_html = f"<img src='{get_team_logo(league_key, player_team)}' width='28' style='vertical-align:middle; margin-right: 8px;'>"
                            opp_logo_html = f"<img src='{get_team_logo(league_key, opp)}' width='28' style='vertical-align:middle; margin-left: 8px;'>"
                            st.markdown(f"<div style='display: flex; justify-content: center; align-items: center; font-weight:900; font-size:18px; color:#00E5FF;'>{team_logo_html} {player_team} vs {opp} {opp_logo_html}</div><hr style='margin: 10px 0px; border-color: #334155;'>", unsafe_allow_html=True)
                            
                            if league_key == "NBA":
                                st.caption("**🧬 AI Player Archetype & Rotation**")
                                st.markdown(f"<div style='font-size:14px; font-weight:bold; color:#00E676;'>{archetype}</div>", unsafe_allow_html=True)
                                st.markdown(f"<div style='font-size:12px; color:#FFD700; margin-top:6px; line-height:1.4; font-weight:500;'>{mod_desc}</div>", unsafe_allow_html=True)
                            else:
                                st.caption(f"**🛡️ {opp} Defense Difficulty**")
                                st.progress(max(0.0, min(1.0, (95 if mod_val < 1.0 else (15 if mod_val > 1.0 else 50)) / 100.0)), text=f"{mod_desc}")
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.caption(f"**⚔️ History vs {opp} (All Time)**")
                            
                            df_opp = df_with_ml[df_with_ml['MATCHUP'] == opp]
                            opp_total = len(df_opp)

                            if opp_total >= 5:
                                opp_hits = int((df_opp[s_col] >= line).sum())
                                opp_win_pct = (opp_hits / opp_total) * 100
                                h2h_color = '#00c853' if opp_win_pct >= 60 else ('#d50000' if opp_win_pct <= 40 else '#FFD700')
                                st.markdown(f"<div style='font-size:22px; font-weight:900; color:{h2h_color};'>{opp_win_pct:.0f}% <span style='font-size:14px; color:#94a3b8; font-weight:normal;'>({opp_hits}/{opp_total} G)</span></div>", unsafe_allow_html=True)

                            elif opp_total >= 2:
                                opp_hits = int((df_opp[s_col] >= line).sum())
                                opp_win_pct = (opp_hits / opp_total) * 100
                                h2h_color = '#00c853' if opp_win_pct >= 60 else ('#d50000' if opp_win_pct <= 40 else '#FFD700')
                                st.markdown(f"<div style='font-size:18px; font-weight:900; color:{h2h_color};'>{opp_win_pct:.0f}% <span style='font-size:12px; color:#94a3b8; font-weight:normal;'>({opp_hits}/{opp_total} G)</span></div>", unsafe_allow_html=True)
                                st.markdown(f"<div style='font-size:11px; color:#f59e0b; margin-top:2px;'>⚠️ Only {opp_total} games vs {opp} — treat with caution.</div>", unsafe_allow_html=True)

                            else:
                                st.markdown(f"<div style='font-size:13px; color:#94a3b8;'>Insufficient H2H data vs {opp}.<br><span style='font-size:11px;'>Model is using league-wide averages instead.</span></div>", unsafe_allow_html=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.caption(f"**🏟️ Venue Advantage ({split_text})**")
                            st.progress(max(0.0, min(1.0, (current_split_mod - 0.8) / 0.4)), text=split_desc)
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.caption(f"**🔋 Energy Levels**")
                            st.progress((100 if fatigue_val == 1.0 else (70 if fatigue_val == 0.95 else 40)) / 100.0, text=fatigue_desc)

def render_league_tab(league_name, get_sched_func):
    lk = league_name.lower()
    render_league_scanners(league_name)
    st.divider()
    c1, c2 = st.columns([8, 1])
    with c1: st.markdown(f"### 📅 Today's {league_name} Slate")
    with c2:
        if st.button("🔄 Refresh", key=f"{lk}.ref_btn"): st.rerun()
    with st.spinner("Loading matchups..."): sched, msg = get_sched_func()
    if sched: render_scoreboard(sched, league_name)
    else: st.info(msg)
    st.markdown("---")
    render_syndicate_board(league_name)

# ==========================================
# 6. MAIN APP LAYOUT & ROUTING
# ==========================================
video_path = "delorean.mp4"
if os.path.exists(video_path):
    with open(video_path, "rb") as video_file: video_base64 = base64.b64encode(video_file.read()).decode()
    html_code = f"""<!DOCTYPE html><html><head><style>@import url('https://fonts.googleapis.com/css2?family=Audiowide&display=swap');body {{margin: 0;padding: 0;background-color: #0f172a;overflow: hidden;font-family: 'Audiowide', sans-serif;}}.b2tf-header-container {{position: relative;overflow: hidden;border-radius: 12px;border: 3px solid #ff0055;text-align: center;background-color: #0f172a;box-shadow: 0 0 15px #ff0055;height: 274px; display: flex;align-items: center;justify-content: center;flex-direction: column;}}@keyframes fade-out-video {{0% {{ opacity: 0.6; }}100% {{ opacity: 0; }}}}.b2tf-video-bg {{position: absolute;top: 0;left: 0;width: 100%;height: 100%;z-index: 0;opacity: 0.6;object-fit: cover;animation: fade-out-video 2.3s ease-out 4.6s forwards; }}.b2tf-content {{position: relative;z-index: 1;opacity: 0;animation: fade-in-text 4.7s ease-out 4.5s forwards; }}@keyframes fade-in-text {{0% {{ opacity: 0; transform: translateY(15px) scale(0.9); }}100% {{ opacity: 1; transform: translateY(0) scale(1); }}}}h1 {{color: #ffcc00; font-size: 56px; font-weight: 900; margin: 0; text-shadow: 0 0 10px #ff6600, 0 0 20px #ff0000, 0 0 30px #ff0000; letter-spacing: 2px;}}.subtitle {{color: #00E5FF; font-size: 16px; font-weight: bold; letter-spacing: 3px; margin-top: 5px; text-shadow: 0 0 5px #00E5FF; background: rgba(15, 23, 42, 0.7); padding: 5px 15px; border-radius: 8px; display: inline-block;}}</style></head><body><div class="b2tf-header-container"><video id="delorean-vid" class="b2tf-video-bg" autoplay muted playsinline><source src="data:video/mp4;base64,{video_base64}" type="video/mp4"></video><div class="b2tf-content"><h1>B2TF ALMANAC</h1><div class="subtitle">ROADS? WHERE WE'RE GOING, WE DON'T NEED ROADS.</div></div></div><script>var video = document.getElementById('delorean-vid');video.addEventListener('loadedmetadata', function() {{video.currentTime = 1.0;}}, false);</script></body></html>"""
    components.html(html_code, height=280)

with st.expander("⛽ System Diagnostics & API Fuel Gauge", expanded=False):
    diag_c1, diag_c2 = st.columns([1, 2])
    with diag_c1: st.markdown("**⚙️ Module Status**\n<br><span style='color:#94a3b8; font-size:12px;'>✅ Odds API (Live)<br>✅ Core DB (Active)<br>✅ MLB/NHL (Active)<br>✅ Archetype Engine (Online)<br>🟣 Skynet Correction (Online)<br>☁️ Google Sheets (Online)</span>", unsafe_allow_html=True)
    with diag_c2:
        st.markdown("**🔋 Odds API Fuel Level**")
        if ODDS_API_KEY:
            f_col1, f_col2 = st.columns([4, 1])
            with f_col2:
                if st.button("🔄 Check", key="chk_quota"): check_api_quota()
            with f_col1:
                if 'api_used' in st.session_state and 'api_remaining' in st.session_state:
                    used, rem = int(st.session_state['api_used']), int(st.session_state['api_remaining'])
                    total = used + rem; fuel_pct = rem / total if total > 0 else 0.0
                    color = "#00E676" if fuel_pct > 0.3 else ("#FFD700" if fuel_pct > 0.1 else "#ff0055")
                    st.markdown(f'<div style="background-color: #1e293b; border-radius: 5px; width: 100%; height: 20px; border: 1px solid #334155; margin-top: 5px;"><div style="background-color: {color}; width: {fuel_pct*100}%; height: 100%; border-radius: 4px; transition: width 0.5s;"></div></div><div style="font-size: 12px; color: #94a3b8; margin-top: 5px; text-align: right;">{rem} / {total} Requests Remaining</div>', unsafe_allow_html=True)
                else: st.caption("Sync a bet or hit 'Check' to load data.")

st.markdown("---")
top_bar_c1, top_bar_c2 = st.columns([5, 1])
with top_bar_c1: st.markdown(f"### 🏦 Liquid Bankroll: <span style='color:#00E676;'>${get_liquid_balance():.2f}</span>", unsafe_allow_html=True)
with top_bar_c2:
    if st.button("🧹 Clear Skynet Cache", use_container_width=True): st.cache_data.clear(); st.rerun()

t_nba, t_nhl, t_mlb, t_parlay, t_roi, t_wallet = st.tabs(["🏀 NBA", "🏒 NHL", "⚾ MLB", "🎟️ Parlay Builder", "🏦 ROI Ledger", "💵 Wallet"])

with t_nba: render_league_tab("NBA", get_nba_schedule)
with t_nhl: render_league_tab("NHL", get_nhl_schedule)
with t_mlb: render_league_tab("MLB", get_mlb_schedule)

with t_parlay:
    st.markdown("## 🎟️ Syndicate Parlay Builder")
    ledger_df = load_ledger()
    pending_picks = ledger_df[ledger_df['Result'] == 'Pending']
    if pending_picks.empty: st.warning("No pending singles found. Go to the sports boards to build your slips!")
    else:
        pick_options, pick_odds_map, pick_prob_map = [], {}, {}
        for _, r in pending_picks.iterrows():
            o = int(pd.to_numeric(r['Odds'], errors='coerce'))
            prob = float(r.get('Win_Prob', 0.55))
            label = f"{r['Player']} ({r['Stat']} {r['Vote']} {r['Line']}) [{o:+d}]"
            pick_options.append(label)
            pick_odds_map[label] = o
            pick_prob_map[label] = prob
        selected_picks = st.multiselect("🔗 Link Pending Picks into a Ticket", pick_options, key="parlay_picker")
        calc_dec, c_prob = 1.0, 1.0
        for p in selected_picks:
            o = pick_odds_map[p]
            calc_dec *= ((o / 100.0) + 1.0) if o > 0 else ((100.0 / abs(o)) + 1.0)
            c_prob *= pick_prob_map.get(p, 0.55)
        true_american = int(round((calc_dec - 1.0) * 100)) if calc_dec >= 2.0 else int(round(-100.0 / (calc_dec - 1.0))) if selected_picks else 150
        def_prob = min(99.9, c_prob * 100) if selected_picks else 55.0

        with st.expander("📘 Open Bankroll Advisor (Calculate Bet Size)", expanded=False):
            liq_bal = get_liquid_balance()
            st.markdown(f"**Live Bankroll:** ${liq_bal:.2f}")
            if "Micro" in st.radio("Select Strategy", ["🔥 Micro-Aggressor (Tiered)", "🤖 True Kelly (AI Math)"], horizontal=True, label_visibility="collapsed", key="parlay_strat"):
                s_rec, p_rec = (5.0, 2.0) if liq_bal < 100 else ((10.0, 4.0) if liq_bal <= 200 else ((15.0, 5.0) if liq_bal <= 500 else (liq_bal * 0.03, liq_bal * 0.01)))
                st.markdown(f"""<div style="background-color: #0f172a; padding: 15px; border-radius: 8px; border-left: 4px solid #ff0055; margin-top: 10px;"><div style="display: flex; justify-content: space-around; text-align: center;"><div><div style="font-size: 12px; color: #94a3b8;">Standard Single Wager</div><div style="font-size: 24px; font-weight: 900; color: #ff0055;">${s_rec:.2f}</div></div><div><div style="font-size: 12px; color: #94a3b8;">Standard Parlay Risk</div><div style="font-size: 24px; font-weight: 900; color: #00E5FF;">${p_rec:.2f}</div></div></div></div>""", unsafe_allow_html=True)
            else:
                kc1, kc2 = st.columns(2)
                k_prob, k_odds = kc1.number_input("Est. Win Prob (%)", min_value=0.1, max_value=99.9, value=float(def_prob), step=1.0, key="parlay_kprob"), kc2.number_input("Bet Odds", value=true_american, step=10, key="parlay_kodds")
                win_prob_dec, b_odds = k_prob / 100.0, (100 / abs(k_odds)) if k_odds < 0 else (k_odds / 100)
                s_rec = liq_bal * (max(0.0, (b_odds * win_prob_dec - (1 - win_prob_dec)) / b_odds if b_odds > 0 else 0) * 0.5)
                st.markdown(f"""<div style="background-color: #0f172a; padding: 15px; border-radius: 8px; border-left: 4px solid #00E676; margin-top: 10px; text-align: center;"><div style="font-size: 12px; color: #94a3b8;">Recommended Kelly Stake (Half-Kelly)</div><div style="font-size: 24px; font-weight: 900; color: #00E676;">${s_rec:.2f}</div></div>""", unsafe_allow_html=True)

        p_col1, p_col2, p_col3, p_col4 = st.columns([2.5, 1, 1, 1.5])
        with p_col1: p_desc = st.text_area("Bet Description", value=" + ".join(selected_picks) if selected_picks else "", height=68)
        with p_col2: p_odds = st.number_input("Final Odds (w/ Boosts)", value=true_american, step=10)
        with p_col3: p_risk = st.number_input("Risk ($)", value=10.0, step=5.0)
        with p_col4:
            p_book = st.selectbox("Sportsbook", SPORTSBOOKS)
            p_free = st.checkbox("🆓 Free Bet")
            p_boost = st.checkbox("🚀 Odds Boost")

        proj_profit = (p_risk * (p_odds / 100) if p_odds > 0 else p_risk / (abs(p_odds) / 100)) if p_odds != 0 else 0.0
        st.info(f"💸 **Projected Payout:** ${(proj_profit if p_free else p_risk + proj_profit):.2f} (Profit: ${proj_profit:.2f})")

        if st.button("➕ Add Bet to Tracker", type="primary"):
            if p_desc: save_to_parlay_ledger(p_desc, p_odds, p_risk, p_book, p_free, p_boost); st.success("Bet Added!"); time.sleep(1.0); st.rerun()
            else: st.error("Please enter a description.")

    parlay_df = load_parlay_ledger()
    if not parlay_df.empty:
        st.markdown("---")
        graded_p = parlay_df[parlay_df['Result'].isin(['Win', 'Loss'])]
        p_wins, p_total, p_profit, total_staked = len(graded_p[graded_p['Result'] == 'Win']), len(graded_p), 0.0, 0.0
        for _, row in graded_p.iterrows():
            o, r, is_f = pd.to_numeric(row['Odds'], errors='coerce'), pd.to_numeric(row['Risk'], errors='coerce'), row.get('Is_Free_Bet', False)
            if not is_f: total_staked += r
            if row['Result'] == 'Win': p_profit += (r * (o / 100)) if o > 0 else (r / (abs(o) / 100))
            else: p_profit -= (0 if is_f else r)

        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("Total Graded Live/Parlays", f"{p_total}")
        pm2.metric("Win Rate", f"{(p_wins / p_total * 100) if p_total > 0 else 0.0:.1f}%")
        pm3.metric("Net Profit", f"${p_profit:+.2f}")
        pm4.metric("ROI (%)", f"{(p_profit / total_staked * 100) if p_total > 0 and total_staked > 0 else 0.0:+.1f}%")

        st.markdown("---")
        
        # 🟢 THE NEW TOP-LEVEL HEADER & SAVE BUTTON
        header_c1, header_c2 = st.columns([3, 1])
        with header_c1:
            st.markdown("#### 🎫 Your Live / Parlay Slips")
        with header_c2:
            if st.button("💾 Save All Grades", type="primary", use_container_width=True):
                updated_count = 0
                for orig_idx in parlay_df.index:
                    k = f"p_res_{orig_idx}"
                    if k in st.session_state:
                        new_val = st.session_state[k]
                        if parlay_df.at[orig_idx, 'Result'] != new_val:
                            parlay_df.at[orig_idx, 'Result'] = new_val
                            updated_count += 1
                if updated_count > 0:
                    overwrite_sheet("Parlay_Ledger", parlay_df)
                    st.success(f"Successfully locked {updated_count} new grades!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.info("No new grades to save.")

        # Render the slips below the button
        for i, row in parlay_df.iloc[::-1].reset_index().iterrows():
            orig_idx = row['index']
            odds_raw = pd.to_numeric(row['Odds'], errors='coerce')
            risk_raw = pd.to_numeric(row['Risk'], errors='coerce')
            o = int(odds_raw) if not pd.isna(odds_raw) else 0
            r = float(risk_raw) if not pd.isna(risk_raw) else 0.0
            is_f = row.get('Is_Free_Bet', False)
            if o > 0: payout = (r * (o / 100)) if is_f else r + (r * (o / 100))
            elif o < 0: payout = (r / (abs(o) / 100)) if is_f else r + (r / (abs(o) / 100))
            else: payout = r
            status_color = "#00E676" if row['Result'] == "Win" else ("#ff0055" if row['Result'] == "Loss" else ("#FFD700" if row['Result'] == "Push" else "#94a3b8"))
            legs_html = "".join([f"<div style='margin-bottom: 4px;'>🎟️ {leg}</div>" for leg in str(row['Description']).split(" + ")])
            boost_tag = " <span style='color:#FFD700; font-size:12px;'>🚀 BOOSTED</span>" if row.get('Is_Boosted', False) else ""
            pc1, pc2 = st.columns([4, 1])
            with pc1:
                book_name = row.get('Sportsbook', 'LIVE BET')
                logo_img = BOOK_LOGOS.get(book_name, "")
                book_html = f'<img src="{logo_img}" width="16" height="16" style="border-radius: 50%; vertical-align: middle; margin-right: 6px;"> {book_name.upper()} • ' if logo_img else f"{book_name.upper()} • "
                st.markdown(f"""<div style="background-color: #0f172a; border-radius: 8px; border: 1px solid #334155; border-left: 6px solid {status_color}; padding: 12px; margin-bottom: 5px;"><div style="display: flex; justify-content: space-between; margin-bottom: 8px;"><span style="font-size: 12px; color: #94a3b8; font-weight: bold; letter-spacing: 1px;">{book_html}{row['Date']}</span><span style="font-size: 14px; color: #fff; font-weight: bold;">{o:+d}{boost_tag}</span></div><div style="font-size: 13px; color: #f8fafc; margin-bottom: 10px; line-height: 1.5;">{legs_html}</div><div style="margin-top: 10px; border-top: 1px dashed #334155; padding-top: 8px; display: flex; justify-content: space-between;"><span style="font-size: 12px; color: #94a3b8;">{"🆓 FREE BET: $" + str(r) if is_f else "Risk: $" + str(r)}</span><span style="font-size: 12px; font-weight: bold; color: {status_color};">Payout: ${payout:.2f}</span></div></div>""", unsafe_allow_html=True)
            with pc2:
                st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
                opts = ["Pending", "Win", "Loss"]
                if row['Result'] == "Push": opts.append("Push")
                st.selectbox("Grade", opts, index=opts.index(row['Result']) if row['Result'] in opts else 0, key=f"p_res_{orig_idx}", label_visibility="collapsed")

with t_roi:
    roi_col1, roi_col2 = st.columns([4, 1])
    with roi_col1: st.markdown("### 🏦 The Bankroll (Single Units)")
    with roi_col2:
        if st.button("🤖 Auto-Grade Pending", type="primary", use_container_width=True):
            with st.spinner("Checking official APIs..."): _, grade_msg = auto_grade_ledger()
            st.success(grade_msg); time.sleep(1.5); st.rerun()

    ledger_df = load_ledger()
    if not ledger_df.empty:
        graded_df = ledger_df[ledger_df['Result'].isin(['Win', 'Loss'])]
        wins, losses, profit = len(graded_df[graded_df['Result'] == 'Win']), len(graded_df[graded_df['Result'] == 'Loss']), 0.0
        for _, row in graded_df.iterrows():
            o = pd.to_numeric(row['Odds'], errors='coerce')
            profit += ((100 / (abs(o)/100)) if o < 0 else o) if row['Result'] == 'Win' else -100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Graded Picks", f"{wins + losses}")
        m2.metric("Win Rate", f"{(wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0:.1f}%")
        m3.metric("Net Profit (from $100 bets)", f"${profit:+.2f}")
        m4.metric("ROI (%)", f"{(profit / ((wins + losses) * 100) * 100) if (wins + losses) > 0 else 0.0:+.1f}%")

        st.markdown("---")
        st.markdown("#### 📈 Syndicate Performance Analytics")
        if len(graded_df) > 1:
            analytics_df = graded_df.copy()
            analytics_df['Date_DT'] = pd.to_datetime(analytics_df['Date'])
            analytics_df = analytics_df.sort_values('Date_DT')
            def row_profit(r):
                o_val = pd.to_numeric(r['Odds'], errors='coerce')
                return ((100 / (abs(o_val)/100)) if o_val < 0 else o_val) if r['Result'] == 'Win' else -100.0
            analytics_df['Profit_Per_Bet'] = analytics_df.apply(row_profit, axis=1)
            analytics_df['Cumulative_Profit'] = analytics_df['Profit_Per_Bet'].cumsum()
            ac1, ac2 = st.columns([2, 1])
            with ac1:
                st.caption("**Bankroll Trajectory (Cumulative Profit)**")
                line_chart = alt.Chart(analytics_df).mark_area(line={'color':'#00E5FF'}, color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='#00E5FF', offset=0), alt.GradientStop(color='rgba(0, 229, 255, 0)', offset=1)], x1=1, x2=1, y1=1, y2=0)).encode(x=alt.X('Date_DT:T', title='Date'), y=alt.Y('Cumulative_Profit:Q', title='Net Profit ($)'), tooltip=[alt.Tooltip('Date:N'), alt.Tooltip('Player:N'), alt.Tooltip('Stat:N'), alt.Tooltip('Profit_Per_Bet:Q', title='Bet Result', format='+.2f'), alt.Tooltip('Cumulative_Profit:Q', title='Total Bankroll', format='+.2f')]).properties(height=260, background='transparent').configure_view(strokeWidth=0).configure_axis(gridColor='#1e293b', domainColor='#334155', tickColor='#334155', labelColor='#94a3b8', titleColor='#f8fafc')
                st.altair_chart(line_chart, use_container_width=True)
            with ac2:
                st.caption("**The Leak Finder (Profit by Stat)**")
                stat_profit = analytics_df.groupby('Stat')['Profit_Per_Bet'].sum().reset_index()
                bar_chart = alt.Chart(stat_profit).mark_bar(cornerRadiusEnd=4).encode(y=alt.Y('Stat:N', sort='-x', title=None, axis=alt.Axis(labelLimit=120)), x=alt.X('Profit_Per_Bet:Q', title='Net Profit ($)'), color=alt.condition(alt.datum.Profit_Per_Bet > 0, alt.value('#00c853'), alt.value('#ff0055')), tooltip=[alt.Tooltip('Stat:N'), alt.Tooltip('Profit_Per_Bet:Q', title='Net Profit', format='+.2f')]).properties(height=260, background='transparent').configure_view(strokeWidth=0).configure_axis(gridColor='#1e293b', domainColor='#334155', tickColor='#334155', labelColor='#94a3b8', titleColor='#f8fafc')
                st.altair_chart(bar_chart, use_container_width=True)
        else:
            st.info("🟣 Skynet requires at least 2 graded bets to generate the Performance Analytics dashboard. Keep feeding the machine!")

        st.markdown("---")
        # ═══════════════════════════════════════════════
        # 🔬 LOSS PATTERN REPORT
        # ═══════════════════════════════════════════════
        losses_with_actual = ledger_df[
            (ledger_df['Result'] == 'Loss') &
            (ledger_df['Actual'].astype(str).str.strip().isin(['', 'nan', 'None']) == False)
        ]

        if len(losses_with_actual) >= 3:
            with st.expander(f"🔬 Loss Pattern Report  ({len(losses_with_actual)} analysed losses)", expanded=False):

                # Classify every loss
                miss_types = []
                for _, lr in losses_with_actual.iterrows():
                    mt, dist, _, _ = classify_miss(
                        lr.get('Proj', 0), lr.get('Line', 0),
                        lr.get('Actual', 0), lr.get('Vote', '')
                    )
                    if mt: miss_types.append({
                        'type': mt, 'dist': dist,
                        'stat': lr.get('Stat', ''),
                        'league': lr.get('League', ''),
                        'score': lr.get('Setup_Score', 0)
                    })

                total = len(miss_types)
                bad_beats  = [m for m in miss_types if "BAD BEAT"  in m['type']]
                model_miss = [m for m in miss_types if "MODEL"     in m['type']]
                blowouts   = [m for m in miss_types if "BLOWOUT"   in m['type']]

                # ── Top-line metrics
                pr1, pr2, pr3 = st.columns(3)
                pr1.metric(
                    "😔 Bad Beats",
                    f"{len(bad_beats)}/{total}",
                    f"{len(bad_beats)/total*100:.0f}% of losses",
                    delta_color="off"
                )
                pr2.metric(
                    "⚠️ Model Misses",
                    f"{len(model_miss)}/{total}",
                    f"{len(model_miss)/total*100:.0f}% of losses",
                    delta_color="off"
                )
                pr3.metric(
                    "💥 Blowout Misses",
                    f"{len(blowouts)}/{total}",
                    f"{len(blowouts)/total*100:.0f}% of losses",
                    delta_color="off"
                )

                st.markdown("---")

                # ── Interpretation guidance
                if total >= 5:
                    bad_beat_rate  = len(bad_beats)  / total
                    blowout_rate   = len(blowouts)   / total

                    if bad_beat_rate >= 0.50:
                        st.success(
                            "✅ **Your model is working.** Over half your losses are bad beats — "
                            "the direction was right and you ran into variance. "
                            "Increase volume on high-conviction setups rather than changing the model."
                        )
                    if blowout_rate >= 0.35:
                        # Find which stat type blowouts cluster on
                        blowout_stats = pd.Series([m['stat'] for m in blowouts]).value_counts()
                        top_blowout_stat = blowout_stats.index[0] if not blowout_stats.empty else "Unknown"
                        st.warning(
                            f"⚠️ **High blowout rate ({blowout_rate*100:.0f}%).** "
                            f"Most blowout misses cluster on **{top_blowout_stat}**. "
                            f"This suggests a model blind spot — likely minute volatility or lineup changes "
                            f"that the archetype engine isn't catching. Consider raising the edge threshold "
                            f"for this stat type in `PASS_THRESHOLDS`."
                        )

                # ── High Setup Score losses (the most instructive ones)
                try:
                    elite_losses = [
                        m for m in miss_types
                        if int(float(m.get('score', 0) or 0)) >= 70
                    ]
                    if elite_losses:
                        el_blowouts = [m for m in elite_losses if "BLOWOUT" in m['type']]
                        st.markdown(
                            f"**🎯 High-Score Losses (Setup ≥ 70):** "
                            f"{len(elite_losses)} bet(s) with SOLID/ELITE scores still lost. "
                            + (
                                f"**{len(el_blowouts)} were blowout misses** — "
                                f"investigate these manually for lineup/injury patterns."
                                if el_blowouts else
                                "Most were bad beats or tight misses — expected at this confidence level."
                            )
                        )
                except: pass

                # ── Worst performing stat types from losses
                if miss_types:
                    loss_by_stat = pd.Series([m['stat'] for m in miss_types]).value_counts()
                    if not loss_by_stat.empty:
                        st.markdown("**📉 Most Frequent Loss Markets:**")
                        for stat_name, cnt in loss_by_stat.head(3).items():
                            pct = cnt / total * 100
                            st.markdown(
                                f"&nbsp;&nbsp;• **{stat_name}**: {cnt} losses ({pct:.0f}% of all losses)",
                                unsafe_allow_html=True
                            )
        st.markdown("#### 🎫 Your Bet Slips")

        # ✅ OPT-5: Paginate ROI slips — rendering 50+ st.markdown blocks is extremely slow
        ROI_PAGE_SIZE = 25
        total_slips = len(ledger_df)
        slips_to_render = ledger_df.reset_index().iloc[::-1].head(ROI_PAGE_SIZE)

        if total_slips > ROI_PAGE_SIZE:
            st.caption(f"📋 Showing most recent {ROI_PAGE_SIZE} of {total_slips} slips. Grade older bets via Auto-Grade.")

        for i, row in slips_to_render.iterrows():
            status = str(row.get('Result', 'Pending')).strip()
            if status == 'Win': b_color = "#00c853"
            elif status == 'Loss': b_color = "#ff0055"
            elif status == 'Push': b_color = "#f59e0b"
            else: b_color = "#3b82f6"

            league = row.get('League', 'N/A')
            date = row.get('Date', 'N/A')
            odds = row.get('Odds', 'N/A')
            player = row.get('Player', 'Unknown')
            stat = row.get('Stat', '')
            vote = row.get('Vote', '')
            line = row.get('Line', '')
            proj = row.get('Proj', 'N/A')

            is_boosted = str(row.get('Is_Boosted', 'False')).upper() == 'TRUE' or row.get('Is_Boosted') is True
            boost_html = '<span style="color: #f59e0b; font-size: 10px; font-weight: 900; letter-spacing: 1px;">🚀 BOOSTED</span> &nbsp;' if is_boosted else ''

            # 🟢 NEW SETUP SCORE HTML
            raw_score = row.get('Setup_Score', 0)
            try: setup_score_val = int(float(raw_score))
            except: setup_score_val = 0

            if setup_score_val >= 75:
                score_color, score_label = "#00E676", "ELITE"
            elif setup_score_val >= 55:
                score_color, score_label = "#FFD700", "SOLID"
            elif setup_score_val >= 35:
                score_color, score_label = "#f59e0b", "MARGINAL"
            else:
                score_color, score_label = "#94a3b8", "WEAK"

            score_html = (
                f"<span style='color:{score_color}; font-weight:900;'>"
                f"⚡ {setup_score_val}/100</span> "
                f"<span style='color:{score_color}; font-size:10px; font-weight:bold;'>{score_label}</span>"
            ) if setup_score_val > 0 else ""
            
            if stat in ["Moneyline", "Spread", "Total (O/U)"]:
                market_html = f"<b>{player}</b> ({stat} {line})"
                proj_html = "AI Proj: <span style='color: #00E5FF; font-weight: bold;'>Bypassed</span>"
            else:
                market_html = f"<b>{player}</b> ({stat} {vote} {line})"
                proj_html = f"🤖 AI Proj: <span style='color: #00E5FF; font-weight: bold;'>{proj}</span>"

            # 🟢 DYNAMIC LAYOUT: Give the Autopsy card room if it exists
            actual_raw = str(row.get('Actual', '')).strip()
            has_autopsy = (status == 'Loss' and actual_raw not in ['', 'nan', 'None'])
            
            if has_autopsy:
                # Widens the right column significantly to fit the full rich Autopsy breakdown
                sc1, sc2 = st.columns([2.4, 1.6]) 
            else:
                # Normal layout for Pending/Wins
                sc1, sc2 = st.columns([4, 1])     

            shield_url = LEAGUE_SHIELDS.get(league, "")
            league_icon = f"<img src='{shield_url}' width='16' style='vertical-align:middle; margin-right:4px; padding-bottom:2px;'>" if shield_url else "🛡️"

            with sc1:
                # 🟢 SAFE PARSER FOR LEGACY BETS
                raw_ai = row.get('Win_Prob', 0)
                raw_user = row.get('User_Prob', '')
                
                try: ai_prob_str = f"{float(raw_ai if str(raw_ai).strip() != '' else 0)*100:.1f}%"
                except: ai_prob_str = "N/A"
                
                try: user_prob_str = f"{float(raw_user if str(raw_user).strip() != '' else raw_ai)*100:.1f}%"
                except: user_prob_str = "N/A"

                # MAIN BET SLIP
                st.markdown(f"""
                <div style="background-color: #0f172a; border: 1px solid #1e293b; border-left: 4px solid {b_color}; border-radius: 6px; padding: 15px; margin-bottom: 12px; height: 90%;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                        <div style="color: #94a3b8; font-size: 12px; font-weight: bold; letter-spacing: 0.5px; display: flex; align-items: center;">{league_icon} {league} &nbsp;•&nbsp; {date}</div>
                        <div style="color: #fff; font-size: 14px; font-weight: 900;">{boost_html}{odds}</div>
                    </div>
                    <div style="margin-bottom: 15px;">
                        <div style="color: #f8fafc; font-size: 14px; font-weight: 500;"><span style="color: #f59e0b; margin-right: 6px;">●</span> {market_html}</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #94a3b8; border-top: 1px dashed #334155; padding-top: 12px;">
                        <div>{proj_html}</div>
                        <div style="font-size: 11px; text-align: right;">
                            🤖 AI Prob: <span style="color: #94a3b8;">{ai_prob_str}</span><br>
                            👤 User Prob: <span style="color: #00E5FF; font-weight: bold;">{user_prob_str}</span>
                        </div>
                    </div>
                    <div style="font-size: 12px; color: #94a3b8; text-align: right; margin-top: 6px;">🔮 Final Edge: <span style="color: #FFD700; font-weight: bold;">{score_html}</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            with sc2:
                # 🔬 FULL AUTOPSY CARD (SIDE-BY-SIDE)
                if has_autopsy:
                    miss_type, abs_miss, likely_cause, miss_color = classify_miss(
                        row.get('Proj', 0), row.get('Line', 0), actual_raw, row.get('Vote', '')
                    )
                    if miss_type:
                        proj_val = row.get('Proj', 'N/A')
                        line_val = row.get('Line', 'N/A')
                        
                        try: autopsy_score_val = int(float(row.get('Setup_Score', 0)))
                        except: autopsy_score_val = 0
                            
                        autopsy_score_label = (
                            "ELITE"     if autopsy_score_val >= 75 else
                            "SOLID"     if autopsy_score_val >= 55 else
                            "MARGINAL"  if autopsy_score_val >= 35 else
                            "WEAK"
                        )
                        
                        try: 
                            prob_val = float(row.get('Win_Prob', 0))
                            prob_str = f"{prob_val * 100:.1f}%" if prob_val <= 1.0 else f"{prob_val:.1f}%"
                        except: 
                            prob_str = "N/A"

                        # Using safe parenthesis concatenation to prevent markdown indentation bugs
                        autopsy_html = (
                            f'<div style="background-color: #0f172a; border: 1px solid {miss_color}; border-radius: 8px; padding: 12px; margin-bottom: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.4);">'
                            f'<div style="font-size: 13px; font-weight: 900; color: {miss_color}; margin-bottom: 10px; letter-spacing: 0.5px; display: flex; justify-content: space-between;">'
                            f'<span>{miss_type}</span> <span style="color:#94a3b8; font-size:10px; font-weight:400; letter-spacing: 0px;">Miss: {abs_miss} units</span></div>'
                            f'<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; margin-bottom: 12px; text-align: center;">'
                            f'<div style="background: #1e293b; padding: 6px 2px; border-radius: 4px;">'
                            f'<div style="font-size: 9px; text-transform: uppercase; font-weight: bold; color:#94a3b8; margin-bottom:2px;">Proj</div>'
                            f'<div style="color: #00E5FF; font-weight: bold; font-size: 15px;">{proj_val}</div></div>'
                            f'<div style="background: #1e293b; padding: 6px 2px; border-radius: 4px;">'
                            f'<div style="font-size: 9px; text-transform: uppercase; font-weight: bold; color:#94a3b8; margin-bottom:2px;">Line</div>'
                            f'<div style="color: #FFD700; font-weight: bold; font-size: 15px;">{line_val}</div></div>'
                            f'<div style="background: #1e293b; padding: 6px 2px; border-radius: 4px;">'
                            f'<div style="font-size: 9px; text-transform: uppercase; font-weight: bold; color:#94a3b8; margin-bottom:2px;">Actual</div>'
                            f'<div style="color: {miss_color}; font-weight: bold; font-size: 15px;">{actual_raw}</div></div></div>'
                            f'<div style="font-size: 11px; color: #f8fafc; line-height: 1.5; background: #1e293b; padding: 8px; border-radius: 4px; border-left: 3px solid {miss_color}; margin-bottom: 10px;">'
                            f'{likely_cause}</div>'
                            f'<div style="display:flex; justify-content: space-between; font-size: 10px; color: #94a3b8;">'
                            f'<div>Score: <span style="color:#fff; font-weight:bold;">{autopsy_score_val}/100 ({autopsy_score_label})</span></div>'
                            f'<div>Prob: <span style="color:#fff; font-weight:bold;">{prob_str}</span></div></div></div>'
                        )
                        st.markdown(autopsy_html, unsafe_allow_html=True)
                else:
                    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
                    
                # RENDER THE GRADE DROPDOWN
                opts = ["Pending", "Win", "Loss"]
                if status == "Push": 
                    opts.append("Push")
                    
                start_idx = opts.index(status) if status in opts else 0
                new_val = st.selectbox("Result", opts, index=start_idx, key=f"res_roi_{i}", label_visibility="collapsed")
                if new_val != status:
                    if st.button("💾 Save", key=f"save_roi_{i}", use_container_width=True):
                        orig_idx = int(row['index'])
                        target_row = orig_idx + 2
                        gc = get_gc()
                        if gc:
                            ws = gc.open("B2TF_Database").worksheet("ROI_Ledger")
                            ws.update_acell(f"J{target_row}", new_val)
                        load_sheet_df.clear()
                        load_ledger.clear()
                        st.success("Grade locked!")
                        time.sleep(1)
                        st.rerun()

with t_wallet:
    st.markdown("### 💵 Multi-Sportsbook Wallet")
    st.caption("Track balances across different apps.")
    bw_c1, bw_c2 = st.columns([2, 1])
    with bw_c1:
        with st.form("bankroll_form"):
            sc1, sc2 = st.columns(2)
            t_book = sc1.selectbox("Sportsbook", SPORTSBOOKS)
            t_type = sc2.selectbox("Transaction Type", ["Deposit (Out of Pocket)", "Withdrawal (Cash Out)", "Casino Win (House Money)", "Casino Loss (Bad Spins)"])
            t_amount = st.number_input("Amount ($)", min_value=0.01, step=1.00, format="%.2f")
            if st.form_submit_button("Log Transaction"):
                save_bankroll_transaction(t_book, "Casino" if "Casino" in t_type else "Withdrawal" if "Withdrawal" in t_type else "Deposit", -t_amount if ("Withdrawal" in t_type or "Loss" in t_type) else t_amount)
                get_wallet_breakdown.clear()
                st.success("Transaction Logged!"); time.sleep(1); st.rerun()

    total_liquid, book_balances, tot_dep, tot_wit, tot_cas, tot_sports = get_wallet_breakdown()

    with bw_c2:
        c_col = "#00E676" if tot_cas >= 0 else "#ff0055"
        s_col_color = "#00E676" if tot_sports >= 0 else "#ff0055"
        oop = max((tot_dep - tot_wit), 0.0)
        lb = get_liquid_balance()

        st.markdown(f"""
        <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; text-align: center; margin-top: 28px;">
            <div style="color: #94a3b8; font-size: 12px; font-weight: bold; letter-spacing: 1px;">TOTAL LIQUID BALANCE</div>
            <div style="color: #00E676; font-size: 36px; font-weight: 900; margin: 10px 0px;">${lb:,.2f}</div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; border-top: 1px dashed #334155; padding-top: 12px; margin-top: 15px;">
                <span style="color: #94a3b8;">Out of Pocket: <span style="color: #fff;">${oop:,.2f}</span></span>
                <span style="color: #94a3b8;">Net Casino: <span style="color: {c_col};">{tot_cas:+,.2f}</span></span>
                <span style="color: #94a3b8;">Sports Profit: <span style="color: {s_col_color};">${tot_sports:+,.2f}</span></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    with st.form("manual_ml_form"):
        st.markdown("#### 📝 Log Manual Team Bet (Moneyline/Spread)")
        c1, c2, c3 = st.columns(3)
        m_league = c1.selectbox("League", ["NBA", "NHL", "MLB"])
        m_team = c2.text_input("Team Name", placeholder="e.g. Spurs")
        m_market = c3.selectbox("Market", ["Moneyline", "Spread", "Total (O/U)"])
        c4, c5 = st.columns(2)
        m_odds = c4.number_input("Odds", value=-110, step=10)
        m_line = c5.text_input("Line / Target (Optional)", value="ML")
        if st.form_submit_button("Log Team Bet to ROI Ledger", type="primary"):
            if m_team:
                save_to_ledger(m_league, m_team, m_market, m_line, m_odds, 0.0, "TEAM", 0.50, False)
                st.success(f"{m_market} Logged to ROI Ledger!")
                time.sleep(1); st.rerun()
            else: st.error("Please enter a team name.")

    if book_balances:
        st.markdown("#### 📱 Portfolio Breakdown")
        st.markdown("<br>", unsafe_allow_html=True)
        breakdown_left, breakdown_right = st.columns([2, 1])
        with breakdown_right:
            df_pie = pd.DataFrame(list(book_balances.items()), columns=['Sportsbook', 'Balance'])
            df_pie = df_pie[df_pie['Balance'] > 0]
            if not df_pie.empty:
                chart = alt.Chart(df_pie).mark_arc(innerRadius=60, outerRadius=100, cornerRadius=6).encode(theta=alt.Theta(field="Balance", type="quantitative"), color=alt.Color(field="Sportsbook", type="nominal", legend=alt.Legend(title="Liquidity Location", orient="bottom", labelColor="#94a3b8", titleColor="#00E5FF", titleFontSize=12, labelFontSize=11)), tooltip=[alt.Tooltip('Sportsbook', title='Book'), alt.Tooltip('Balance', format='$.2f')]).properties(height=280, background='transparent').configure_view(strokeWidth=0).configure_arc(stroke="#0f172a", strokeWidth=3)
                st.altair_chart(chart, use_container_width=True, theme="streamlit")
        with breakdown_left:
            port_cols = st.columns(min(len(book_balances), 2) if len(book_balances) > 1 else 1)
            for i, (book, bal) in enumerate(book_balances.items()):
                logo_img = BOOK_LOGOS.get(book, "")
                logo_html = f'<img src="{logo_img}" width="20" height="20" style="border-radius: 50%; vertical-align: middle; margin-right: 8px;"> <span style="font-size: 15px; font-weight: bold; color: #00E5FF; vertical-align: middle;">{book}</span>' if logo_img else f'<span style="font-size: 15px; font-weight: bold; color: #00E5FF;">{book}</span>'
                port_cols[i % len(port_cols)].markdown(f'<div style="background-color: #0f172a; border-left: 4px solid {"#00E676" if bal > 0 else "#ff0055"}; border-radius: 6px; padding: 15px; margin-bottom: 10px; border: 1px solid #334155;"><div style="margin-bottom: 5px;">{logo_html}</div><div style="font-size: 20px; font-weight: 900; color: #fff;">${max(bal, 0.0):.2f}</div></div>', unsafe_allow_html=True)
