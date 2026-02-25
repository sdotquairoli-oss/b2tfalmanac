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

# --- ML IMPORTS ---
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
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
SPORTSBOOKS = ["FanDuel", "Fanatics", "DraftKings", "BetMGM", "Caesars", "ESPN Bet", "Hard Rock", "Other"]
BOOK_LOGOS = {
    "FanDuel": "https://www.google.com/s2/favicons?domain=fanduel.com&sz=128",
    "DraftKings": "https://www.google.com/s2/favicons?domain=draftkings.com&sz=128",
    "BetMGM": "https://www.google.com/s2/favicons?domain=betmgm.com&sz=128",
    "Caesars": "https://www.google.com/s2/favicons?domain=caesars.com&sz=128",
    "Fanatics": "https://www.google.com/s2/favicons?domain=sportsbook.fanatics.com&sz=128",
    "ESPN Bet": "https://www.google.com/s2/favicons?domain=espnbet.com&sz=128",
    "Hard Rock": "https://www.google.com/s2/favicons?domain=hardrock.bet&sz=128"
}

NBA_FULL_TO_ABBREV = {'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN', 'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE', 'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND', 'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'}
ODDS_MEGA_MAP = {**NBA_FULL_TO_ABBREV, "ANA": "Anaheim Ducks", "BUF": "Sabres", "CGY": "Flames", "CAR": "Hurricanes", "COL": "Avalanche", "CBJ": "Blue Jackets", "EDM": "Oilers", "FLA": "Panthers", "LAK": "Kings", "MTL": "Canadiens", "NSH": "Predators", "NJD": "Devils", "NYI": "Islanders", "NYR": "Rangers", "OTT": "Senators", "PIT": "Penguins", "SJS": "Sharks", "SEA": "Kraken", "STL": "Blues", "TBL": "Lightning", "VAN": "Canucks", "VGK": "Knights", "WPG": "Jets"}
S_MAP = {
    "Points": "PTS",
    "Goals": "G",
    "Assists": "A",
    "Shots on Goal": "SOG",
    "Rebounds": "TRB",
    "PRA (Pts+Reb+Ast)": "PRA",
    "Power Play Points": "PPP",
    "Minutes Played": "MINS",
    "Threes Made": "FG3M",
    "Points + Rebounds": "PR",
    "Points + Assists": "PA",
    "Rebounds + Assists": "RA",
    "Hits": "H",
    "Home Runs": "HR",
    "Total Bases": "TB",
    "Pitcher Strikeouts": "K",
    "Pitcher Earned Runs": "ER",
    "Double Double": "DD",
    "Triple Double": "TD"
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

@st.cache_data(ttl=300)
def load_sheet_df(sheet_name, expected_cols):
    gc = get_gc()
    if not gc: return pd.DataFrame(columns=expected_cols)
    try:
        ws = gc.open("B2TF_Database").worksheet(sheet_name)
        data = ws.get_all_records()
        if not data:
            if ws.row_count == 0 or not ws.row_values(1): ws.append_row(expected_cols)
            return pd.DataFrame(columns=expected_cols)
        return pd.DataFrame(data)
    except: return pd.DataFrame(columns=expected_cols)

def append_to_sheet(sheet_name, row_dict, expected_cols):
    gc = get_gc()
    if not gc: return
    try:
        ws = gc.open("B2TF_Database").worksheet(sheet_name)
        if ws.row_count == 0 or not ws.row_values(1): ws.append_row(expected_cols)
        ws.append_row([row_dict.get(col, "") for col in expected_cols])
        load_sheet_df.clear() 
    except Exception as e: st.error(f"Failed to save to database: {e}")

def overwrite_sheet(sheet_name, df):
    gc = get_gc()
    if not gc: return
    max_retries = 3
    for attempt in range(max_retries):
        try:
            ws = gc.open("B2TF_Database").worksheet(sheet_name)
            clean_df = df.fillna("")
            new_values = [clean_df.columns.values.tolist()] + clean_df.values.tolist()
            
            # 1. Write new data first
            ws.update(values=new_values, range_name='A1')
            
            # 2. Only clear rows BELOW the new data to avoid wipe-then-fail
            last_row = len(new_values)
            total_rows = ws.row_count
            if total_rows > last_row:
                ws.batch_clear([f'A{last_row + 1}:Z{total_rows}'])
            
            load_sheet_df.clear()
            return
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt) # Exponential backoff: 1s, 2s, 4s
            else:
                st.error(f"Failed to update database after {max_retries} attempts: {e}")

def load_ledger(): 
    df = load_sheet_df("ROI_Ledger", ["Date", "League", "Player", "Stat", "Line", "Odds", "Proj", "Vote", "Result", "Win_Prob", "Is_Boosted"])
    if "Is_Boosted" not in df.columns: df["Is_Boosted"] = False
    else: df["Is_Boosted"] = df["Is_Boosted"].apply(lambda x: str(x).strip().upper() == 'TRUE' or x is True)
    return df

def save_to_ledger(league, player, stat, line, odds, proj, vote, win_prob=0.55, is_boosted=False):
    row = {"Date":datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d"), "League": league, "Player": player.split('(')[0].strip(), "Stat": stat, "Line": line, "Odds": odds, "Proj": round(proj, 2), "Vote": vote, "Result": "Pending", "Win_Prob": float(win_prob), "Is_Boosted": is_boosted}
    append_to_sheet("ROI_Ledger", row, ["Date", "League", "Player", "Stat", "Line", "Odds", "Proj", "Vote", "Result", "Win_Prob", "Is_Boosted"])

def load_parlay_ledger():
    df = load_sheet_df("Parlay_Ledger", ["Date", "Description", "Odds", "Risk", "Result", "Sportsbook", "Is_Free_Bet", "Is_Boosted"])
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

def get_wallet_breakdown():
    b_df, p_df = load_bankroll(), load_parlay_ledger()
    book_balances = {book: 0.0 for book in SPORTSBOOKS}
    tot_dep, tot_wit, tot_cas, tot_sports = 0.0, 0.0, 0.0, 0.0
    
    if not b_df.empty:
        b_df['Amount'] = pd.to_numeric(b_df['Amount'], errors='coerce').fillna(0)
        for _, r in b_df.iterrows():
            amt = r['Amount']
            bk = str(r.get('Sportsbook', '')).strip() # 🚨 Strips accidental spaces
            t = str(r.get('Type', ''))
            
            if bk in book_balances: book_balances[bk] += amt
            elif bk: book_balances[bk] = amt  # Dynamically catches new books!
            
            if 'Deposit' in t: tot_dep += amt
            elif 'Withdrawal' in t: tot_wit += abs(amt)
            elif 'Casino' in t: tot_cas += amt

    if not p_df.empty:
        processed_slips = set() 
        for orig_idx, r in p_df.iterrows():
            # 🚨 Added DataFrame index to the ID so identical bets aren't skipped
            slip_id = f"{r.get('Date', '')}_{r.get('Sportsbook', '')}_{r.get('Risk', 0)}_{orig_idx}"
            if slip_id in processed_slips: continue
            processed_slips.add(slip_id)
            
            bk = str(r.get('Sportsbook', '')).strip()
            o = pd.to_numeric(r.get('Odds', 0), errors='coerce')
            risk = pd.to_numeric(r.get('Risk', 0), errors='coerce')
            is_f = r.get('Is_Free_Bet', False)
            res = r.get('Result', 'Pending')
            
            if pd.isna(o) or pd.isna(risk): continue
            
            prof = 0.0
            if res == 'Win': prof = (risk * (o/100)) if o > 0 else (risk / (abs(o)/100))
            elif res in ['Loss', 'Pending']: prof = -(0 if is_f else risk)
            
            if bk in book_balances: book_balances[bk] += prof
            elif bk: book_balances[bk] = prof
            
            tot_sports += prof
            
    # Clean up empty accounts
    book_balances = {k: v for k, v in book_balances.items() if v != 0.0}
    total_liquid = sum(book_balances.values())
    return max(total_liquid, 0.0), book_balances, tot_dep, tot_wit, tot_cas, tot_sports

def get_liquid_balance():
    return get_wallet_breakdown()[0] # The Top Bar now safely calls the Unified Engine!

# ==========================================
# 2. AUTO-GRADER & AI AUTOPSY
# ==========================================
def auto_grade_ledger():
    df = load_ledger()
    if not (df['Result'] == 'Pending').any(): return df, "No pending bets."
    updated = 0
    stats_cache = {} # 🧠 Skynet Local Memory Cache
    
    for idx, r in df[df['Result'] == 'Pending'].iterrows():
        try:
            league, player = r['League'], r['Player']
            cache_key = (league, player)
            
            # Check local memory before burning an API call
            if cache_key not in stats_cache:
                time.sleep(1) # Protect against rate limits
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
            if len(g_row) > 1:
                g_row = g_row[g_row['td'] == bet_date]
            if g_row.empty:
                g_row = stats[stats['td'] == next_date]
            if not g_row.empty:
                val, line_val = g_row.iloc[0][s_col], float(r['Line'])
                if r['Vote'] == "OVER": df.at[idx, 'Result'] = 'Win' if val > line_val else 'Loss' if val < line_val else 'Push'
                elif r['Vote'] == "UNDER": df.at[idx, 'Result'] = 'Win' if val < line_val else 'Loss' if val > line_val else 'Push'
                updated += 1
        except: continue
    
    overwrite_sheet("ROI_Ledger", df)
    return df, f"Graded {updated} bets synced to Cloud!"

@st.cache_data(ttl=3600)
def generate_ai_autopsy(league, player, stat, line, vote, bet_date_str):
    try:
        dt = pd.to_datetime(bet_date_str).date()
        if league == "NBA": df, _, _ = get_nba_stats(player); dc = 'ValidDate'
        elif league == "NHL": df, _, _ = get_nhl_stats(player); dc = 'gameDate'
        else: df, _, _ = get_mlb_stats(player); dc = 'gameDate'
        if df.empty: return "No data."
        
        s_col = S_MAP.get(stat, "PTS")
        
        if league == "NBA":
            if s_col == "A": s_col = "AST"
            if s_col == "PRA" and 'PTS' in df: df['PRA'] = df['PTS'] + df['TRB'] + df['AST']
            if s_col in ["DD", "TD"]:
                tens = (df['PTS'] >= 10).astype(int) + (df['TRB'] >= 10).astype(int) + (df['AST'] >= 10).astype(int) + (df.get('STL', 0) >= 10).astype(int) + (df.get('BLK', 0) >= 10).astype(int)
                df['DD'] = (tens >= 2).astype(int)
                df['TD'] = (tens >= 3).astype(int)
            
        df['td'] = pd.to_datetime(df[dc]).dt.date
        g_df = df[df['td'] == dt]
        if g_df.empty: return "Final box score missing."
        
        act_s, act_m = g_df.iloc[0].get(s_col, 0), g_df.iloc[0].get('MINS', 0)
        past = df[df['td'] < dt]
        if past.empty: past = df 
        avg_s, avg_m = past[s_col].mean(), past['MINS'].mean()
        
        an = []
        if abs(act_s - float(line)) <= 1.5: an.append(f"💔 **Bad Beat:** Missed line by <1.5 ({act_s} vs {line}).")
        if act_m < (avg_m * 0.75): an.append(f"⏱️ **Floor Time Cut:** {act_m:.1f} vs Avg {avg_m:.1f} mins.")
        elif act_m > (avg_m * 1.1): an.append(f"⏱️ **Opportunity:** Saw extra time ({act_m:.1f} mins) but failed to convert.")
        if act_m >= (avg_m * 0.75) and s_col != "MINS":
            pm_a, pm_avg = act_s / max(act_m, 1), avg_s / max(avg_m, 1)
            if pm_a < (pm_avg * 0.7): an.append("🥶 **Cold Night:** Efficiency crashed.")
            elif pm_a > (pm_avg * 1.2) and vote == "UNDER": an.append("🔥 **Hot Hand:** Bust the Under with unusual efficiency.")
        if not an: an.append("🃏 **Standard Variance:** Mins & efficiency normal. Market was too sharp.")
        return "<br><br>".join(an)
    except: return "Failed to parse logs."

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
    if not api_key: return None, None, "API Key missing in secrets.toml", None, None
    m_map = {"Points": "player_points", "Goals": "player_goals", "Assists": "player_assists", "Shots on Goal": "player_shots_on_goal", "Power Play Points": "player_power_play_points", "Rebounds": "player_rebounds", "PRA (Pts+Reb+Ast)": "player_points_rebounds_assists", "Threes Made": "player_threes", "Hits": "batter_hits", "Home Runs": "batter_home_runs", "Pitcher Strikeouts": "pitcher_strikeouts", "Double Double": "player_double_double", "Triple Double": "player_triple_double"}
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
        
        # 🛑 Strict Filtering: Only check events containing the player's team
        if target_team_name:
            events_to_check = [e for e in events_data if target_team_name in e.get('home_team', '') or target_team_name in e.get('away_team', '')]
        
        # If blindly searching (no team string found), limit the damage to max 2 event checks, not 5.
        if not events_to_check: 
            events_to_check = events_data[:2]

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
    except Exception as e: return None, None, f"API Error: {str(e)}", used, rem

@st.cache_data(ttl=300) 
def get_nba_stats(player_label):
    cn = player_label.split("(")[0].strip()
    try:
        from nba_api.stats.static import players
        from nba_api.stats.endpoints import playergamelog
        import unicodedata
        
        def clean_name(name):
            return unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8').lower()
            
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
            except:
                pass
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
            except:
                return 0.0
        
        df['MINS'] = df['MIN'].apply(parse_mins)
        df = df.rename(columns={'REB': 'TRB'})
        
        today = pd.to_datetime(datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d"))
        df['Days_Ago'] = (today - df['ValidDate']).dt.days
        df = df[(df['Days_Ago'] >= 0) & (df['Days_Ago'] <= 1095)]
        df['Weight'] = np.exp(-0.003465 * df['Days_Ago'])
        
        final_cols = [c for c in ['ValidDate', 'ShortDate', 'MATCHUP', 'Is_Home', 'MINS', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG3M', 'Weight'] if c in df.columns]
        return df[final_cols].sort_values('ValidDate').reset_index(drop=True), 200, []
    except:
        return pd.DataFrame(), 500, []
        
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

def get_archetype_defense_modifier(league, opp, archetype):
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
        if opp in ["FLA", "DAL", "CAR", "WPG", "VGK", "LAK"]: return 0.90, "Elite Goalie (-10%)"
        elif opp in ["SJS", "ANA", "CBJ", "CHI", "MTL", "NYI"]: return 1.10, "Swiss Cheese Def (+10%)"
        return 1.00, "Average Def (Neutral)"

def get_fatigue_modifier(rest_status):
    if "B2B" in rest_status: return 0.95, "Tired Legs (-5%)"
    if "3 in 4" in rest_status: return 0.90, "Exhausted (-10%)"
    return 1.00, "Fully Rested"

def estimate_alt_odds(orig_line, orig_odds, new_line, stat_type):
    if orig_line is None or orig_odds is None or orig_line == new_line: return orig_odds
    p_orig = abs(orig_odds)/(abs(orig_odds)+100) if orig_odds < 0 else 100/(orig_odds+100)
    std_est = {"Points": 6.0, "Rebounds": 2.5, "Assists": 2.5, "Threes Made": 1.2, "PRA (Pts+Reb+Ast)": 8.0, "Minutes Played": 5.0, "Hits": 1.0, "Pitcher Strikeouts": 2.0}.get(stat_type, 3.0)
    z_shift = (orig_line - new_line) / std_est
    p_new = max(0.05, min(0.95, p_orig + (z_shift * 0.35)))
    new_odds = int(round((-100*p_new)/(1-p_new))) if p_new > 0.50 else int(round((100*(1-p_new))/p_new))
    return 5 * round(new_odds/5)

# --- ⚡ ML REFACTOR HELPERS ---

def build_models(df_ml, s_col, weights, league):
    if league == "MLB":
        # For MLB, MINS = plate appearances/batters faced
        # Per-opportunity rate makes more sense than per-minute
        mins = df_ml['MINS'].replace(0, 1.0).fillna(1.0)
        df_ml['Per_Min'] = df_ml[s_col].fillna(0) / mins
        df_ml['Per_Min'] = df_ml['Per_Min'].clip(0, 10) # Cap runaway ratios on small samples
    else:
        mins = df_ml['MINS'].replace(0, 1.0).fillna(1.0)
        df_ml['Per_Min'] = df_ml[s_col].fillna(0) / mins
    
    y = df_ml[s_col].fillna(0).values
    X = np.arange(len(df_ml)).reshape(-1, 1)
    
    expected_mins = df_ml['MINS'].tail(5).mean()
    if pd.isna(expected_mins) or expected_mins == 0:
        if league == "MLB": expected_mins = 4.0
        elif league == "NHL": expected_mins = 18.0
        else: expected_mins = 15.0
    else:
        if league == "MLB": expected_mins = np.clip(expected_mins, 2.0, 6.0)
        elif league == "NHL": expected_mins = np.clip(expected_mins, 10.0, 28.0)
        else: expected_mins = np.clip(expected_mins, 5.0, 42.0)
        
    # 1. Ridge Baseline (Shock Absorber for linear trends)
    lr = Ridge(alpha=1.0).fit(X, df_ml['Per_Min'].values, sample_weight=weights)
    trend_proj = lr.predict([[len(X)]])[0] * expected_mins
    
    # 2. Random Forest (Short-term Form)
    df_ml['Roll3'] = df_ml[s_col].rolling(3).mean().fillna(df_ml[s_col].mean()).fillna(0)
    X_rf = df_ml[['Roll3', 'MINS']].fillna(0).values
    rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_rf, y, sample_weight=weights)
    stat_proj = rf.predict([[df_ml['Roll3'].iloc[-1], expected_mins]])[0]
    
    # 3. Gradient Boosting (Variance/Regression Flag)
    s_mean = df_ml[s_col].mean()
    if pd.isna(s_mean): s_mean = 0.0
    df_ml['Dev'] = df_ml[s_col].fillna(0) - s_mean
    X_gb = df_ml[['MINS', 'Dev']].fillna(0).values
    gb = GradientBoostingRegressor(n_estimators=50, random_state=42).fit(X_gb, y, sample_weight=weights)
    con_proj = gb.predict([[expected_mins, trend_proj - s_mean]])[0]

    # 4. HistGradientBoosting (Robust Long-term Baseline)
    df_ml['EWMA'] = df_ml[s_col].ewm(span=5, adjust=False).mean().fillna(s_mean)
    X_hgbr = df_ml[['EWMA', 'MINS']].fillna(0).values
    hgbr = HistGradientBoostingRegressor(max_iter=50, random_state=42).fit(X_hgbr, y, sample_weight=weights)
    base_proj = hgbr.predict([[df_ml['EWMA'].iloc[-1], expected_mins]])[0]
    
    return trend_proj, stat_proj, con_proj, base_proj, lr, rf, gb, hgbr, X, X_rf, X_gb, X_hgbr, expected_mins

def apply_context_mods(df, s_col, league, opp, rest, is_home_current, archetype):
    mod_val, mod_desc = get_archetype_defense_modifier(league, opp, archetype)
    fatigue_val, fatigue_desc = get_fatigue_modifier(rest)
    
    # Safe defaults — always defined no matter what
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
    except:
        home_mod = 1.0
        away_mod = 1.0

    current_split_mod = home_mod if is_home_current == 1 else away_mod
    split_desc = f"+{((current_split_mod-1)*100):.0f}%" if current_split_mod > 1 else f"{((current_split_mod-1)*100):.0f}%"

    return mod_val, mod_desc, fatigue_val, fatigue_desc, current_split_mod, split_text, split_desc, home_mod, away_mod
    
def apply_skynet(raw_vote, stat_type, league):
    if raw_vote == "PASS": return {"mod": 1.0, "msg": "🟣 Skynet: Market is efficient. Pass.", "color": "#94a3b8"}
    try:
        ledger = load_ledger()
        if not ledger.empty and 'Result' in ledger.columns and 'League' in ledger.columns:
            graded = ledger[ledger['Result'].isin(['Win', 'Loss'])]
            # 🚨 BUG FIX: Added strict League filtering so NHL doesn't contaminate MLB
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

# --- ⚡ THE SLEEK COORDINATOR ---

@st.cache_data(show_spinner=False, ttl=300)
def run_ml_board(df, s_col, line, opp, league, rest, is_home_current, stat_type):
    df_ml = df.copy()
    archetype = get_player_archetype(df_ml, league)

    if len(df_ml) < 5: 
        return df_ml, [], 0, "PASS", "#94a3b8", 1.0, "Not enough data", 1.0, "", "", 1.0, "", archetype, "Awaiting Data", "#94a3b8"
    
    weights = df_ml['Weight'].values if 'Weight' in df_ml.columns else np.ones(len(df_ml))
    
    # 1. Build Models (Now league-aware!)
    trend_proj, stat_proj, con_proj, base_proj, lr, rf, gb, hgbr, X, X_rf, X_gb, X_hgbr, expected_mins = build_models(df_ml, s_col, weights, league)
    
    # 2. Context Modifiers
    mod_val, mod_desc, fatigue_val, fatigue_desc, current_split_mod, split_text, split_desc, home_mod, away_mod = apply_context_mods(df_ml, s_col, league, opp, rest, is_home_current, archetype)

    # 3. Consensus Math
    guru_proj = ((trend_proj + stat_proj) / 2) * mod_val * fatigue_val * current_split_mod
    raw_consensus = (trend_proj + stat_proj + con_proj + base_proj + guru_proj) / 5
    def get_raw_vote(p): return "OVER" if p >= line + 0.3 else ("UNDER" if p <= line - 0.3 else "PASS")
    raw_vote = get_raw_vote(raw_consensus)
    
    # 4. Skynet Injection (Now league-aware!)
    skynet_data = apply_skynet(raw_vote, stat_type, league)
    final_consensus = raw_consensus * skynet_data["mod"]
    
    def get_final_vote(p): return ("OVER", "#00c853") if p >= line + 0.3 else (("UNDER", "#d50000") if p <= line - 0.3 else ("PASS", "#94a3b8"))
    f_vote, f_color = get_final_vote(final_consensus)
    
    # 5. Build Historical Plot Baseline
    lr_hist = lr.predict(X) * df_ml['MINS'].replace(0, 1.0).values
    rf_hist = rf.predict(X_rf)
    gb_hist = gb.predict(X_gb)
    hgbr_hist = hgbr.predict(X_hgbr)
    unique_mods = {team: get_archetype_defense_modifier(league, team, archetype)[0] for team in df_ml['MATCHUP'].unique()}
    mods = df_ml['MATCHUP'].map(unique_mods).values
    hist_split_mods = np.where(df_ml['Is_Home'] == 1, home_mod, away_mod)
    guru_hist = ((lr_hist + rf_hist) / 2) * mods * 1.0 * hist_split_mods
    df_ml['AI_Proj'] = ((lr_hist + rf_hist + gb_hist + hgbr_hist + guru_hist) / 5) * skynet_data["mod"]

    # 6. UI Payload
    board = [
        {"name": "⏱️ MIN Maximizer", "model": "Ridge Regression", "proj": trend_proj, "vote": get_raw_vote(trend_proj), "color": get_final_vote(trend_proj)[1], "quote": f"Projects {trend_proj:.1f} by weighting recent mins."},
        {"name": "📊 Statistician", "model": "Random Forest", "proj": stat_proj, "vote": get_raw_vote(stat_proj), "color": get_final_vote(stat_proj)[1], "quote": f"Deep Memory sets stable floor. Trees favor {get_raw_vote(stat_proj)}."},
        {"name": "🃏 Contrarian", "model": "Gradient Boosting", "proj": con_proj, "vote": get_raw_vote(con_proj), "color": get_final_vote(con_proj)[1], "quote": f"Flags variance {'regression' if con_proj < df_ml[s_col].mean() else 'spike'} from season norms."},
        {"name": "🛡️ Baseline", "model": "Hist-Gradient Boosting", "proj": base_proj, "vote": get_raw_vote(base_proj), "color": get_final_vote(base_proj)[1], "quote": "Weighted multi-year mapping using exponentially weighted moving averages."},
        {"name": "🎯 Context Guru", "model": "Radar, Rest, Arena", "proj": guru_proj, "vote": get_raw_vote(guru_proj), "color": get_final_vote(guru_proj)[1], "quote": f"Factors {mod_desc.split('(')[0].strip().replace('🛡️', '').replace('🏃', '').strip()}."}
    ]
    
    return df_ml, board, final_consensus, f_vote, f_color, mod_val, mod_desc, current_split_mod, split_text, split_desc, fatigue_val, fatigue_desc, archetype, skynet_data["msg"], skynet_data["color"]
# ==========================================
# 4. SCANNERS (RADAR)
# ==========================================
@st.cache_data(ttl=3600)
def run_nba_heaters(stat_choice="Points"):
    try:
        from nba_api.stats.endpoints import leagueleaders
        sched, _ = get_nba_schedule()
        if not sched: return None, "No NBA games scheduled today."
        teams_today = [g['home'] for g in sched] + [g['away'] for g in sched]

        api_abbr = {"Points": "PTS", "Rebounds": "REB", "Assists": "AST", "Threes Made": "FG3M"}.get(stat_choice, "PTS")
        s_col = {"Points": "PTS", "Rebounds": "TRB", "Assists": "AST", "Threes Made": "FG3M"}.get(stat_choice, "PTS")

        leaders = leagueleaders.LeagueLeaders(stat_category_abbreviation=api_abbr, per_mode48='PerGame').get_data_frames()[0]
        
        heaters = []
        for _, r in leaders.head(50).iterrows():
            team_abbrev = r['TEAM']
            if team_abbrev in teams_today:
                player_name = r['PLAYER']
                season_val = round(r[api_abbr], 1)
                
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
                    
                    _, _, c_proj, _, _, _, _, _, _, _, _, _, _, _, _ = run_ml_board(
                        df, s_col, float(season_val), opp, "NBA", "Rested (1+ Days)", is_home, stat_choice
                    )
                    ai_proj = round(c_proj, 1)
                
                time.sleep(0.5) 
                
                heaters.append({
                    "Player": player_name,
                    "Team": team_abbrev,
                    "Season Stat": season_val,
                    "AI Proj": ai_proj,
                    "Status": matchup_status
                })

        if not heaters: return None, f"No top 50 {stat_choice} leaders playing tonight."
        return pd.DataFrame(heaters), f"✅ Deep Scan Complete: {stat_choice} Projections loaded."
    except Exception as e: return None, f"API Error: {str(e)}"

@st.cache_data(ttl=3600)
def get_nhl_roster(team_abbr):
    try:
        r = requests.get(f"https://api-web.nhle.com/v1/roster/{team_abbr}/current", timeout=5).json()
        players = []
        for cat in ['forwards', 'defensemen']:
            for p in r.get(cat, []):
                players.append({
                    'Name': f"{p['firstName']['default']} {p['lastName']['default']}",
                    'ID': p['id']
                })
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
        
        return {
            'Player': player_name,
            'Opp': opponent,
            'L5 SOG Avg': round(l5_avg, 1),
            'Recent Log': str(sogs),
            'Rating': rating,
            'Diamond': is_diamond
        }
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
            group = "pitching"
            sort_stat = "strikeOuts"
            s_col = "K"
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
                if team_abbr in teams_today:
                    raw_leaders.append((p, team_abbr))
                    
        # Sort by the chosen stat!
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
                    
                _, _, c_proj, _, _, _, _, _, _, _, _, _, _, _, _ = run_ml_board(
                    df, s_col, 0.5, opp, "MLB", "Rested (1+ Days)", is_home, stat_choice
                )
                ai_proj = round(c_proj, 2)
                
            time.sleep(0.5)
            
            heaters.append({
                "Player": player_name,
                "Team": team_abbr,
                "Season Stat": season_val,
                "AI Proj": ai_proj,
                "Status": matchup_status
            })

        if not heaters: return None, f"No top {stat_choice} leaders playing today."
        return pd.DataFrame(heaters), f"✅ Deep Scan Complete: MLB {stat_choice} Projections loaded."
    except Exception as e: return None, f"API Error: {str(e)}"
# ==========================================
# 5. UI RENDERERS
# ==========================================
def init_state(key, default):
    """Sleek helper to initialize and retrieve session state variables."""
    if key not in st.session_state: st.session_state[key] = default
    return st.session_state[key]
    
def render_scoreboard(sd):
    if not sd: return
    for i in range(0, len(sd), 5):
        cols = st.columns(5)
        for j, g in enumerate(sd[i:i+5]):
            with cols[j]:
                dt = f"{g['away']} <span style='color: #FFD700;'>{g['away_score']}</span> - <span style='color: #FFD700;'>{g['home_score']}</span> {g['home']}" if g['is_live_or_final'] else f"{g['away']} @ {g['home']}"
                st.markdown(f'<div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px; text-align: center; margin-bottom: 10px;"><div style="font-size: 14px; font-weight: bold; color: #fff;">{dt}</div><div style="font-size: 11px; color: #00E5FF; margin-top: 4px;">{g["status"]}</div></div>', unsafe_allow_html=True)

def render_league_scanners(league_name):
    lk = league_name.lower()
    with st.expander(f"📡 Launch {league_name} Skynet Radar", expanded=False):
        
        if league_name == "NBA":
            c1, c2 = st.columns([1, 1.5])
            with c1: scan_stat = st.selectbox("🎯 Target Stat", ["Points", "Rebounds", "Assists", "Threes Made"], key=f"{lk}.scan_stat")
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
        
        # --- ⚡ THE SKYNET FAST-TRACK PIPELINE ---
        if f'{lk}.radar.heaters' in st.session_state:
            df_radar = st.session_state[f'{lk}.radar.heaters']
            
            st.dataframe(
                df_radar, use_container_width=True, hide_index=True,
                column_config={
                    "Player": st.column_config.TextColumn("🔥 Player", width="medium"),
                    "Team": st.column_config.TextColumn("🛡️ Team", width="small"),
                    "Season Stat": st.column_config.NumberColumn("🎯 Season Avg", format="%.1f", width="small"),
                    "AI Proj": st.column_config.NumberColumn("🤖 AI Proj", format="%.1f", width="small"),
                    "Status": st.column_config.TextColumn("⚡ Matchup", width="medium")
                }
            )
            
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
            st.dataframe(
                st.session_state[f'{lk}.radar.bb'], use_container_width=True, hide_index=True,
                column_config={
                    "Team": st.column_config.TextColumn("🛡️ Target Team", width="medium"),
                    "Opp": st.column_config.TextColumn("🎯 Weak Opponent", width="medium"),
                    "Opp Status": st.column_config.TextColumn("🚨 Defense Metric", width="large")
                }
            )

def render_syndicate_board(league_key):
    lk = league_key.lower() # ⚡ The Universal Namespace Prefix
    sport_path = "basketball_nba" if league_key == "NBA" else ("baseball_mlb" if league_key == "MLB" else "icehockey_nhl")
    teams = NBA_TEAMS if league_key == "NBA" else (MLB_TEAMS if league_key == "MLB" else NHL_TEAMS)
    
    # Invisibly pull today's schedule to cross-reference matchups
    sched_func = get_nba_schedule if league_key == "NBA" else (get_mlb_schedule if league_key == "MLB" else get_nhl_schedule)
    sched, _ = sched_func()

    top_c1, top_c2, _ = st.columns([1, 1, 2])
    placeholder_sync = top_c1.empty()
    placeholder_home = top_c2.empty()
    
    with st.container():
        c1, c2, c3, c4 = st.columns([2, 1.5, 1, 1.5])
        with c1: 
            search_query = st.text_input(f"🔍 1. Search Player", placeholder="e.g. Judge, LeBron", key=f"{lk}.search_query")
            player_name = None
            if search_query:
                matches = search_nba_players(search_query) if league_key == "NBA" else (search_mlb_players(search_query) if league_key == "MLB" else search_nhl_players(search_query))
                if matches: player_name = st.selectbox("🎯 2. Select Exact Match", matches, key=f"{lk}.dropdown")
                else: st.caption("No players found.")
                
        # 🕵️‍♂️ SKYNET AUTO-DETECT MATCHUP LOGIC
        auto_opp = None
        auto_is_home = True
        if player_name and sched:
            if "(" in player_name and ")" in player_name:
                team_abbr = player_name.split("(")[1].split(")")[0].strip().upper()
                for g in sched:
                    if g['home'].upper() == team_abbr: auto_opp = g['away'].upper(); auto_is_home = True; break
                    elif g['away'].upper() == team_abbr: auto_opp = g['home'].upper(); auto_is_home = False; break

        # ⚡ STATE OVERRIDE using init_state helper logic
        if player_name and player_name != st.session_state.get(f"{lk}.last_player"):
            st.session_state[f"{lk}.last_player"] = player_name
            if auto_opp and auto_opp in teams:
                st.session_state[f"{lk}.opp"] = auto_opp
                st.session_state[f"{lk}.is_home"] = auto_is_home

        # Clean initialization using your new helper
        init_state(f"{lk}.sync", False)
        init_state(f"{lk}.is_home", True)
        init_state(f"{lk}.opp", teams[0])

        # Render top toggles dynamically based on Session State
        sync = placeholder_sync.toggle("📡 Auto-Sync Vegas Odds", key=f"{lk}.sync")
        is_home_current = 1 if placeholder_home.toggle("🏠 Playing at Home?", key=f"{lk}.is_home") else 0
            
        with c2: 
            opts = ["Points", "Rebounds", "Assists", "Threes Made", "PRA (Pts+Reb+Ast)", "Points + Rebounds", "Points + Assists", "Rebounds + Assists", "Double Double", "Triple Double", "Minutes Played"] if league_key == "NBA" else (["Hits", "Home Runs", "Total Bases", "Pitcher Strikeouts", "Pitcher Earned Runs"] if league_key == "MLB" else ["Points", "Goals", "Assists", "Shots on Goal"])
            stat_type = st.selectbox("Stat", opts, key=f"{lk}.stat")
            live_odds_display = st.empty()

        f_line, f_odds, msg, used, rem = None, None, "", None, None
        if sync and player_name:
            with st.spinner("Syncing Odds..."): f_line, f_odds, msg, used, rem = get_live_line(player_name, stat_type, ODDS_API_KEY, sport_path)
            if used and rem: st.session_state['api_used'], st.session_state['api_remaining'] = int(used), int(rem)
            
            if f_line is not None and f_odds is not None: live_odds_display.markdown(f'<div style="background-color: rgba(0, 230, 118, 0.1); border: 1px solid #00E676; padding: 10px; border-radius: 6px; margin-top: 10px;"><div style="font-size: 11px; font-weight: 900; color: #00E676; letter-spacing: 1px;">📡 LIVE MARKET SYNCED</div><div style="font-size: 16px; font-weight: bold; color: #fff;">{stat_type} O/U {f_line} <span style="color: #94a3b8; font-size: 14px;">({"+"+str(f_odds) if f_odds>0 else f_odds})</span></div></div>', unsafe_allow_html=True)
            elif f_odds is not None: live_odds_display.markdown(f'<div style="background-color: rgba(255, 215, 0, 0.1); border: 1px solid #FFD700; padding: 10px; border-radius: 6px; margin-top: 10px;"><div style="font-size: 11px; font-weight: 900; color: #FFD700; letter-spacing: 1px;">🟡 MARKET PARTIAL SYNC</div><div style="font-size: 16px; font-weight: bold; color: #fff;">{stat_type} Odds <span style="color: #94a3b8; font-size: 14px;">({"+"+str(f_odds) if f_odds>0 else f_odds})</span></div></div>', unsafe_allow_html=True)
            else: live_odds_display.caption(f"🟡 {msg}")

        with c3: 
            start_line = float(f_line) if (sync and f_line is not None) else 0.5
            line = st.number_input("Line", value=start_line, step=0.5, key=f"{lk}.line")
            if sync and f_line is not None and f_odds is not None: st.session_state[f"{lk}.odds"] = estimate_alt_odds(float(f_line), int(f_odds), line, stat_type)
            elif f"{lk}.odds" not in st.session_state: st.session_state[f"{lk}.odds"] = -110
            odds = st.number_input("Odds", step=5, key=f"{lk}.odds")
            is_boosted = st.checkbox("🚀 Odds Boost Applied", key=f"{lk}.boost")
                
        with c4: 
            opp = st.selectbox("Opponent", teams, key=f"{lk}.opp")
            rest = st.selectbox("Fatigue", ["Rested (1+ Days)", "Tired (B2B)", "Exhausted (3 in 4)"], key=f"{lk}.rest")

    btn_c1, btn_c2, _ = st.columns([1, 1, 2])
    with btn_c1: analyze_pressed = st.button(f"🚀 Analyze {league_key} Player", type="primary", use_container_width=True, key=f"{lk}.btn_analyze")
    
    if analyze_pressed and player_name: st.session_state[f"{lk}.target_player"] = player_name
    target_player = st.session_state.get(f"{lk}.target_player")

    if target_player:
        with st.spinner(f"Scouting data for {target_player}..."):
            if league_key == "NBA": df, status_code, _ = get_nba_stats(target_player)
            elif league_key == "MLB": df, status_code, _ = get_mlb_stats(target_player)
            else: df, status_code, _ = get_nhl_stats(target_player)
            
        if status_code == 429: st.error("🚨 **Error 429: Rate Limited.** Please wait 60 seconds.")
        elif status_code == 500: st.warning("🟡 **Server Error.** Try again in a moment.")
        elif not df.empty:
            s_col = S_MAP.get(stat_type, "PTS")
            
            if league_key == "NBA":
                if s_col == "A": s_col = "AST"
                if s_col == "PRA": df['PRA'] = df['PTS'] + df['TRB'] + df['AST']
                if s_col == "PR": df['PR'] = df['PTS'] + df['TRB']
                if s_col == "PA": df['PA'] = df['PTS'] + df['AST']
                if s_col == "RA": df['RA'] = df['TRB'] + df['AST']
                if s_col in ["DD", "TD"]:
                    # ⚡ Skynet counts how many categories hit double digits
                    tens = (df['PTS'] >= 10).astype(int) + (df['TRB'] >= 10).astype(int) + (df['AST'] >= 10).astype(int) + (df.get('STL', pd.Series(0, index=df.index)) >= 10).astype(int) + (df.get('BLK', pd.Series(0, index=df.index)) >= 10).astype(int)
                    df['DD'] = (tens >= 2).astype(int)
                    df['TD'] = (tens >= 3).astype(int)
            
            df_with_ml, board, c_proj, c_vote, c_color, mod_val, mod_desc, current_split_mod, split_text, split_desc, fatigue_val, fatigue_desc, archetype, skynet_msg, skynet_color = run_ml_board(df, s_col, line, opp, league_key, rest, is_home_current, stat_type)
            
            if len(board) == 0: st.warning(f"⚠️ **Insufficient Data:** {target_player} has played fewer than 5 games this season.")
            else:
                # 🧠 Residual Monte Carlo: Calculate the AI's actual historical error margin
                df_with_ml['Residual'] = df_with_ml[s_col] - df_with_ml['AI_Proj']
                residual_std = df_with_ml['Residual'].std()
                
                # Fallback safety net for tiny samples or zero-variance bugs
                if pd.isna(residual_std) or residual_std == 0: 
                    residual_std = df_with_ml[s_col].std()
                    if pd.isna(residual_std) or residual_std == 0: residual_std = 1.0
                
                # 🚨 BUG FIX: Use Poisson distribution for binary/low-count stats to prevent negative numbers
                if stat_type in ['HR', 'Goals', 'RBI', 'R', 'Steals', 'SB', 'Double Double', 'Triple Double']:
                    lam_val = max(0.001, c_proj) # Lambda must be > 0
                    sims = np.random.poisson(lam=lam_val, size=10000)
                else:
                    sims = np.random.normal(loc=c_proj, scale=residual_std, size=10000)
                
                if c_vote == "OVER": win_prob = np.sum(sims > line) / 10000.0
                elif c_vote == "UNDER": win_prob = np.sum(sims < line) / 10000.0
                else: win_prob = 0.50
                
                lock_pressed = False
                with btn_c2:
                    if c_vote != "PASS": lock_pressed = st.button(f"🔒 Lock {league_key} Pick", use_container_width=True, type="primary", key=f"{lk}.lock")
                
                if lock_pressed:
                    save_to_ledger(league_key, target_player, stat_type, line, odds, c_proj, c_vote, win_prob, is_boosted)
                    st.success("Pick locked! It is safely stored in your Google Sheets Database.")
                    
                if odds < 0: implied_prob = abs(odds) / (abs(odds) + 100); profit = 100 / (abs(odds) / 100); risk = 100; b_odds = 100 / abs(odds)
                else: implied_prob = 100 / (odds + 100); profit = odds; risk = 100; b_odds = odds / 100
                
                ev_dollars = (win_prob * profit) - ((1 - win_prob) * risk)
                edge_pct = (win_prob - implied_prob) * 100
                
                # 🧠 Half-Kelly Stake Calculator
                liq_bal = get_liquid_balance()
                kelly_pct = max(0.0, (b_odds * win_prob - (1 - win_prob)) / b_odds) if b_odds > 0 else 0.0
                rec_stake = liq_bal * (kelly_pct * 0.5) # Half-Kelly for aggressive growth with safety
                
                ai_summary_short = f"Projected to {'clear' if c_vote == 'OVER' else ('stay under' if c_vote == 'UNDER' else 'too close to')} {line} with a {win_prob*100:.1f}% probability."
                if league_key == "NBA" and "Exploit" in mod_desc: ai_summary_short += f"<br><span style='color:#FFD700; font-weight:bold;'>🚨 Archetype Exploit vs {opp}</span>"
                elif league_key == "NBA" and "Fade" in mod_desc: ai_summary_short += f"<br><span style='color:#ff0055; font-weight:bold;'>🛑 Archetype Fade vs {opp}</span>"
                ai_summary_short += f"<br><br><span style='color:{skynet_color}; font-weight:bold;'>{skynet_msg}</span>"

                if win_prob >= 0.60 and edge_pct >= 5.0 and c_vote != "PASS":
                    st.markdown(f"""
                    <div style="background: linear-gradient(90deg, #FFD700 0%, #ff8c00 100%); padding: 3px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0px 0px 15px rgba(255, 215, 0, 0.4);">
                        <div style="background-color: #0f172a; padding: 12px; border-radius: 6px; text-align: center;">
                            <span style="font-size: 22px;">🌟</span> <span style="font-size: 18px; font-weight: 900; color: #FFD700; letter-spacing: 2px;">OFFICIAL AI TOP PICK</span> <span style="font-size: 22px;">🌟</span>
                            <div style="font-size: 13px; color: #f8fafc; margin-top: 4px;">Massive Edge Detected! {win_prob*100:.1f}% Win Prob | Recommend risking ${rec_stake:.2f}.</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                sum_c1, sum_c2, sum_c3, sum_c4 = st.columns(4)
                with sum_c1:
                    st.markdown(f"""<div class="verdict-box" style="background-color: {c_color}15; border-color: {c_color}; color: #fff; height: 100%;"><div style="font-size:10px; font-weight:bold; color:{c_color}; letter-spacing: 1px;">AI CONSENSUS</div><div style="font-size:26px; font-weight:900; margin: 4px 0px;">{c_vote}</div><div style="font-size:14px; font-weight:bold; margin-bottom: 6px;">Proj: {c_proj:.2f}</div><div style="font-size:11px; color:#94a3b8; border-top: 1px solid {c_color}50; padding-top: 8px; line-height: 1.3;">{ai_summary_short}</div></div>""", unsafe_allow_html=True)
                with sum_c2:
                    if c_vote == "PASS" or edge_pct <= 0: 
                        st.markdown(f'<div class="verdict-box" style="background-color: #1e293b; border-color: #334155; color: #fff; height: 100%;"><div style="font-size:10px; font-weight:bold; color:#94a3b8; letter-spacing: 1px;">RECOMMENDED RISK</div><div style="font-size:22px; font-weight:900; color:#94a3b8;">$0.00 (PASS)</div><div style="font-size:12px; color:#94a3b8;">Negative EV or too tight.</div></div>', unsafe_allow_html=True)
                    else: 
                        st.markdown(f'<div class="verdict-box" style="background-color: #1e293b; border-color: #00E5FF; color: #fff; height: 100%;"><div style="font-size:10px; font-weight:bold; color:#00E5FF; letter-spacing: 1px;">HALF-KELLY STAKE</div><div style="font-size:26px; font-weight:900; color:#00E5FF; margin: 4px 0px;">${rec_stake:.2f}</div><div style="font-size:12px; color:#94a3b8;">EV: ${ev_dollars:+.2f}/$100 | Edge: {edge_pct:+.1f}%</div></div>', unsafe_allow_html=True)
                with sum_c3:
                    df_l10, df_l5 = df_with_ml.tail(10).reset_index(drop=True), df_with_ml.tail(5)
                    l10_hits, l5_hits = int((df_l10[s_col] > line).sum()), int((df_l5[s_col] > line).sum())
                    hit_color = "#00c853" if l10_hits >= 6 else ("#d50000" if l10_hits <= 4 else "#FFD700")
                    st.markdown(f'<div class="verdict-box" style="background-color: #1e293b; border-color: #334155; color: #fff; height: 100%;"><div style="font-size:10px; font-weight:bold; color:#94a3b8; letter-spacing: 1px;">HIT RATE (OVER {line})</div><div style="font-size:22px; font-weight:900; color:{hit_color};">{l10_hits}/10</div><div style="font-size:13px;">L5: {l5_hits}/5</div></div>', unsafe_allow_html=True)
                with sum_c4:
                    s_avg, l10_avg, l5_avg = round(df[s_col].mean(), 1), round(df_l10[s_col].mean(), 1), round(df_l5[s_col].mean(), 1)
                    trend_color = "#00c853" if l5_avg >= s_avg * 1.1 else ("#d50000" if l5_avg <= s_avg * 0.9 else "#fff")
                    st.markdown(f'<div class="verdict-box" style="background-color: #1e293b; border-color: #334155; color: #fff; height: 100%;"><div style="font-size:10px; font-weight:bold; color:#94a3b8; letter-spacing: 1px; margin-bottom: 2px;">RECENT AVERAGES</div><div style="display: flex; justify-content: space-around; align-items: center; margin-top: 2px;"><div><div style="font-size:10px; color:#94a3b8;">Season</div><div style="font-size:18px; font-weight:900;">{s_avg}</div></div><div><div style="font-size:10px; color:#94a3b8;">L10</div><div style="font-size:18px; font-weight:900;">{l10_avg}</div></div><div><div style="font-size:10px; color:#94a3b8;">L5</div><div style="font-size:18px; font-weight:900; color:{trend_color};">{l5_avg}</div></div></div></div>', unsafe_allow_html=True)
                
                b_cols = st.columns(len(board))
                for i, m in enumerate(board):
                    b_cols[i].markdown(f'<div class="board-member"><div class="board-name">{m["name"]}</div><div class="board-model">{m["model"]}</div><div style="font-size:11px; color:#94a3b8; font-style:italic; line-height:1.3; margin-bottom:12px; min-height:45px;">"{m["quote"]}"</div><div style="color:#94a3b8; font-size:12px; border-top:1px dashed #334155; padding-top:8px;">Proj: <span style="color:#fff; font-weight:bold;">{m["proj"]:.2f}</span></div><div class="board-vote" style="color:{m["color"]}; margin-top:2px;">{m["vote"]}</div></div>', unsafe_allow_html=True)

                st.markdown("#### 📊 L10 Performance vs Line")
                chart_col, side_col = st.columns([3.2, 1.2])
                with chart_col:
                    df_l10['Matchup_Formatted'] = np.where(df_l10['Is_Home'] == 1, "vs " + df_l10['MATCHUP'], "@ " + df_l10['MATCHUP'])
                    df_l10['Matchup_Label'] = df_l10['ShortDate'] + "|" + df_l10['Matchup_Formatted']
                    df_l10['Is_Target_Opp'] = df_l10['MATCHUP'] == opp
                    bars = alt.Chart(df_l10).mark_bar(opacity=0.85).encode(
                        x=alt.X('Matchup_Label', sort=None, title=None, axis=alt.Axis(labelAngle=0, labelExpr="split(datum.value, '|')")),
                        y=alt.Y(s_col, title=stat_type),
                        color=alt.condition(alt.datum[s_col] > line, alt.value('#00c853'), alt.value('#d50000')),
                        stroke=alt.condition(alt.datum.Is_Target_Opp, alt.value('#FFD700'), alt.value('transparent')),
                        strokeWidth=alt.condition(alt.datum.Is_Target_Opp, alt.value(3), alt.value(0)),
                        tooltip=[alt.Tooltip('ShortDate', title='Date'), alt.Tooltip('Matchup_Formatted', title='Opponent'), alt.Tooltip(s_col, title='Actual Stats'), alt.Tooltip('AI_Proj', title='AI Projection', format='.2f')]
                    ).properties(height=350)
                    vegas_rule = alt.Chart(pd.DataFrame({'y': [line]})).mark_rule(color='#FFD700', strokeDash=[5,5], size=2).encode(y='y')
                    ai_line = alt.Chart(df_l10).mark_line(color='#00E5FF', strokeWidth=3, point=alt.OverlayMarkDef(color='#00E5FF', size=60)).encode(x=alt.X('Matchup_Label', sort=None), y=alt.Y('AI_Proj'))
                    text = bars.mark_text(align='center', baseline='top', dy=5, fontSize=15, fontWeight='bold').encode(text=alt.Text(s_col, format='.0f'), color=alt.value('#ffffff'))
                    st.altair_chart((bars + vegas_rule + ai_line + text).configure(background='transparent').configure_axis(gridColor='#334155', domainColor='#334155', tickColor='#334155', labelColor='#94a3b8', titleColor='#f8fafc').configure_view(strokeWidth=0), use_container_width=True)
                    st.caption("🟡 Dashed Yellow Line: Vegas Line &nbsp; | &nbsp; 🔵 Solid Cyan Line: AI Projection &nbsp; | &nbsp; 🏆 <span style='color:#FFD700;'>Gold Border: Games vs Tonight's Opponent</span>", unsafe_allow_html=True)
                with side_col:
                    with st.expander("📊 Matchup Intel (Team Stats)", expanded=True):
                        st.markdown(f"<div style='text-align:center; font-weight:900; font-size:16px; color:#00E5FF;'>{(target_player.split('(')[1].replace(')', '').strip() if '(' in target_player else 'Team')} vs {opp}</div><hr style='margin: 10px 0px; border-color: #334155;'>", unsafe_allow_html=True)
                        if league_key == "NBA":
                            st.caption("**🧬 AI Player Archetype**")
                            st.markdown(f"<div style='font-size:14px; font-weight:bold; color:#00E676;'>{archetype}</div>", unsafe_allow_html=True)
                            if "Exploit" in mod_desc or "Fade" in mod_desc: st.markdown(f"<div style='font-size:12px; color:#FFD700; margin-top:2px; font-style:italic;'>{mod_desc}</div>", unsafe_allow_html=True)
                        else:
                            st.caption(f"**🛡️ {opp} Defense Difficulty**")
                            st.progress(max(0.0, min(1.0, (95 if mod_val < 1.0 else (15 if mod_val > 1.0 else 50)) / 100.0)), text=f"{mod_desc}")
                        st.markdown("<br>", unsafe_allow_html=True); st.caption(f"**⚔️ History vs {opp} (All Time)**")
                        df_opp = df_with_ml[df_with_ml['MATCHUP'] == opp]
                        if not df_opp.empty:
                            opp_hits, opp_total = int((df_opp[s_col] > line).sum()), len(df_opp); opp_win_pct = (opp_hits / opp_total) * 100
                            st.markdown(f"<div style='font-size:22px; font-weight:900; color:{'#00c853' if opp_win_pct >= 60 else ('#d50000' if opp_win_pct <= 40 else '#FFD700')};'>{opp_win_pct:.0f}% <span style='font-size:14px; color:#94a3b8; font-weight:normal;'>({opp_hits}/{opp_total} G)</span></div>", unsafe_allow_html=True)
                        else: st.markdown("<div style='font-size:14px; color:#94a3b8;'>No recent data vs this team.</div>", unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True); st.caption(f"**🏟️ Venue Advantage ({split_text})**")
                        st.progress(max(0.0, min(1.0, (current_split_mod - 0.8) / 0.4)), text=split_desc)
                        st.markdown("<br>", unsafe_allow_html=True); st.caption(f"**🔋 Energy Levels**")
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
    if sched: render_scoreboard(sched)
    else: st.info(msg)
    
    st.markdown("---")
    render_syndicate_board(league_name)
# ==========================================
# 6. MAIN APP LAYOUT & ROUTING (CLEAN!)
# ==========================================

# 1. RENDER HEADER FIRST
video_path = "delorean.mp4"
if os.path.exists(video_path):
    with open(video_path, "rb") as video_file: video_base64 = base64.b64encode(video_file.read()).decode()
    html_code = f"""<!DOCTYPE html><html><head><style>@import url('https://fonts.googleapis.com/css2?family=Audiowide&display=swap');body {{margin: 0;padding: 0;background-color: #0f172a;overflow: hidden;font-family: 'Audiowide', sans-serif;}}.b2tf-header-container {{position: relative;overflow: hidden;border-radius: 12px;border: 3px solid #ff0055;text-align: center;background-color: #0f172a;box-shadow: 0 0 15px #ff0055;height: 274px; display: flex;align-items: center;justify-content: center;flex-direction: column;}}@keyframes fade-out-video {{0% {{ opacity: 0.6; }}100% {{ opacity: 0; }}}}.b2tf-video-bg {{position: absolute;top: 0;left: 0;width: 100%;height: 100%;z-index: 0;opacity: 0.6;object-fit: cover;animation: fade-out-video 2.3s ease-out 4.6s forwards; }}.b2tf-content {{position: relative;z-index: 1;opacity: 0;animation: fade-in-text 4.7s ease-out 4.5s forwards; }}@keyframes fade-in-text {{0% {{ opacity: 0; transform: translateY(15px) scale(0.9); }}100% {{ opacity: 1; transform: translateY(0) scale(1); }}}}h1 {{color: #ffcc00; font-size: 56px; font-weight: 900; margin: 0; text-shadow: 0 0 10px #ff6600, 0 0 20px #ff0000, 0 0 30px #ff0000; letter-spacing: 2px;}}.subtitle {{color: #00E5FF; font-size: 16px; font-weight: bold; letter-spacing: 3px; margin-top: 5px; text-shadow: 0 0 5px #00E5FF; background: rgba(15, 23, 42, 0.7); padding: 5px 15px; border-radius: 8px; display: inline-block;}}</style></head><body><div class="b2tf-header-container"><video id="delorean-vid" class="b2tf-video-bg" autoplay muted playsinline><source src="data:video/mp4;base64,{video_base64}" type="video/mp4"></video><div class="b2tf-content"><h1>B2TF ALMANAC</h1><div class="subtitle">ROADS? WHERE WE'RE GOING, WE DON'T NEED ROADS.</div></div></div><script>var video = document.getElementById('delorean-vid');video.addEventListener('loadedmetadata', function() {{video.currentTime = 1.0;}}, false);</script></body></html>"""
    components.html(html_code, height=280)

# 2. RENDER DIAGNOSTICS SECOND
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

# ==========================================
# 6. MAIN NAVIGATION & TAB ROUTING
# ==========================================
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

        st.markdown("#### 🎫 Your Live / Parlay Slips")
        new_p_results = {}
        for i, row in parlay_df.iloc[::-1].reset_index().iterrows():
            orig_idx = row['index']
            o = int(pd.to_numeric(row['Odds'], errors='coerce') or 0)
            r = float(pd.to_numeric(row['Risk'], errors='coerce') or 0.0)
            is_f = row.get('Is_Free_Bet', False)
            status_color = "#00E676" if row['Result'] == "Win" else ("#ff0055" if row['Result'] == "Loss" else ("#FFD700" if row['Result'] == "Push" else "#94a3b8"))
            legs_html = "".join([f"<div style='margin-bottom: 4px;'>🎟️ {leg}</div>" for leg in str(row['Description']).split(" + ")])
            boost_tag = " <span style='color:#FFD700; font-size:12px;'>🚀 BOOSTED</span>" if row.get('Is_Boosted', False) else ""
            
            pc1, pc2 = st.columns([4, 1])
            with pc1: 
                book_name = row.get('Sportsbook', 'LIVE BET')
                logo_img = BOOK_LOGOS.get(book_name, "")
                book_html = f'<img src="{logo_img}" width="16" height="16" style="border-radius: 50%; vertical-align: middle; margin-right: 6px;"> {book_name.upper()} • ' if logo_img else f"{book_name.upper()} • "
                
                st.markdown(f"""<div style="background-color: #0f172a; border-radius: 8px; border: 1px solid #334155; border-left: 6px solid {status_color}; padding: 12px; margin-bottom: 5px;"><div style="display: flex; justify-content: space-between; margin-bottom: 8px;"><span style="font-size: 12px; color: #94a3b8; font-weight: bold; letter-spacing: 1px;">{book_html}{row['Date']}</span><span style="font-size: 14px; color: #fff; font-weight: bold;">{o:+d}{boost_tag}</span></div><div style="font-size: 13px; color: #f8fafc; margin-bottom: 10px; line-height: 1.5;">{legs_html}</div><div style="margin-top: 10px; border-top: 1px dashed #334155; padding-top: 8px; display: flex; justify-content: space-between;"><span style="font-size: 12px; color: #94a3b8;">{"🆓 FREE BET: $" + str(r) if is_f else "Risk: $" + str(r)}</span><span style="font-size: 12px; font-weight: bold; color: {status_color};">Payout: ${( ((r * (o / 100)) if o > 0 else (r / (abs(o) / 100))) if is_f else r + ((r * (o / 100)) if o > 0 else (r / (abs(o) / 100))) ):.2f}</span></div></div>""", unsafe_allow_html=True)
            with pc2:
                st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
                opts = ["Pending", "Win", "Loss", "Push"]
                new_p_results[orig_idx] = st.selectbox("Grade", opts, index=opts.index(row['Result']) if row['Result'] in opts else 0, key=f"p_res_{orig_idx}", label_visibility="collapsed")
        
        if st.button("💾 Save All Live/Parlay Grades", type="primary", use_container_width=True):
            for orig_idx, res in new_p_results.items(): parlay_df.at[orig_idx, 'Result'] = res
            overwrite_sheet("Parlay_Ledger", parlay_df); st.success("Tracker Updated!"); time.sleep(1); st.rerun()

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
        
        # --- 📈 1. THE ANALYTICS ENGINE INJECTION ---
        st.markdown("---")
        st.markdown("#### 📈 Syndicate Performance Analytics")
        
        if len(graded_df) > 1:
            analytics_df = graded_df.copy()
            analytics_df['Date_DT'] = pd.to_datetime(analytics_df['Date'])
            analytics_df = analytics_df.sort_values('Date_DT')
            
            # Re-calculate exact row-by-row profit for the timeline
            def row_profit(r):
                o_val = pd.to_numeric(r['Odds'], errors='coerce')
                return ((100 / (abs(o_val)/100)) if o_val < 0 else o_val) if r['Result'] == 'Win' else -100.0
                
            analytics_df['Profit_Per_Bet'] = analytics_df.apply(row_profit, axis=1)
            analytics_df['Cumulative_Profit'] = analytics_df['Profit_Per_Bet'].cumsum()
            
            ac1, ac2 = st.columns([2, 1])
            with ac1:
                st.caption("**Bankroll Trajectory (Cumulative Profit)**")
                # 📊 Glowing Neon Area Chart
                line_chart = alt.Chart(analytics_df).mark_area(
                    line={'color':'#00E5FF'},
                    color=alt.Gradient(
                        gradient='linear',
                        stops=[alt.GradientStop(color='#00E5FF', offset=0), alt.GradientStop(color='rgba(0, 229, 255, 0)', offset=1)],
                        x1=1, x2=1, y1=1, y2=0
                    )
                ).encode(
                    x=alt.X('Date_DT:T', title='Date'),
                    y=alt.Y('Cumulative_Profit:Q', title='Net Profit ($)'),
                    tooltip=[alt.Tooltip('Date:N'), alt.Tooltip('Player:N'), alt.Tooltip('Stat:N'), alt.Tooltip('Profit_Per_Bet:Q', title='Bet Result', format='+.2f'), alt.Tooltip('Cumulative_Profit:Q', title='Total Bankroll', format='+.2f')]
                ).properties(height=260, background='transparent').configure_view(strokeWidth=0).configure_axis(gridColor='#1e293b', domainColor='#334155', tickColor='#334155', labelColor='#94a3b8', titleColor='#f8fafc')
                st.altair_chart(line_chart, use_container_width=True)
                
            with ac2:
                st.caption("**The Leak Finder (Profit by Stat)**")
                # 📊 Green/Red Horizontal Bar Chart
                stat_profit = analytics_df.groupby('Stat')['Profit_Per_Bet'].sum().reset_index()
                bar_chart = alt.Chart(stat_profit).mark_bar(cornerRadiusEnd=4).encode(
                    y=alt.Y('Stat:N', sort='-x', title=None, axis=alt.Axis(labelLimit=120)),
                    x=alt.X('Profit_Per_Bet:Q', title='Net Profit ($)'),
                    color=alt.condition(alt.datum.Profit_Per_Bet > 0, alt.value('#00c853'), alt.value('#ff0055')),
                    tooltip=[alt.Tooltip('Stat:N'), alt.Tooltip('Profit_Per_Bet:Q', title='Net Profit', format='+.2f')]
                ).properties(height=260, background='transparent').configure_view(strokeWidth=0).configure_axis(gridColor='#1e293b', domainColor='#334155', tickColor='#334155', labelColor='#94a3b8', titleColor='#f8fafc')
                st.altair_chart(bar_chart, use_container_width=True)
        else:
            st.info("🟣 Skynet requires at least 2 graded bets to generate the Performance Analytics dashboard. Keep feeding the machine!")
        
        st.markdown("---")
        # --- 🎫 2. RESUME BET SLIP RENDERER ---
        st.markdown("#### 🎫 Your Bet Slips")

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
                st.success("Transaction Logged!"); time.sleep(1); st.rerun()
                
    # ⚡ ONE LINE: The Wallet now shares the exact same math as the Top Bar Engine!
    total_liquid, book_balances, tot_dep, tot_wit, tot_cas, tot_sports = get_wallet_breakdown()
        
    with bw_c2:
        st.markdown(f"""
        <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; text-align: center; margin-top: 28px;">
            <div style="color: #94a3b8; font-size: 12px; font-weight: bold; letter-spacing: 1px;">TOTAL LIQUID BALANCE</div>
            <div style="color: #00E676; font-size: 36px; font-weight: 900; margin: 10px 0px;">${get_liquid_balance():.2f}</div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; border-top: 1px dashed #334155; padding-top: 12px; margin-top: 15px;">
                <span style="color: #94a3b8;">Out of Pocket: <span style="color: #fff;">${max((tot_dep - tot_wit), 0.0):.2f}</span></span>
                <span style="color: #94a3b8;">Net Casino: <span style="color: {'#00E676' if tot_cas >= 0 else '#ff0055'};">{tot_cas:+.2f}</span></span>
                <span style="color: #94a3b8;">Sports Profit: <span style="color: {'#00E676' if tot_sports >= 0 else '#ff0055'};">${tot_sports:+.2f}</span></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    if book_balances:
        st.markdown("#### 📱 Portfolio Breakdown")
        st.markdown("<br>", unsafe_allow_html=True)
        
        breakdown_left, breakdown_right = st.columns([2, 1])
        
        with breakdown_right:
            df_pie = pd.DataFrame(list(book_balances.items()), columns=['Sportsbook', 'Balance'])
            df_pie = df_pie[df_pie['Balance'] > 0] 
            
            if not df_pie.empty:
                chart = alt.Chart(df_pie).mark_arc(innerRadius=60, outerRadius=100, cornerRadius=6).encode(
                    theta=alt.Theta(field="Balance", type="quantitative"),
                    color=alt.Color(field="Sportsbook", type="nominal", legend=alt.Legend(title="Liquidity Location", orient="bottom", labelColor="#94a3b8", titleColor="#00E5FF", titleFontSize=12, labelFontSize=11)),
                    tooltip=[alt.Tooltip('Sportsbook', title='Book'), alt.Tooltip('Balance', format='$.2f')]
                ).properties(
                    height=280,
                    background='transparent'
                ).configure_view(strokeWidth=0).configure_arc(stroke="#0f172a", strokeWidth=3)
                
                st.altair_chart(chart, use_container_width=True, theme="streamlit")
        
        with breakdown_left:
            port_cols = st.columns(min(len(book_balances), 2) if len(book_balances) > 1 else 1)
            for i, (book, bal) in enumerate(book_balances.items()):
                logo_img = BOOK_LOGOS.get(book, "")
                logo_html = f'<img src="{logo_img}" width="20" height="20" style="border-radius: 50%; vertical-align: middle; margin-right: 8px;"> <span style="font-size: 15px; font-weight: bold; color: #00E5FF; vertical-align: middle;">{book}</span>' if logo_img else f'<span style="font-size: 15px; font-weight: bold; color: #00E5FF;">{book}</span>'
                port_cols[i % len(port_cols)].markdown(f'<div style="background-color: #0f172a; border-left: 4px solid {"#00E676" if bal > 0 else "#ff0055"}; border-radius: 6px; padding: 15px; margin-bottom: 10px; border: 1px solid #334155;"><div style="margin-bottom: 5px;">{logo_html}</div><div style="font-size: 20px; font-weight: 900; color: #fff;">${max(bal, 0.0):.2f}</div></div>', unsafe_allow_html=True)
