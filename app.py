import streamlit as st
import pandas as pd
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# --- CONFIGURATION ---
st.set_page_config(page_title="B2TF Almanac", layout="wide", page_icon="⚡", initial_sidebar_state="expanded")
BDL_API_KEY = "b148807a-bbf0-45ee-b051-0a6d94a01ff9"

try:
    ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
except:
    ODDS_API_KEY = None

# --- CONSTANTS ---
NBA_TEAMS = sorted(["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"])
NHL_TEAMS = sorted(["ANA", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR", "OTT", "PHI", "PIT", "SJS", "SEA", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WSH", "WPG"])
MLB_TEAMS = sorted(["ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH"])
SPORTSBOOKS = ["FanDuel", "Fanatics", "DraftKings", "BetMGM", "Caesars", "ESPN Bet", "Hard Rock", "Other"]

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
            creds_dict = json.loads(st.secrets["google_credentials"])
            return gspread.service_account_from_dict(creds_dict)
    except Exception as e:
        st.error(f"🚨 Google Sheets Auth Error: {e}")
    return None

def load_sheet_df(sheet_name, expected_cols):
    gc = get_gc()
    if not gc: return pd.DataFrame(columns=expected_cols)
    try:
        sh = gc.open("B2TF_Database")
        ws = sh.worksheet(sheet_name)
        data = ws.get_all_records()
        if not data:
            if ws.row_count == 0 or not ws.row_values(1):
                ws.append_row(expected_cols)
            return pd.DataFrame(columns=expected_cols)
        return pd.DataFrame(data)
    except:
        return pd.DataFrame(columns=expected_cols)

def append_to_sheet(sheet_name, row_dict, expected_cols):
    gc = get_gc()
    if not gc: return
    try:
        sh = gc.open("B2TF_Database")
        ws = sh.worksheet(sheet_name)
        if ws.row_count == 0 or not ws.row_values(1):
            ws.append_row(expected_cols)
        row_values = [row_dict.get(col, "") for col in expected_cols]
        ws.append_row(row_values)
    except Exception as e:
        st.error(f"Failed to save to database: {e}")

def overwrite_sheet(sheet_name, df):
    gc = get_gc()
    if not gc: return
    try:
        sh = gc.open("B2TF_Database")
        ws = sh.worksheet(sheet_name)
        ws.clear()
        clean_df = df.fillna("")
        ws.update(values=[clean_df.columns.values.tolist()] + clean_df.values.tolist())
    except Exception as e:
        st.error(f"Failed to update database: {e}")

def load_ledger():
    cols = ["Date", "League", "Player", "Stat", "Line", "Odds", "Proj", "Vote", "Result", "Win_Prob"]
    return load_sheet_df("ROI_Ledger", cols)

def save_to_ledger(league, player, stat, line, odds, proj, vote, win_prob=0.55):
    cols = ["Date", "League", "Player", "Stat", "Line", "Odds", "Proj", "Vote", "Result", "Win_Prob"]
    row = {"Date": datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d"), "League": league, "Player": player.split('(')[0].strip(), "Stat": stat, "Line": line, "Odds": odds, "Proj": round(proj, 2), "Vote": vote, "Result": "Pending", "Win_Prob": float(win_prob)}
    append_to_sheet("ROI_Ledger", row, cols)

def load_parlay_ledger():
    cols = ["Date", "Description", "Odds", "Risk", "Result", "Sportsbook", "Is_Free_Bet"]
    df = load_sheet_df("Parlay_Ledger", cols)
    if "Is_Free_Bet" not in df.columns: df["Is_Free_Bet"] = False
    return df

def save_to_parlay_ledger(desc, odds, risk, book, is_free):
    cols = ["Date", "Description", "Odds", "Risk", "Result", "Sportsbook", "Is_Free_Bet"]
    row = {"Date": datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d"), "Description": desc, "Odds": int(odds), "Risk": float(risk), "Result": "Pending", "Sportsbook": book, "Is_Free_Bet": is_free}
    append_to_sheet("Parlay_Ledger", row, cols)

def load_bankroll():
    cols = ["Date", "Sportsbook", "Type", "Amount"]
    return load_sheet_df("Bankroll_Ledger", cols)

def save_bankroll_transaction(book, trans_type, amount):
    cols = ["Date", "Sportsbook", "Type", "Amount"]
    row = {"Date": datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d %H:%M"), "Sportsbook": book, "Type": trans_type, "Amount": float(amount)}
    append_to_sheet("Bankroll_Ledger", row, cols)

def get_liquid_balance():
    b_df, p_df, bal = load_bankroll(), load_parlay_ledger(), 0.0
    if not b_df.empty: 
        bal += pd.to_numeric(b_df['Amount'], errors='coerce').sum()
    if not p_df.empty:
        for _, r in p_df.iterrows():
            o = pd.to_numeric(r['Odds'], errors='coerce')
            risk = pd.to_numeric(r['Risk'], errors='coerce')
            is_f = r.get('Is_Free_Bet', False)
            if r['Result'] == 'Win': bal += (risk * (o/100)) if o > 0 else (risk / (abs(o)/100))
            elif r['Result'] in ['Loss', 'Pending']: bal -= (0 if is_f else risk)
    return max(bal, 0.0)

# ==========================================
# AUTO-GRADER & AI AUTOPSY
# ==========================================
def auto_grade_ledger():
    df = load_ledger()
    if not (df['Result'] == 'Pending').any(): return df, "No pending bets."
    updated = 0
    for idx, r in df[df['Result'] == 'Pending'].iterrows():
        try:
            time.sleep(1) 
            if r['League'] == "NBA": stats, _, _ = get_nba_stats(r['Player']); d_col = 'ValidDate'
            elif r['League'] == "NHL": stats, _, _ = get_nhl_stats(r['Player']); d_col = 'gameDate'
            else: stats, _, _ = get_mlb_stats(r['Player']); d_col = 'gameDate'
            if stats.empty: continue
            
            s_map = {"Points": "PTS", "Goals": "G", "Assists": "A", "Shots on Goal": "SOG", "Rebounds": "TRB", "PRA (Pts+Reb+Ast)": "PRA", "Minutes Played": "MINS", "Hits": "H", "Pitcher Strikeouts": "K"}
            s_col = s_map.get(r['Stat'], "PTS")
            if r['League'] == "NBA":
                if s_col == "A": s_col = "AST"
                if s_col == "PRA" and 'PTS' in stats: stats['PRA'] = stats['PTS'] + stats['TRB'] + stats['AST']
            
            stats['td'] = pd.to_datetime(stats[d_col]).dt.date
            g_row = stats[stats['td'] == pd.to_datetime(r['Date']).date()]
            if not g_row.empty:
                val = g_row.iloc[0][s_col]
                line_val = float(r['Line'])
                if r['Vote'] == "OVER": df.at[idx, 'Result'] = 'Win' if val > line_val else 'Loss' if val < line_val else 'Push'
                elif r['Vote'] == "UNDER": df.at[idx, 'Result'] = 'Win' if val < line_val else 'Loss' if val > line_val else 'Push'
                updated += 1
        except: continue
    
    overwrite_sheet("ROI_Ledger", df)
    return df, f"Graded {updated} bets synced to Cloud!"

def generate_ai_autopsy(league, player, stat, line, vote, bet_date_str):
    try:
        dt = pd.to_datetime(bet_date_str).date()
        if league == "NBA": df, _, _ = get_nba_stats(player); dc = 'ValidDate'
        elif league == "NHL": df, _, _ = get_nhl_stats(player); dc = 'gameDate'
        else: df, _, _ = get_mlb_stats(player); dc = 'gameDate'
        if df.empty: return "No data."
        
        s_map = {"Points": "PTS", "Goals": "G", "Assists": "A", "Shots on Goal": "SOG", "Rebounds": "TRB", "PRA (Pts+Reb+Ast)": "PRA", "Minutes Played": "MINS", "Hits": "H", "Pitcher Strikeouts": "K"}
        s_col = s_map.get(stat, "PTS")
        if league == "NBA":
            if s_col == "A": s_col = "AST"
            if s_col == "PRA" and 'PTS' in df: df['PRA'] = df['PTS'] + df['TRB'] + df['AST']
            
        df['td'] = pd.to_datetime(df[dc]).dt.date
        g_df = df[df['td'] == dt]
        if g_df.empty: return "Final box score missing."
        
        act_s, act_m = g_df.iloc[0].get(s_col, 0), g_df.iloc[0].get('MINS', 0)
        past = df[df['td'] < dt]
        if past.empty: past = df 
        avg_s, avg_m = past[s_col].mean(), past['MINS'].mean()
        
        an = []
        if abs(act_s - float(line)) <= 1.5: an.append(f"💔 **Bad Beat:** Missed line by <1.5 ({act_s} vs {line}).")
        if act_m < (avg_m * 0.75): an.append(f"⏱️ **Floor Time Cut:** {act_m:.1f} vs Avg {avg_m:.1f} mins. Likely foul trouble, injury, or blowout.")
        elif act_m > (avg_m * 1.1): an.append(f"⏱️ **Opportunity:** Saw extra time ({act_m:.1f} mins) but failed to convert.")
        if act_m >= (avg_m * 0.75) and s_col != "MINS":
            pm_a, pm_avg = act_s / max(act_m, 1), avg_s / max(avg_m, 1)
            if pm_a < (pm_avg * 0.7): an.append("🥶 **Cold Night:** Efficiency crashed.")
            elif pm_a > (pm_avg * 1.2) and vote == "UNDER": an.append("🔥 **Hot Hand:** Unusually efficient tonight buster the Under.")
        if not an: an.append("🃏 **Standard Variance:** Mins & efficiency normal. Market was too sharp.")
        return "<br><br>".join(an)
    except: return "Failed to parse logs."

# ==========================================
# 🏀 AI PLAYER ARCHETYPE ENGINE & DEFENSE RADAR
# ==========================================
@st.cache_data(ttl=43200) 
def get_live_nba_team_stats():
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        stats = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced')
        df = stats.get_data_frames()[0]
        
        team_mapping = {
            'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN', 'Charlotte Hornets': 'CHA',
            'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE', 'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN',
            'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
            'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA',
            'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK',
            'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
            'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
            'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
        }
        df['TEAM_ABBREV'] = df['TEAM_NAME'].map(team_mapping)
        df_clean = df[['TEAM_ABBREV', 'DEF_RATING', 'PACE']].set_index('TEAM_ABBREV')
        
        df_clean['DEF_RANK'] = df_clean['DEF_RATING'].rank(ascending=True) 
        df_clean['PACE_RANK'] = df_clean['PACE'].rank(ascending=False)
        return df_clean.to_dict('index')
    except Exception as e: return {}

def get_player_archetype(df, league):
    if df.empty or league != "NBA": return "Unknown Profile"
    
    avg_mins = df['MINS'].mean()
    if pd.isna(avg_mins) or avg_mins < 5: avg_mins = 15.0
    
    pts_36 = (df.get('PTS', pd.Series([0])).mean() / avg_mins) * 36
    trb_36 = (df.get('TRB', pd.Series([0])).mean() / avg_mins) * 36
    ast_36 = (df.get('AST', pd.Series([0])).mean() / avg_mins) * 36
    fg3m_36 = (df.get('FG3M', pd.Series([0])).mean() / avg_mins) * 36
    
    clusters = {
        "👑 Primary Playmaker (High USG)": [26.0, 6.0, 9.5, 2.5],
        "🦍 Paint Beast / Rim Runner": [17.0, 13.5, 2.0, 0.1],
        "🧬 Versatile Point-Forward": [21.0, 9.0, 6.0, 1.5],
        "🎯 3&D Wing / Spot-Up Shooter": [15.0, 5.0, 2.0, 3.8],
        "🛡️ Two-Way Connector": [13.0, 4.5, 5.5, 1.5]
    }
    
    player_vec = [[pts_36, trb_36, ast_36, fg3m_36]]
    best_match = "Unknown"
    min_dist = float('inf')
    
    for name, centroid in clusters.items():
        dist = euclidean_distances(player_vec, [centroid])[0][0]
        if dist < min_dist:
            min_dist = dist
            best_match = name
            
    return best_match

def get_archetype_defense_modifier(league, opp, archetype):
    if league == "NBA":
        live_stats = get_live_nba_team_stats()
        if opp in live_stats:
            def_rank = live_stats[opp]['DEF_RANK']
            pace_rank = live_stats[opp]['PACE_RANK']
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
            tough = ["MIN", "BOS", "OKC", "ORL", "MIA", "NYK"]; weak = ["WAS", "DET", "CHA", "SAS", "POR", "ATL", "UTA"]
            if opp in tough: return 0.90, "Elite Defense (-10%)"
            elif opp in weak: return 1.10, "Weak Defense (+10%)"
            return 1.00, "Average Def (Neutral)"
            
    elif league == "MLB":
        tough = ["ATL", "HOU", "LAD", "BAL", "PHI", "NYY"]; weak = ["COL", "OAK", "CHW", "KC", "WSH"]
        if opp in tough: return 0.90, "Elite Pitching (-10%)"
        elif opp in weak: return 1.10, "Weak Pitching (+10%)"
        return 1.00, "Average Pitching (Neutral)"
    else: 
        tough = ["FLA", "DAL", "CAR", "WPG", "VGK", "LAK"]; weak = ["SJS", "ANA", "CBJ", "CHI", "MTL", "NYI"]
        if opp in tough: return 0.90, "Elite Goalie (-10%)"
        elif opp in weak: return 1.10, "Swiss Cheese Def (+10%)"
        return 1.00, "Average Def (Neutral)"

# ==========================================
# ⛽ API FUEL GAUGE CHECKER
# ==========================================
def check_api_quota():
    if not ODDS_API_KEY: return
    try:
        r = requests.get(f"https://api.the-odds-api.com/v4/sports?apiKey={ODDS_API_KEY}", timeout=5)
        u = r.headers.get('x-requests-used')
        rem = r.headers.get('x-requests-remaining')
        if u is not None and rem is not None:
            st.session_state['api_used'] = int(u)
            st.session_state['api_remaining'] = int(rem)
    except: pass

# ==========================================
# API DATA PULLS
# ==========================================
@st.cache_data(ttl=3600)
def run_nba_heaters(target_stat="Points"):
    from nba_api.stats.endpoints import leaguedashplayerstats
    import time
    
    col_map = {
        "Points": "PTS", "Rebounds": "REB", "Assists": "AST", 
        "Threes Made": "FG3M", "PRA (Pts+Reb+Ast)": "PRA", 
        "Points + Rebounds": "PR", "Points + Assists": "PA", "Rebounds + Assists": "RA"
    }
    sc = col_map.get(target_stat, "PTS")
    
    try:
        # Pull overall season averages and Last 5 Games averages for the entire league
        season_stats = leaguedashplayerstats.LeagueDashPlayerStats(per_mode_detailed='PerGame').get_data_frames()[0]
        time.sleep(0.5) # Prevent rate-limiting
        l5_stats = leaguedashplayerstats.LeagueDashPlayerStats(per_mode_detailed='PerGame', last_n_games=5).get_data_frames()[0]
        
        # Calculate Combo Stats (PRA, etc.) since the API doesn't provide them natively
        for df_temp in [season_stats, l5_stats]:
            df_temp['PRA'] = df_temp['PTS'] + df_temp['REB'] + df_temp['AST']
            df_temp['PR'] = df_temp['PTS'] + df_temp['REB']
            df_temp['PA'] = df_temp['PTS'] + df_temp['AST']
            df_temp['RA'] = df_temp['REB'] + df_temp['AST']
            
        # Merge the two databases together
        merged = pd.merge(l5_stats[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', sc, 'MIN']], 
                          season_stats[['PLAYER_ID', sc]], 
                          on='PLAYER_ID', suffixes=('_L5', '_Season'))
                          
        # Filter for actual rotational players (must average at least 15 mins in their Last 5)
        merged = merged[merged['MIN'] >= 15]
        
        # Pull today's active schedule to filter out players who aren't playing tonight
        schedule_data, _ = get_nba_schedule()
        if not schedule_data: return None, "No games scheduled today."
        
        teams_today = []
        for g in schedule_data: teams_today.extend([g['home'], g['away']])
            
        merged = merged[merged['TEAM_ABBREVIATION'].isin(teams_today)]
        if merged.empty: return None, "Scan complete. No rotational players found for today's active slate."
        
        # Calculate the mathematical differential
        merged['Diff'] = merged[f'{sc}_L5'] - merged[f'{sc}_Season']
        
        # Isolate Top 8 Heaters and Top 8 Freezers
        heaters = merged.sort_values('Diff', ascending=False).head(8).copy()
        heaters['Status'] = "🔥 HEATER"
        
        freezers = merged.sort_values('Diff', ascending=True).head(8).copy()
        freezers['Status'] = "❄️ FREEZER"
        
        # Stitch them together and format the table
        final_df = pd.concat([heaters, freezers])
        final_df = final_df[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'Status', f'{sc}_L5', f'{sc}_Season', 'Diff']]
        final_df.columns = ['Player', 'Team', 'Trend', 'L5 Avg', 'Season Avg', '+/- Diff']
        
        final_df['L5 Avg'] = final_df['L5 Avg'].round(1)
        final_df['Season Avg'] = final_df['Season Avg'].round(1)
        final_df['+/- Diff'] = final_df['+/- Diff'].round(1).apply(lambda x: f"+{x}" if x > 0 else str(x))
        
        return final_df, f"✅ Found the top {target_stat} trends for today's active games."
        
    except Exception as e: return None, f"API Error: {str(e)}"
@st.cache_data(ttl=3600)
def search_nba_players(query):
    if not query: return []
    try:
        r = requests.get("https://api.balldontlie.io/v1/players", headers={"Authorization": BDL_API_KEY}, params={"search": query, "per_page": 100}, timeout=5)
        if r.status_code == 200: return [f"{p['first_name']} {p['last_name']} ({p['team']['abbreviation']})" for p in r.json().get('data', []) if p.get('team')]
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
    m_t = {"Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "Seattle Mariners": "SEA", "San Francisco Giants": "SF", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH"}
    try:
        r = requests.get(f"https://statsapi.mlb.com/api/v1/people/search?names={requests.utils.quote(query)}", timeout=5).json()
        if not r.get('people'): return []
        ids = ",".join([str(p['id']) for p in r['people'][:15]])
        br = requests.get(f"https://statsapi.mlb.com/api/v1/people?personIds={ids}&hydrate=currentTeam", timeout=5).json()
        return [f"{p.get('fullName')} ({m_t.get(p.get('currentTeam', {}).get('name', ''), 'FA')})" for p in br.get('people', [])]
    except: pass; return []

@st.cache_data(ttl=60)
def get_nba_schedule():
    try:
        from nba_api.stats.endpoints import scoreboardv2
        from nba_api.stats.static import teams
        import pandas as pd
        
        # Pull live scoreboard directly from NBA official servers
        board = scoreboardv2.ScoreboardV2()
        games = board.get_data_frames()[0]
        
        if games.empty: 
            return None, "No games scheduled."
            
        # Get official team abbreviations (e.g., LAL, BOS)
        nba_teams = teams.get_teams()
        team_dict = {t['id']: t['abbreviation'] for t in nba_teams}
        
        matchups = []
        for _, g in games.iterrows():
            home_id = g['HOME_TEAM_ID']
            away_id = g['VISITOR_TEAM_ID']
            
            home_abbrev = team_dict.get(home_id, 'HOME')
            away_abbrev = team_dict.get(away_id, 'AWAY')
            
            status_id = g['GAME_STATUS_ID'] # 1=Pre-Game, 2=Live, 3=Final
            status_text = g['GAME_STATUS_TEXT']
            is_live_or_final = status_id in [2, 3]
            
            # Extract live scores if the game is actively being played
            line_score = board.get_data_frames()[1]
            home_score, away_score = 0, 0
            if not line_score.empty:
                try:
                    home_row = line_score[line_score['TEAM_ID'] == home_id]
                    away_row = line_score[line_score['TEAM_ID'] == away_id]
                    if not home_row.empty and pd.notna(home_row['PTS'].iloc[0]): 
                        home_score = int(home_row['PTS'].iloc[0])
                    if not away_row.empty and pd.notna(away_row['PTS'].iloc[0]): 
                        away_score = int(away_row['PTS'].iloc[0])
                except: pass
                
            # Format the display status for the dashboard
            if status_id == 1:
                ds = f"Today - {status_text.replace(' ET', '').replace(' EST', '').upper()}"
            else:
                ds = status_text

            matchups.append({
                "home": home_abbrev,
                "away": away_abbrev,
                "status": ds,
                "home_score": home_score,
                "away_score": away_score,
                "is_live_or_final": is_live_or_final
            })
            
        return matchups, "Success"
    except Exception as e: 
        return None, "Failed to connect to NBA API."

@st.cache_data(ttl=60)
def get_nhl_schedule():
    try:
        r = requests.get("https://api-web.nhle.com/v1/schedule/now", timeout=5).json()
        if not r.get('gameWeek') or not r['gameWeek'][0].get('games'): return None, "No games scheduled."
        matchups = []
        for g in r['gameWeek'][0]['games']:
            state = g.get('gameState', 'FUT') 
            il = state in ['LIVE', 'CRIT', 'FINAL', 'OFF']
            ds = "Final" if state in ['FINAL', 'OFF'] else "LIVE" if state in ['LIVE', 'CRIT'] else pd.to_datetime(g['startTimeUTC']).tz_convert('US/Eastern').strftime("%I:%M %p").lstrip("0")
            matchups.append({"home": g['homeTeam']['abbrev'], "away": g['awayTeam']['abbrev'], "status": ds, "home_score": g.get('homeTeam', {}).get('score', 0), "away_score": g.get('awayTeam', {}).get('score', 0), "is_live_or_final": il})
        return matchups, "Success"
    except: return None, "Failed to connect to NHL API."

@st.cache_data(ttl=60)
def get_mlb_schedule():
    try:
        r = requests.get("https://statsapi.mlb.com/api/v1/schedule?sportId=1", timeout=5).json()
        if not r.get('dates') or not r['dates'][0].get('games'): return None, "No games scheduled."
        matchups = []
        for g in r['dates'][0]['games']:
            home = g['teams']['home']['team']['name'].split()[-1][:3].upper()
            away = g['teams']['away']['team']['name'].split()[-1][:3].upper()
            sr = g['status']['detailedState']
            il = sr in ['In Progress', 'Final', 'Game Over', 'Completed Early']
            ds = pd.to_datetime(g['gameDate']).tz_convert('US/Eastern').strftime("%I:%M %p").lstrip("0") if not il and sr in ['Scheduled', 'Pre-Game', 'Warmup'] else sr
            matchups.append({"home": home, "away": away, "status": ds, "home_score": g['teams']['home'].get('score', 0), "away_score": g['teams']['away'].get('score', 0), "is_live_or_final": il})
        return matchups, "Success"
    except: return None, "Failed to connect to MLB API."

def render_scoreboard(sd):
    if not sd: return
    for i in range(0, len(sd), 5):
        cols = st.columns(5)
        for j, g in enumerate(sd[i:i+5]):
            with cols[j]:
                dt = f"{g['away']} <span style='color: #FFD700;'>{g['away_score']}</span> - <span style='color: #FFD700;'>{g['home_score']}</span> {g['home']}" if g['is_live_or_final'] else f"{g['away']} @ {g['home']}"
                st.markdown(f'<div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px; text-align: center; margin-bottom: 10px;"><div style="font-size: 14px; font-weight: bold; color: #fff;">{dt}</div><div style="font-size: 11px; color: #00E5FF; margin-top: 4px;">{g["status"]}</div></div>', unsafe_allow_html=True)

@st.cache_data(ttl=600)
def get_live_line(player_label, stat_type, api_key, sport_path):
    if not api_key: return None, None, "API Key missing in secrets.toml", None, None
    m_map = {"Points": "player_points", "Goals": "player_goals", "Assists": "player_assists", "Shots on Goal": "player_shots_on_goal", "Power Play Points": "player_power_play_points", "Rebounds": "player_rebounds", "PRA (Pts+Reb+Ast)": "player_points_rebounds_assists", "Threes Made": "player_threes", "Hits": "batter_hits", "Home Runs": "batter_home_runs", "Pitcher Strikeouts": "pitcher_strikeouts"}
    market = m_map.get(stat_type, "player_points")
    clean_name = player_label.split("(")[0].strip().lower()
    team_abbr = player_label.split("(")[1].split(")")[0].strip().upper() if "(" in player_label else ""
    
    mega_map = {"ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons", "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers", "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards", "ANA": "Anaheim Ducks", "BUF": "Sabres", "CGY": "Flames", "CAR": "Hurricanes", "COL": "Avalanche", "CBJ": "Blue Jackets", "EDM": "Oilers", "FLA": "Panthers", "LAK": "Kings", "MTL": "Canadiens", "NSH": "Predators", "NJD": "Devils", "NYI": "Islanders", "NYR": "Rangers", "OTT": "Senators", "PIT": "Penguins", "SJS": "Sharks", "SEA": "Kraken", "STL": "Blues", "TBL": "Lightning", "VAN": "Canucks", "VGK": "Knights", "WPG": "Jets"}
    
    used, rem = None, None
    try:
        events_resp = requests.get(f"https://api.the-odds-api.com/v4/sports/{sport_path}/events?apiKey={api_key}", timeout=10)
        events_data = events_resp.json()
        used = events_resp.headers.get('x-requests-used')
        rem = events_resp.headers.get('x-requests-remaining')
        
        if not isinstance(events_data, list) or len(events_data) == 0: return None, None, "No active events", used, rem
        
        target_team_name = mega_map.get(team_abbr)
        target_event_id = None
        if target_team_name:
            for e in events_data:
                if target_team_name in e.get('home_team', '') or target_team_name in e.get('away_team', ''):
                    target_event_id = e['id']; break
        
        events_to_check = [{'id': target_event_id}] if target_event_id else events_data[:5]
        
        for event in events_to_check:
            odds_resp = requests.get(f"https://api.the-odds-api.com/v4/sports/{sport_path}/events/{event['id']}/odds?apiKey={api_key}&regions=us&markets={market}&oddsFormat=american", timeout=10)
            odds_data = odds_resp.json()
            used = odds_resp.headers.get('x-requests-used', used)
            rem = odds_resp.headers.get('x-requests-remaining', rem)
            
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
        import time
        
        nba_players = players.get_players()
        player_dict = [p for p in nba_players if p['full_name'].lower() == cn.lower()]
        if not player_dict: return pd.DataFrame(), 404, []
        
        pid = player_dict[0]['id']
        
        # 🕰️ THE TIME MACHINE: Pull the last 3 Seasons
        seasons = ['2025-26', '2024-25', '2023-24']
        df_list = []
        for s in seasons:
            try:
                log = playergamelog.PlayerGameLog(player_id=pid, season=s)
                df_list.append(log.get_data_frames()[0])
                time.sleep(0.5) # Prevent rate-limiting
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
                if ':' in s: return float(s.split(':')[0]) + float(s.split(':')[1])/60.0
                return float(s)
            except: return 0.0
            
        df['MINS'] = df['MIN'].apply(parse_mins)
        df = df.rename(columns={'REB': 'TRB'}) 
        
        # 🧮 TIME DECAY MATH: Calculate Half-Life weights
        today = pd.to_datetime(datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d"))
        df['Days_Ago'] = (today - df['ValidDate']).dt.days
        df = df[(df['Days_Ago'] >= 0) & (df['Days_Ago'] <= 1095)] # Hard cutoff at 3 years
        
        # Exponential Decay Formula (Half-Life of exactly 200 Days)
        df['Weight'] = np.exp(-0.003465 * df['Days_Ago'])
        
        final_cols = ['ValidDate', 'ShortDate', 'MATCHUP', 'Is_Home', 'MINS', 'PTS', 'TRB', 'AST', 'FG3M', 'Weight']
        df = df[final_cols].sort_values('ValidDate').reset_index(drop=True)
        return df, 200, []
    except Exception as e: return pd.DataFrame(), 500, []

@st.cache_data(ttl=300)
def get_nhl_stats(player_label):
    cn = player_label.split("(")[0].strip()
    try:
        r = requests.get(f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=25&q={requests.utils.quote(cn)}", timeout=5).json()
        pid = next((p.get('playerId', p.get('id')) for p in r if p.get('name','').lower() == cn.lower()), r[0].get('playerId', r[0].get('id')) if r else None)
        log = requests.get(f"https://api-web.nhle.com/v1/player/{pid}/game-log/20252026/2", timeout=5).json().get('gameLog', [])
        if not log: return pd.DataFrame(), 404, []
        df = pd.DataFrame(log)
        for c in ['points', 'goals', 'assists', 'shots', 'powerPlayPoints']: df[c.upper()[:3] if c != 'powerPlayPoints' else 'PPP'] = pd.to_numeric(df.get(c, 0))
        df['Is_Home'] = np.where(df.get('homeRoadFlag', 'H') == 'H', 1, 0)
        df['MINS'] = df.get('toi', '15:00').apply(lambda x: int(str(x).split(':')[0]) + int(str(x).split(':')[1])/60.0 if ':' in str(x) else 0.0)
        df['MATCHUP'], df['ShortDate'] = df['opponentAbbrev'], pd.to_datetime(df['gameDate']).dt.strftime('%b %d')
        return df.iloc[::-1].reset_index(drop=True), 200, []
    except: return pd.DataFrame(), 500, []

@st.cache_data(ttl=300)
def get_mlb_stats(player_label):
    cn = player_label.split("(")[0].strip()
    try:
        sr = requests.get(f"https://statsapi.mlb.com/api/v1/people/search?names={requests.utils.quote(cn)}", timeout=5).json()
        if not sr.get('people'): return pd.DataFrame(), 404, []
        pid = next((p['id'] for p in sr.get('people', []) if p.get('fullName','').lower() == cn.lower()), sr['people'][0]['id'] if sr.get('people') else None)
        log = requests.get(f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=gameLog&group=hitting,pitching", timeout=5).json()
        splits = log.get('stats', [{}])[0].get('splits', [])
        if not splits: return pd.DataFrame(), 404, []
        data = [{'gameDate': s.get('date', '2025-01-01'), 'MATCHUP': s.get('opponent', {}).get('name', 'OPP').split(' ')[-1][:3].upper(), 'Is_Home': 1 if s.get('isHome', True) else 0, 'H': s.get('stat', {}).get('hits', 0), 'HR': s.get('stat', {}).get('homeRuns', 0), 'TB': s.get('stat', {}).get('totalBases', 0), 'K': s.get('stat', {}).get('strikeOuts', 0), 'ER': s.get('stat', {}).get('earnedRuns', 0), 'MINS': float(s.get('stat', {}).get('plateAppearances', s.get('stat', {}).get('battersFaced', 1)))} for s in splits]
        df = pd.DataFrame(data).iloc[::-1].reset_index(drop=True)
        df['ShortDate'] = pd.to_datetime(df['gameDate']).dt.strftime('%b %d')
        return df, 200, []
    except: return pd.DataFrame(), 500, []

def get_fatigue_modifier(rest_status):
    if "B2B" in rest_status: return 0.95, "Tired Legs (-5%)"
    if "3 in 4" in rest_status: return 0.90, "Exhausted (-10%)"
    return 1.00, "Fully Rested"

def estimate_alt_odds(orig_line, orig_odds, new_line, stat_type):
    if orig_line is None or orig_odds is None or orig_line == new_line: return orig_odds
    p_orig = abs(orig_odds)/(abs(orig_odds)+100) if orig_odds < 0 else 100/(orig_odds+100)
    std_matrix = {"Points": 6.0, "Rebounds": 2.5, "Assists": 2.5, "Threes Made": 1.2, "PRA (Pts+Reb+Ast)": 8.0, "Minutes Played": 5.0, "Hits": 1.0, "Pitcher Strikeouts": 2.0}
    std_est = std_matrix.get(stat_type, 3.0)
    z_shift = (orig_line - new_line) / std_est
    p_new = max(0.05, min(0.95, p_orig + (z_shift * 0.35)))
    new_odds = int(round((-100*p_new)/(1-p_new))) if p_new > 0.50 else int(round((100*(1-p_new))/p_new))
    return 5 * round(new_odds/5)

# ==========================================
# 🧠 SKYNET ML ENGINE (Phase 3)
# ==========================================
def run_ml_board(df, s_col, line, opp, league, rest, is_home_current, stat_type):
    df_ml = df.copy()
    
    archetype = get_player_archetype(df_ml, league)
    mod_val, mod_desc = get_archetype_defense_modifier(league, opp, archetype)

    if len(df_ml) < 5: 
        return df_ml, [], 0, "PASS", "#94a3b8", 1.0, "Not enough data", 1.0, "", "", 1.0, "", archetype, "Awaiting Data", "#94a3b8"
    
    y = df_ml[s_col].values
    X = np.arange(len(df_ml)).reshape(-1, 1)
    
    df_ml['Per_Min'] = df_ml[s_col] / df_ml['MINS'].replace(0, 1)
    expected_mins = df_ml['MINS'].tail(5).mean()
    lr = LinearRegression().fit(X, df_ml['Per_Min'].values)
    trend_proj = lr.predict([[len(X)]])[0] * expected_mins
    
    df_ml['Roll3'] = df_ml[s_col].rolling(3).mean().fillna(df_ml[s_col].mean())
    X_rf = df_ml[['Roll3', 'MINS']].values
    rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_rf, y)
    stat_proj = rf.predict([[df_ml['Roll3'].iloc[-1], expected_mins]])[0]
    
    df_ml['Dev'] = df_ml[s_col] - df_ml[s_col].mean()
    X_gb = df_ml[['MINS', 'Dev']].values
    gb = GradientBoostingRegressor(n_estimators=50, random_state=42).fit(X_gb, y)
    today_dev = trend_proj - df_ml[s_col].mean() 
    con_proj = gb.predict([[expected_mins, today_dev]])[0]

    scaler = StandardScaler()
    X_svm_scaled = scaler.fit_transform(X_rf)
    svm = SVR(kernel='rbf', C=10).fit(X_svm_scaled, y)
    base_proj = svm.predict(scaler.transform([[df_ml['Roll3'].iloc[-1], expected_mins]]))[0]
    
    season_avg = df_ml[s_col].mean()
    if season_avg == 0: season_avg = 1 
    
    home_avg = df_ml[df_ml['Is_Home'] == 1][s_col].mean()
    away_avg = df_ml[df_ml['Is_Home'] == 0][s_col].mean()
    if pd.isna(home_avg): home_avg = season_avg
    if pd.isna(away_avg): away_avg = season_avg
    
    home_mod = np.clip(home_avg / season_avg, 0.80, 1.20)
    away_mod = np.clip(away_avg / season_avg, 0.80, 1.20)
    
    current_split_mod = home_mod if is_home_current == 1 else away_mod
    split_text = "Home" if is_home_current == 1 else "Road"
    split_desc = f"+{((current_split_mod-1)*100):.0f}%" if current_split_mod > 1 else f"{((current_split_mod-1)*100):.0f}%"
    
    fatigue_val, fatigue_desc = get_fatigue_modifier(rest)
    base_stat_avg = (trend_proj + stat_proj) / 2
    guru_proj = base_stat_avg * mod_val * fatigue_val * current_split_mod

    # RAW CONSENSUS
    raw_consensus = (trend_proj + stat_proj + con_proj + base_proj + guru_proj) / 5
    
    def get_raw_vote(p):
        if p >= line + 0.3: return "OVER"
        elif p <= line - 0.3: return "UNDER"
        return "PASS"

    raw_vote = get_raw_vote(raw_consensus)
    
    # 🟣 SKYNET SELF-CORRECTION LOOP
    skynet_mod = 1.0
    skynet_msg = "🟣 Skynet: Awaiting enough ledger data to self-correct."
    skynet_color = "#94a3b8"
    
    if raw_vote != "PASS":
        try:
            ledger = load_ledger()
            if not ledger.empty and 'Result' in ledger.columns:
                graded = ledger[ledger['Result'].isin(['Win', 'Loss'])]
                subset = graded[(graded['Stat'] == stat_type) & (graded['Vote'] == raw_vote)]
                total_graded = len(subset)
                
                if total_graded >= 3:
                    wins = len(subset[subset['Result'] == 'Win'])
                    win_rate = wins / total_graded
                    
                    if win_rate <= 0.35: 
                        skynet_mod = 0.85 if raw_vote == "OVER" else 1.15
                        skynet_msg = f"🛑 SKYNET TAX: You are {wins}-{total_graded-wins} on {stat_type} {raw_vote}s. Applying -15% mathematical penalty."
                        skynet_color = "#ff0055"
                    elif win_rate >= 0.60: 
                        skynet_mod = 1.05 if raw_vote == "OVER" else 0.95
                        skynet_msg = f"🔥 SKYNET BOOST: You are {wins}-{total_graded-wins} on {stat_type} {raw_vote}s. Trusting the AI edge."
                        skynet_color = "#00E676"
                    else:
                        skynet_msg = f"⚖️ SKYNET AUDIT: You are {wins}-{total_graded-wins} on {stat_type} {raw_vote}s. Within normal variance."
                        skynet_color = "#FFD700"
                else:
                    skynet_msg = f"🟣 Skynet: Gathering data on {stat_type} {raw_vote}s ({total_graded}/3 required)."
        except Exception as e: pass

    # FINAL CONSENSUS (After Skynet Tax)
    final_consensus = raw_consensus * skynet_mod
    
    def get_final_vote(p):
        if p >= line + 0.3: return "OVER", "#00c853" 
        elif p <= line - 0.3: return "UNDER", "#d50000"
        return "PASS", "#94a3b8"
        
    f_vote, f_color = get_final_vote(final_consensus)
    
    # Scale history for chart
    lr_hist = lr.predict(X) * df_ml['MINS'].values
    rf_hist = rf.predict(X_rf)
    gb_hist = gb.predict(X_gb)
    svm_hist = svm.predict(X_svm_scaled)
    mods = df_ml['MATCHUP'].apply(lambda x: get_archetype_defense_modifier(league, x, archetype)[0]).values
    hist_split_mods = np.where(df_ml['Is_Home'] == 1, home_mod, away_mod)
    guru_hist = ((lr_hist + rf_hist) / 2) * mods * 1.0 * hist_split_mods
    
    df_ml['AI_Proj'] = ((lr_hist + rf_hist + gb_hist + svm_hist + guru_hist) / 5) * skynet_mod

    min_exp = f"Projects {trend_proj:.1f} by weighting recent minutes ({expected_mins:.1f} MPG)."
    stat_exp = f"L3 avg of {df_ml['Roll3'].iloc[-1]:.1f} sets stable floor. Trees favor {get_raw_vote(stat_proj)}."
    dev_dir = "regression" if con_proj < season_avg else "spike"
    con_exp = f"Flags a potential variance {dev_dir} from season norms."
    base_exp = f"High-dimension mapping of minute/production correlation."
    mod_short = mod_desc.split('(')[0].strip().replace('🛡️', '').replace('🏃', '').strip()
    guru_exp = f"Factors {mod_short}."

    board = [
        {"name": "⏱️ MIN Maximizer", "model": "Linear Regression", "proj": trend_proj, "vote": get_raw_vote(trend_proj), "color": get_final_vote(trend_proj)[1], "quote": min_exp},
        {"name": "📊 Statistician", "model": "Random Forest", "proj": stat_proj, "vote": get_raw_vote(stat_proj), "color": get_final_vote(stat_proj)[1], "quote": stat_exp},
        {"name": "🃏 Contrarian", "model": "Gradient Boosting", "proj": con_proj, "vote": get_raw_vote(con_proj), "color": get_final_vote(con_proj)[1], "quote": con_exp},
        {"name": "🛡️ Baseline", "model": "Support Vector Machine", "proj": base_proj, "vote": get_raw_vote(base_proj), "color": get_final_vote(base_proj)[1], "quote": base_exp},
        {"name": "🎯 Context Guru", "model": "Radar, Rest, Arena", "proj": guru_proj, "vote": get_raw_vote(guru_proj), "color": get_final_vote(guru_proj)[1], "quote": guru_exp}
    ]
    return df_ml, board, final_consensus, f_vote, f_color, mod_val, mod_desc, current_split_mod, split_text, split_desc, fatigue_val, fatigue_desc, archetype, skynet_msg, skynet_color

# ==========================================
# 8. UI ENGINE
# ==========================================
def render_syndicate_board(league_key):
    if league_key == "NBA": sport_path = "basketball_nba"; teams = NBA_TEAMS
    elif league_key == "MLB": sport_path = "baseball_mlb"; teams = MLB_TEAMS
    else: sport_path = "icehockey_nhl"; teams = NHL_TEAMS
    
    top_c1, top_c2, _ = st.columns([1, 1, 2])
    with top_c1: sync = st.toggle("📡 Auto-Sync Vegas Odds", value=False, key=f"sy_{league_key}")
    with top_c2:
        is_home_toggle = st.toggle("🏠 Playing at Home?", value=True, key=f"loc_{league_key}")
        is_home_current = 1 if is_home_toggle else 0
    
    with st.container():
        c1, c2, c3, c4 = st.columns([2, 1.5, 1, 1.5])
        with c1: 
            search_query = st.text_input(f"🔍 1. Search Player", placeholder="e.g. Judge, LeBron, or McDavid", key=f"sq_{league_key}")
            player_name = None
            if search_query:
                if league_key == "NBA": matches = search_nba_players(search_query)
                elif league_key == "MLB": matches = search_mlb_players(search_query)
                else: matches = search_nhl_players(search_query)
                if matches: player_name = st.selectbox("🎯 2. Select Exact Match", matches, key=f"dd_{league_key}")
                else: st.caption("No players found in database.")
                
        with c2: 
            if league_key == "NBA": opts = ["Points", "Rebounds", "Assists", "Threes Made", "PRA (Pts+Reb+Ast)", "Points + Rebounds", "Points + Assists", "Rebounds + Assists", "Minutes Played"]
            elif league_key == "MLB": opts = ["Hits", "Home Runs", "Total Bases", "Pitcher Strikeouts", "Pitcher Earned Runs"]
            else: opts = ["Points", "Goals", "Assists", "Shots on Goal"]
            stat_type = st.selectbox("Stat", opts, key=f"st_{league_key}")
            live_odds_display = st.empty()

        f_line, f_odds, msg, used, rem = None, None, "", None, None
        if sync and player_name:
            with st.spinner("Syncing Odds..."):
                f_line, f_odds, msg, used, rem = get_live_line(player_name, stat_type, ODDS_API_KEY, sport_path)
            if used is not None and rem is not None:
                st.session_state['api_used'] = int(used); st.session_state['api_remaining'] = int(rem)
            
            if f_line is not None and f_odds is not None:
                display_odds = f"+{f_odds}" if f_odds > 0 else f"{f_odds}"
                bg_color, border_color = "rgba(0, 230, 118, 0.1)", "#00E676"
                live_odds_display.markdown(f'<div style="background-color: {bg_color}; border: 1px solid {border_color}; padding: 10px; border-radius: 6px; margin-top: 10px;"><div style="font-size: 11px; font-weight: 900; color: {border_color}; letter-spacing: 1px;">📡 LIVE MARKET SYNCED</div><div style="font-size: 16px; font-weight: bold; color: #fff;">{stat_type} O/U {f_line} <span style="color: #94a3b8; font-size: 14px;">({display_odds})</span></div></div>', unsafe_allow_html=True)
            elif f_odds is not None:
                display_odds = f"+{f_odds}" if f_odds > 0 else f"{f_odds}"
                bg_color, border_color = "rgba(255, 215, 0, 0.1)", "#FFD700"
                live_odds_display.markdown(f'<div style="background-color: {bg_color}; border: 1px solid {border_color}; padding: 10px; border-radius: 6px; margin-top: 10px;"><div style="font-size: 11px; font-weight: 900; color: {border_color}; letter-spacing: 1px;">🟡 MARKET PARTIAL SYNC</div><div style="font-size: 16px; font-weight: bold; color: #fff;">{stat_type} Odds <span style="color: #94a3b8; font-size: 14px;">({display_odds})</span></div></div>', unsafe_allow_html=True)
            else: live_odds_display.caption(f"🟡 {msg}")

        with c3: 
            start_line = float(f_line) if (sync and f_line is not None) else 0.5
            line = st.number_input("Line", value=start_line, step=0.5, key=f"ln_{league_key}")
            if sync and f_line is not None and f_odds is not None: st.session_state[f"odds_{league_key}"] = estimate_alt_odds(float(f_line), int(f_odds), line, stat_type)
            elif f"odds_{league_key}" not in st.session_state: st.session_state[f"odds_{league_key}"] = -110
            odds = st.number_input("Odds", step=5, key=f"odds_{league_key}")
                
        with c4: 
            opp = st.selectbox("Opponent", teams, key=f"op_{league_key}")
            rest = st.selectbox("Fatigue", ["Rested (1+ Days)", "Tired (B2B)", "Exhausted (3 in 4)"], key=f"rest_{league_key}")

    btn_c1, btn_c2, _ = st.columns([1, 1, 2])
    with btn_c1: analyze_pressed = st.button(f"🚀 Analyze {league_key} Player", type="primary", use_container_width=True, key=f"btn_analyze_{league_key}")
    
    if analyze_pressed and player_name: st.session_state[f"target_player_{league_key}"] = player_name
    target_player = st.session_state.get(f"target_player_{league_key}")

    if target_player:
        with st.spinner(f"Scouting data for {target_player}..."):
            if league_key == "NBA": df, status_code, logs = get_nba_stats(target_player)
            elif league_key == "MLB": df, status_code, logs = get_mlb_stats(target_player)
            else: df, status_code, logs = get_nhl_stats(target_player)
            
        if status_code == 429: st.error("🚨 **Error 429: Rate Limited.** Please wait 60 seconds.")
        elif status_code == 500: st.warning("🟡 **Server Error.** Try again in a moment.")
        elif not df.empty:
            s_map = {"Points": "PTS", "Goals": "G", "Assists": "A", "Shots on Goal": "SOG", "Rebounds": "TRB", "PRA (Pts+Reb+Ast)": "PRA", "Power Play Points": "PPP", "Minutes Played": "MINS", "Threes Made": "FG3M", "Points + Rebounds": "PR", "Points + Assists": "PA", "Rebounds + Assists": "RA", "Hits": "H", "Home Runs": "HR", "Total Bases": "TB", "Pitcher Strikeouts": "K", "Pitcher Earned Runs": "ER"}
            s_col = s_map.get(stat_type, "PTS")
            
            if league_key == "NBA":
                if s_col == "A": s_col = "AST"
                if s_col == "PRA": df['PRA'] = df['PTS'] + df['TRB'] + df['AST']
                if s_col == "PR": df['PR'] = df['PTS'] + df['TRB']
                if s_col == "PA": df['PA'] = df['PTS'] + df['AST']
                if s_col == "RA": df['RA'] = df['TRB'] + df['AST']
            
            df_with_ml, board, c_proj, c_vote, c_color, mod_val, mod_desc, current_split_mod, split_text, split_desc, fatigue_val, fatigue_desc, archetype, skynet_msg, skynet_color = run_ml_board(df, s_col, line, opp, league_key, rest, is_home_current, stat_type)
            
            if len(board) == 0:
                st.warning(f"⚠️ **Insufficient Data:** {target_player} has played fewer than 5 games this season. The Machine Learning models require at least 5 games of history to generate an accurate projection.")
            else:
                std_dev = df_with_ml[s_col].std()
                if pd.isna(std_dev) or std_dev == 0: std_dev = 1.0
                
                sims = np.random.normal(loc=c_proj, scale=std_dev, size=10000)
                if c_vote == "OVER": win_prob = np.sum(sims > line) / 10000.0
                elif c_vote == "UNDER": win_prob = np.sum(sims < line) / 10000.0
                else: win_prob = 0.50
                
                lock_pressed = False
                with btn_c2:
                    if c_vote != "PASS": lock_pressed = st.button(f"🔒 Lock {league_key} Pick", use_container_width=True, type="primary", key=f"lock_{league_key}")
                if lock_pressed:
                    save_to_ledger(league_key, target_player, stat_type, line, odds, c_proj, c_vote, win_prob)
                    st.success("Pick locked! It is safely stored in your Google Sheets Database.")
                    
                if odds < 0: implied_prob = abs(odds) / (abs(odds) + 100); profit = 100 / (abs(odds) / 100); risk = 100
                else: implied_prob = 100 / (odds + 100); profit = odds; risk = 100
                ev_dollars = (win_prob * profit) - ((1 - win_prob) * risk)
                edge_pct = (win_prob - implied_prob) * 100

                if c_vote == "OVER": ai_summary_short = f"Projected to clear {line} with a {win_prob*100:.1f}% probability."
                elif c_vote == "UNDER": ai_summary_short = f"Projected to stay under {line} with a {win_prob*100:.1f}% probability."
                else: ai_summary_short = f"Projection too close to {line} to recommend."
                    
                if league_key == "NBA" and "Exploit" in mod_desc: ai_summary_short += f"<br><span style='color:#FFD700; font-weight:bold;'>🚨 Archetype Exploit vs {opp}</span>"
                elif league_key == "NBA" and "Fade" in mod_desc: ai_summary_short += f"<br><span style='color:#ff0055; font-weight:bold;'>🛑 Archetype Fade vs {opp}</span>"
                
                ai_summary_short += f"<br><br><span style='color:{skynet_color}; font-weight:bold;'>{skynet_msg}</span>"

                sum_c1, sum_c2, sum_c3, sum_c4 = st.columns(4)
                
                with sum_c1:
                    st.markdown(f"""
                    <div class="verdict-box" style="background-color: {c_color}15; border-color: {c_color}; color: #fff; height: 100%;">
                        <div style="font-size:10px; font-weight:bold; color:{c_color}; letter-spacing: 1px;">AI CONSENSUS</div>
                        <div style="font-size:26px; font-weight:900; margin: 4px 0px;">{c_vote}</div>
                        <div style="font-size:14px; font-weight:bold; margin-bottom: 6px;">Proj: {c_proj:.2f}</div>
                        <div style="font-size:11px; color:#94a3b8; border-top: 1px solid {c_color}50; padding-top: 8px; line-height: 1.3;">
                            {ai_summary_short}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with sum_c2:
                    if c_vote == "PASS":
                        ev_color = "#94a3b8"
                        st.markdown(f'<div class="verdict-box" style="background-color: #1e293b; border-color: #334155; color: #fff; height: 100%;"><div style="font-size:10px; font-weight:bold; color:#94a3b8; letter-spacing: 1px;">EXPECTED VALUE (+EV)</div><div style="font-size:22px; font-weight:900; color:{ev_color};">N/A - PASS</div><div style="font-size:13px;">Market is highly efficient.</div></div>', unsafe_allow_html=True)
                    else:
                        ev_color = "#00c853" if ev_dollars > 0 else "#d50000"
                        st.markdown(f'<div class="verdict-box" style="background-color: #1e293b; border-color: {ev_color}; color: #fff; height: 100%;"><div style="font-size:10px; font-weight:bold; color:#94a3b8; letter-spacing: 1px;">EXPECTED VALUE (+EV)</div><div style="font-size:22px; font-weight:900; color:{ev_color};">${ev_dollars:+.2f} per $100</div><div style="font-size:13px;">Win Prob: {win_prob*100:.1f}% | Edge: {edge_pct:+.1f}%</div></div>', unsafe_allow_html=True)

                with sum_c3:
                    df_l10 = df_with_ml.tail(10).reset_index(drop=True)
                    l10_hits = int((df_l10[s_col] > line).sum())
                    df_l5 = df_l10.tail(5)
                    l5_hits = int((df_l5[s_col] > line).sum())
                    hit_color = "#00c853" if l10_hits >= 6 else ("#d50000" if l10_hits <= 4 else "#FFD700")
                    st.markdown(f'<div class="verdict-box" style="background-color: #1e293b; border-color: #334155; color: #fff; height: 100%;"><div style="font-size:10px; font-weight:bold; color:#94a3b8; letter-spacing: 1px;">HIT RATE (OVER {line})</div><div style="font-size:22px; font-weight:900; color:{hit_color};">{l10_hits}/10</div><div style="font-size:13px;">L5: {l5_hits}/5</div></div>', unsafe_allow_html=True)

                with sum_c4:
                    season_avg = round(df[s_col].mean(), 1)
                    l10_avg = round(df_l10[s_col].mean(), 1)
                    l5_avg = round(df_l5[s_col].mean(), 1)
                    trend_color = "#00c853" if l5_avg >= season_avg * 1.1 else ("#d50000" if l5_avg <= season_avg * 0.9 else "#fff")
                    st.markdown(f'<div class="verdict-box" style="background-color: #1e293b; border-color: #334155; color: #fff; height: 100%;"><div style="font-size:10px; font-weight:bold; color:#94a3b8; letter-spacing: 1px; margin-bottom: 2px;">RECENT AVERAGES</div><div style="display: flex; justify-content: space-around; align-items: center; margin-top: 2px;"><div><div style="font-size:10px; color:#94a3b8;">Season</div><div style="font-size:18px; font-weight:900;">{season_avg}</div></div><div><div style="font-size:10px; color:#94a3b8;">L10</div><div style="font-size:18px; font-weight:900;">{l10_avg}</div></div><div><div style="font-size:10px; color:#94a3b8;">L5</div><div style="font-size:18px; font-weight:900; color:{trend_color};">{l5_avg}</div></div></div></div>', unsafe_allow_html=True)
                
                b_cols = st.columns(len(board))
                for i, m in enumerate(board):
                    b_cols[i].markdown(f'<div class="board-member"><div class="board-name">{m["name"]}</div><div class="board-model">{m["model"]}</div><div style="font-size:11px; color:#94a3b8; font-style:italic; line-height:1.3; margin-bottom:12px; min-height:45px;">"{m["quote"]}"</div><div style="color:#94a3b8; font-size:12px; border-top:1px dashed #334155; padding-top:8px;">Proj: <span style="color:#fff; font-weight:bold;">{m["proj"]:.2f}</span></div><div class="board-vote" style="color:{m["color"]}; margin-top:2px;">{m["vote"]}</div></div>', unsafe_allow_html=True)

                st.markdown("#### 📊 L10 Performance vs Line")
                chart_col, side_col = st.columns([3.2, 1.2])
                
                with chart_col:
                    df_l10['Matchup_Formatted'] = np.where(df_l10['Is_Home'] == 1, "vs " + df_l10['MATCHUP'], "@ " + df_l10['MATCHUP'])
                    df_l10['Matchup_Label'] = df_l10['ShortDate'] + "|" + df_l10['Matchup_Formatted']
                    
                    bars = alt.Chart(df_l10).mark_bar(opacity=0.7).encode(
                        x=alt.X('Matchup_Label', sort=None, title=None, axis=alt.Axis(labelAngle=0, labelExpr="split(datum.value, '|')")),
                        y=alt.Y(s_col, title=stat_type),
                        color=alt.condition(alt.datum[s_col] > line, alt.value('#00c853'), alt.value('#d50000')),
                        tooltip=[alt.Tooltip('ShortDate', title='Date'), alt.Tooltip('Matchup_Formatted', title='Opponent'), alt.Tooltip(s_col, title='Actual Stats'), alt.Tooltip('AI_Proj', title='AI Projection', format='.2f')]
                    ).properties(height=350)

                    vegas_rule = alt.Chart(pd.DataFrame({'y': [line]})).mark_rule(color='#FFD700', strokeDash=[5,5], size=2).encode(y='y')
                    ai_line = alt.Chart(df_l10).mark_line(color='#00E5FF', strokeWidth=3, point=alt.OverlayMarkDef(color='#00E5FF', size=60)).encode(x=alt.X('Matchup_Label', sort=None), y=alt.Y('AI_Proj'))
                    text = bars.mark_text(align='center', baseline='top', dy=5, fontSize=15, fontWeight='bold').encode(text=alt.Text(s_col, format='.0f'), color=alt.value('#ffffff'))
                    final_chart = (bars + vegas_rule + ai_line + text).configure(background='transparent').configure_axis(gridColor='#334155', domainColor='#334155', tickColor='#334155', labelColor='#94a3b8', titleColor='#f8fafc').configure_view(strokeWidth=0)
                    st.altair_chart(final_chart, use_container_width=True)
                    st.caption("🟡 Dashed Yellow Line: Vegas Line &nbsp; | &nbsp; 🔵 Solid Cyan Line: AI Projection (Skynet Adjusted)")
                    
                with side_col:
                    with st.expander("📊 Matchup Intel (Team Stats)", expanded=True):
                        player_team = target_player.split('(')[1].replace(')', '').strip() if '(' in target_player else "Team"
                        st.markdown(f"<div style='text-align:center; font-weight:900; font-size:16px; color:#00E5FF;'>{player_team} vs {opp}</div>", unsafe_allow_html=True)
                        st.markdown("<hr style='margin: 10px 0px; border-color: #334155;'>", unsafe_allow_html=True)
                        
                        if league_key == "NBA":
                            st.caption("**🧬 AI Player Archetype**")
                            st.markdown(f"<div style='font-size:14px; font-weight:bold; color:#00E676;'>{archetype}</div>", unsafe_allow_html=True)
                            if "Exploit" in mod_desc or "Fade" in mod_desc:
                                st.markdown(f"<div style='font-size:12px; color:#FFD700; margin-top:2px; font-style:italic;'>{mod_desc}</div>", unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)
                        else:
                            diff_pct = 95 if mod_val < 1.0 else (15 if mod_val > 1.0 else 50)
                            st.caption(f"**🛡️ {opp} Defense Difficulty**")
                            st.progress(max(0.0, min(1.0, diff_pct / 100.0)), text=f"{mod_desc}")
                            st.markdown("<br>", unsafe_allow_html=True)

                        st.caption(f"**⚔️ History vs {opp} (All Time)**")
                        df_opp = df_with_ml[df_with_ml['MATCHUP'] == opp]
                        if not df_opp.empty:
                            opp_hits = int((df_opp[s_col] > line).sum()); opp_total = len(df_opp); opp_win_pct = (opp_hits / opp_total) * 100
                            hit_color = "#00c853" if opp_win_pct >= 60 else ("#d50000" if opp_win_pct <= 40 else "#FFD700")
                            st.markdown(f"<div style='font-size:22px; font-weight:900; color:{hit_color};'>{opp_win_pct:.0f}% <span style='font-size:14px; color:#94a3b8; font-weight:normal;'>({opp_hits}/{opp_total} G)</span></div>", unsafe_allow_html=True)
                        else: st.markdown("<div style='font-size:14px; color:#94a3b8;'>No recent data vs this team.</div>", unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.caption(f"**🏟️ Venue Advantage ({split_text})**")
                        split_bar_val = (current_split_mod - 0.8) / 0.4
                        st.progress(max(0.0, min(1.0, split_bar_val)), text=split_desc)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.caption(f"**🔋 {player_team} Energy Levels**")
                        fat_pct = 100 if fatigue_val == 1.0 else (70 if fatigue_val == 0.95 else 40)
                        st.progress(fat_pct / 100.0, text=fatigue_desc)
        else: st.warning(f"🚨 **Warning:** No stats available for {target_player}. They may be a rookie or not logged yet.")

# ==========================================
# 8. APP WRAPPER
# ==========================================
video_path = "delorean.mp4"
if os.path.exists(video_path):
    with open(video_path, "rb") as video_file: video_base64 = base64.b64encode(video_file.read()).decode()
    html_code = f"""<!DOCTYPE html><html><head><style>@import url('https://fonts.googleapis.com/css2?family=Audiowide&display=swap');body {{margin: 0;padding: 0;background-color: #0f172a;overflow: hidden;font-family: 'Audiowide', sans-serif;}}.b2tf-header-container {{position: relative;overflow: hidden;border-radius: 12px;border: 3px solid #ff0055;text-align: center;background-color: #0f172a;box-shadow: 0 0 15px #ff0055;height: 274px; display: flex;align-items: center;justify-content: center;flex-direction: column;}}@keyframes fade-out-video {{0% {{ opacity: 0.6; }}100% {{ opacity: 0; }}}}.b2tf-video-bg {{position: absolute;top: 0;left: 0;width: 100%;height: 100%;z-index: 0;opacity: 0.6;object-fit: cover;animation: fade-out-video 2.3s ease-out 4.6s forwards; }}.b2tf-content {{position: relative;z-index: 1;opacity: 0;animation: fade-in-text 4.7s ease-out 4.5s forwards; }}@keyframes fade-in-text {{0% {{ opacity: 0; transform: translateY(15px) scale(0.9); }}100% {{ opacity: 1; transform: translateY(0) scale(1); }}}}h1 {{color: #ffcc00; font-size: 56px; font-weight: 900; margin: 0; text-shadow: 0 0 10px #ff6600, 0 0 20px #ff0000, 0 0 30px #ff0000; letter-spacing: 2px;}}.subtitle {{color: #00E5FF; font-size: 16px; font-weight: bold; letter-spacing: 3px; margin-top: 5px; text-shadow: 0 0 5px #00E5FF; background: rgba(15, 23, 42, 0.7); padding: 5px 15px; border-radius: 8px; display: inline-block;}}</style></head><body><div class="b2tf-header-container"><video id="delorean-vid" class="b2tf-video-bg" autoplay muted playsinline><source src="data:video/mp4;base64,{video_base64}" type="video/mp4"></video><div class="b2tf-content"><h1>B2TF ALMANAC</h1><div class="subtitle">ROADS? WHERE WE'RE GOING, WE DON'T NEED ROADS.</div></div></div><script>var video = document.getElementById('delorean-vid');video.addEventListener('loadedmetadata', function() {{video.currentTime = 1.0;}}, false);</script></body></html>"""
    components.html(html_code, height=280)
else: st.error("⚠️ The system cannot find the video file.")

with st.expander("⛽ System Diagnostics & API Fuel Gauge", expanded=False):
    diag_c1, diag_c2 = st.columns([1, 2])
    with diag_c1:
        st.markdown("**⚙️ Module Status**")
        st.caption("✅ Odds API (Live)\n✅ NBA Core (Active)\n✅ NHL/MLB (Active)\n✅ Archetype Engine (Online)\n🟣 Skynet Self-Correction (Online)\n☁️ Google DB (Online)")
    with diag_c2:
        st.markdown("**🔋 Odds API Fuel Level**")
        if ODDS_API_KEY:
            f_col1, f_col2 = st.columns([4, 1])
            with f_col2:
                if st.button("🔄 Check", key="chk_quota"): check_api_quota()
            with f_col1:
                if 'api_used' in st.session_state and 'api_remaining' in st.session_state:
                    used = int(st.session_state['api_used']); rem = int(st.session_state['api_remaining']); total = used + rem; fuel_pct = rem / total if total > 0 else 0.0
                    color = "#00E676" if fuel_pct > 0.3 else ("#FFD700" if fuel_pct > 0.1 else "#ff0055")
                    st.markdown(f'<div style="background-color: #1e293b; border-radius: 5px; width: 100%; height: 20px; border: 1px solid #334155; margin-top: 5px;"><div style="background-color: {color}; width: {fuel_pct*100}%; height: 100%; border-radius: 4px; transition: width 0.5s;"></div></div><div style="font-size: 12px; color: #94a3b8; margin-top: 5px; text-align: right;">{rem} / {total} Requests Remaining</div>', unsafe_allow_html=True)
                else: st.caption("Sync a bet or hit 'Check' to load data.")
        else: st.warning("API Key missing.")

tab_nba, tab_nhl, tab_mlb, tab_roi = st.tabs(["🏀 NBA Board", "🏒 NHL Board", "⚾ MLB Board", "🏦 ROI Ledger"])

with tab_nba:
    t_col1, t_col2 = st.columns([8, 1])
    with t_col1: st.markdown("### 📅 Today's Slate")
    with t_col2: 
        if st.button("🔄 Refresh", key="ref_nba"): st.rerun()
    with st.spinner("Loading today's matchups..."): schedule_data, msg = get_nba_schedule()
    if schedule_data: render_scoreboard(schedule_data)
    else: st.info(msg)
    st.markdown("---")
    render_syndicate_board("NBA")
st.markdown("---")
with st.expander("🔥 Launch NBA Heaters & Freezers Scanner", expanded=False):
        st.markdown("Scan all players active on today's slate to find the most extreme hot and cold streaks (Last 5 Games vs Season Average).")
        nba_heat_c1, nba_heat_c2 = st.columns([1, 2])
        with nba_heat_c1:
            target_stat_nba = st.selectbox("Select Target Stat", ["Points", "Rebounds", "Assists", "Threes Made", "PRA (Pts+Reb+Ast)", "Points + Rebounds", "Points + Assists", "Rebounds + Assists"])
        
        if st.button("🚀 Scan Today's NBA Slate", type="primary"):
            with st.status(f"Scanning the NBA database for {target_stat_nba} trends...", expanded=True) as status:
                df_nba_heat, nba_heat_msg = run_nba_heaters(target_stat_nba)
                if df_nba_heat is not None:
                    status.update(label="Scan Complete!", state="complete", expanded=False)
                    st.success(nba_heat_msg)
                    st.dataframe(df_nba_heat, use_container_width=True, hide_index=True)
                else:
                    status.update(label="Scan Complete!", state="complete", expanded=False)
                    st.warning(nba_heat_msg)
with tab_nhl:
    t_col1, t_col2 = st.columns([8, 1])
    with t_col1: st.markdown("### 📅 Today's Slate")
    with t_col2: 
        if st.button("🔄 Refresh", key="ref_nhl"): st.rerun()
    with st.spinner("Loading today's matchups..."): schedule_data, msg = get_nhl_schedule()
    if schedule_data: render_scoreboard(schedule_data)
    else: st.info(msg)
    st.markdown("---")
    render_syndicate_board("NHL")
    st.markdown("---")
    with st.expander("🔥 Launch Barn Burner 2.0 (Daily SOG Matchup Scanner)", expanded=False):
        st.markdown("This tool scans today's entire NHL schedule to find high-volume shooters playing against 'Swiss Cheese' defenses (teams allowing 32+ SOG/game).")
        if st.button("🚀 Run Barn Burner Scan", type="primary"):
            with st.status("Scanning the NHL schedule and player logs...", expanded=True) as status:
                df_bb, bb_msg = run_barn_burner()
                if df_bb is not None:
                    status.update(label="Scan Complete!", state="complete", expanded=False)
                    st.success(bb_msg)
                    st.dataframe(df_bb, use_container_width=True, hide_index=True)
                    st.caption("💡 LEGEND: '🧀 BAD DEFENSE' means the opponent gives up 32+ shots/game.")
                else: status.update(label="Scan Complete!", state="complete", expanded=False); st.warning(bb_msg)

with tab_mlb:
    t_col1, t_col2 = st.columns([8, 1])
    with t_col1: st.markdown("### 📅 Today's Slate")
    with t_col2: 
        if st.button("🔄 Refresh", key="ref_mlb"): st.rerun()
    with st.spinner("Loading today's matchups..."): schedule_data, msg = get_mlb_schedule()
    if schedule_data: render_scoreboard(schedule_data)
    else: st.info(msg)
    st.markdown("---")
    render_syndicate_board("MLB")
    st.markdown("---")
    with st.expander("⚾ Launch MLB Heaters Scanner", expanded=False):
        st.markdown("Scan today's active rosters to find players on extreme hot streaks based on their Last 5 games.")
        heat_col1, heat_col2 = st.columns([1, 2])
        with heat_col1: target_stat = st.selectbox("Select Target Stat", ["Hits", "Home Runs", "Pitcher Strikeouts"])
        if st.button("🚀 Scan Today's MLB Slate", type="primary"):
            with st.status(f"Scanning MLB rosters for {target_stat} heaters...", expanded=True) as status:
                df_heat, heat_msg = run_mlb_heaters(target_stat)
                if df_heat is not None:
                    status.update(label="Scan Complete!", state="complete", expanded=False)
                    st.success(heat_msg)
                    st.dataframe(df_heat, use_container_width=True, hide_index=True)
                else: status.update(label="Scan Complete!", state="complete", expanded=False); st.warning(heat_msg)

with tab_roi:
    roi_mode = st.radio("Select View:", ["🎯 Individual Picks", "🎟️ Parlay Tracker", "💵 Wallet Manager"], horizontal=True)
    st.markdown("---")
    
    if roi_mode == "💵 Wallet Manager":
        st.markdown("### 💵 Multi-Sportsbook Wallet")
        st.caption("Track balances across different apps. Log casino wins or losses here to keep your true sports ROI pure.")
        bw_c1, bw_c2 = st.columns([2, 1])
        with bw_c1:
            with st.form("bankroll_form"):
                sc1, sc2 = st.columns(2)
                t_book = sc1.selectbox("Sportsbook", SPORTSBOOKS)
                t_type = sc2.selectbox("Transaction Type", ["Deposit (Out of Pocket)", "Withdrawal (Cash Out)", "Casino Win (House Money)", "Casino Loss (Bad Spins)"])
                t_amount = st.number_input("Amount ($)", min_value=1.0, step=10.0)
                if st.form_submit_button("Log Transaction"):
                    amt = -t_amount if ("Withdrawal" in t_type or "Loss" in t_type) else t_amount
                    lbl = "Casino" if "Casino" in t_type else "Withdrawal" if "Withdrawal" in t_type else "Deposit"
                    save_bankroll_transaction(t_book, lbl, amt)
                    st.success("Transaction Logged to Cloud!")
                    time.sleep(1); st.rerun()
                    
        b_df, p_df = load_bankroll(), load_parlay_ledger()
        book_balances, total_liquid = {}, 0.0
        tot_dep, tot_wit, tot_cas, tot_sports = 0.0, 0.0, 0.0, 0.0
        
        if not b_df.empty:
            tot_dep = pd.to_numeric(b_df[b_df['Type'] == 'Deposit']['Amount'], errors='coerce').sum() if 'Deposit' in b_df['Type'].values else 0.0
            tot_wit = abs(pd.to_numeric(b_df[b_df['Type'] == 'Withdrawal']['Amount'], errors='coerce').sum()) if 'Withdrawal' in b_df['Type'].values else 0.0
            tot_cas = pd.to_numeric(b_df[b_df['Type'] == 'Casino']['Amount'], errors='coerce').sum() if 'Casino' in b_df['Type'].values else 0.0
        
        for book in SPORTSBOOKS:
            bal, has_hist = 0.0, False
            if not b_df.empty and book in b_df['Sportsbook'].values:
                bal += pd.to_numeric(b_df[b_df['Sportsbook'] == book]['Amount'], errors='coerce').sum(); has_hist = True
            if not p_df.empty and book in p_df['Sportsbook'].values:
                has_hist = True
                for _, r in p_df[p_df['Sportsbook'] == book].iterrows():
                    o = pd.to_numeric(r['Odds'], errors='coerce'); risk = pd.to_numeric(r['Risk'], errors='coerce'); is_f = r.get('Is_Free_Bet', False)
                    if r['Result'] == 'Win': 
                        prof = (risk * (o/100)) if o > 0 else (risk / (abs(o)/100))
                        bal += prof + (0 if is_f else risk); tot_sports += prof
                    elif r['Result'] in ['Loss', 'Pending']: 
                        bal -= (0 if is_f else risk)
                        if r['Result'] == 'Loss': tot_sports -= (0 if is_f else risk)
            if has_hist or bal != 0.0: book_balances[book] = bal; total_liquid += bal
                
        with bw_c2:
            cas_color = "#00E676" if tot_cas >= 0 else "#ff0055"
            spo_color = "#00E676" if tot_sports >= 0 else "#ff0055"
            st.markdown(f"""<div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; text-align: center; margin-top: 28px;"><div style="color: #94a3b8; font-size: 12px; font-weight: bold; letter-spacing: 1px;">TOTAL LIQUID BALANCE</div><div style="color: #00E676; font-size: 36px; font-weight: 900; margin: 10px 0px;">${max(total_liquid, 0.0):.2f}</div><div style="display: flex; justify-content: space-between; font-size: 12px; border-top: 1px dashed #334155; padding-top: 12px; margin-top: 15px;"><span style="color: #94a3b8;">Out of Pocket: <span style="color: #fff;">${max((tot_dep - tot_wit), 0.0):.2f}</span></span><span style="color: #94a3b8;">Net Casino: <span style="color: {cas_color};">{tot_cas:+.2f}</span></span><span style="color: #94a3b8;">Sports Profit: <span style="color: {spo_color};">${tot_sports:+.2f}</span></span></div></div>""", unsafe_allow_html=True)
            
        if book_balances:
            st.markdown("#### 📱 Portfolio Breakdown")
            port_cols = st.columns(min(len(book_balances), 4))
            for i, (book, bal) in enumerate(book_balances.items()):
                col = port_cols[i % 4]; color = "#00E676" if bal >= 0 else "#ff0055"
                col.markdown(f'<div style="background-color: #0f172a; border-left: 4px solid {color}; border-radius: 6px; padding: 15px; margin-bottom: 10px; border-top: 1px solid #334155; border-right: 1px solid #334155; border-bottom: 1px solid #334155;"><div style="font-size: 14px; font-weight: bold; color: #00E5FF; margin-bottom: 5px;">{book}</div><div style="font-size: 20px; font-weight: 900; color: #fff;">${max(bal, 0.0):.2f}</div></div>', unsafe_allow_html=True)

    elif roi_mode == "🎯 Individual Picks":
        roi_col1, roi_col2 = st.columns([4, 1])
        with roi_col1:
            st.markdown("### 🏦 The Bankroll (Single Units)")
            st.caption("Grades based on a flat $100 bet size at locked-in Odds.")
        with roi_col2:
            if st.button("🤖 Auto-Grade Pending", type="primary", use_container_width=True):
                with st.spinner("Checking official APIs for final game stats..."): _, grade_msg = auto_grade_ledger()
                st.success(grade_msg); time.sleep(1.5); st.rerun()
                
        ledger_df = load_ledger()
        if not ledger_df.empty:
            graded_df = ledger_df[ledger_df['Result'].isin(['Win', 'Loss'])]
            wins = len(graded_df[graded_df['Result'] == 'Win'])
            losses = len(graded_df[graded_df['Result'] == 'Loss'])
            total_graded = wins + losses
            
            profit = 0.0
            for _, row in graded_df.iterrows():
                o = pd.to_numeric(row['Odds'], errors='coerce')
                if row['Result'] == 'Win': profit += (100 / (abs(o)/100)) if o < 0 else o
                else: profit -= 100
                    
            roi = (profit / (total_graded * 100) * 100) if total_graded > 0 else 0.0
            win_pct = (wins / total_graded * 100) if total_graded > 0 else 0.0
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Graded Picks", f"{total_graded}")
            m2.metric("Win Rate", f"{win_pct:.1f}%")
            m3.metric("Net Profit (from $100 bets)", f"${profit:+.2f}")
            m4.metric("ROI (%)", f"{roi:+.1f}%")
            
            st.markdown("#### 🎫 Your Bet Slips")
            new_results = {}
            reversed_df = ledger_df.iloc[::-1].reset_index()
            
            for i, row in reversed_df.iterrows():
                orig_idx = row['index']
                o = int(pd.to_numeric(row['Odds'], errors='coerce'))
                to_win = (100 * (o / 100)) if o > 0 else (100 / (abs(o) / 100)); total_payout = 100 + to_win
                win_prob_val = float(row.get('Win_Prob', 0.55)) * 100
                
                status_color = "#94a3b8"
                if row['Result'] == "Win": status_color = "#00E676"
                elif row['Result'] == "Loss": status_color = "#ff0055"
                elif row['Result'] == "Push": status_color = "#FFD700"
                
                html = f"""<div style="background-color: #0f172a; border-radius: 8px; border: 1px solid #334155; border-left: 6px solid {status_color}; padding: 12px; margin-bottom: 5px;"><div style="display: flex; justify-content: space-between; margin-bottom: 8px;"><span style="font-size: 12px; color: #94a3b8; font-weight: bold; text-transform: uppercase;">{row['League']} • {row['Date']}</span><span style="font-size: 14px; color: #fff; font-weight: bold;">{o:+d}</span></div><div style="font-size: 16px; font-weight: 900; color: #00E5FF;">{row['Player']}</div><div style="font-size: 14px; font-weight: bold; color: #f8fafc; margin-top: 2px;">{row['Stat']} <span style="color: #00E676;">{row['Vote']} {row['Line']}</span></div><div style="margin-top: 10px; border-top: 1px dashed #334155; padding-top: 8px; display: flex; justify-content: space-between;"><span style="font-size: 12px; color: #94a3b8;">Risk: $100.00 &nbsp;|&nbsp; <span style="color: #00E5FF; font-weight: bold;">AI Prob: {win_prob_val:.1f}%</span></span><span style="font-size: 12px; font-weight: bold; color: {status_color};">Payout: ${total_payout:.2f}</span></div></div>"""
                sc1, sc2 = st.columns([4, 1])
                with sc1:
                    st.markdown(html, unsafe_allow_html=True)
                    if row['Result'] == 'Loss':
                        if st.button("🔍 Run AI Autopsy", key=f"run_auto_{orig_idx}"):
                            with st.spinner("Analyzing game logs..."):
                                autopsy_result = generate_ai_autopsy(row['League'], row['Player'], row['Stat'], row['Line'], row['Vote'], row['Date'])
                                st.session_state[f"autopsy_{orig_idx}"] = autopsy_result
                        if st.session_state.get(f"autopsy_{orig_idx}"):
                            st.markdown(f"""<div style="background-color: rgba(255, 0, 85, 0.1); border-left: 3px solid #ff0055; padding: 10px; margin-top: -5px; margin-bottom: 10px; font-size: 13px; color: #f8fafc;">{st.session_state[f"autopsy_{orig_idx}"]}</div>""", unsafe_allow_html=True)
                with sc2:
                    st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)
                    options = ["Pending", "Win", "Loss", "Push"]
                    curr_idx = options.index(row['Result']) if row['Result'] in options else 0
                    res = st.selectbox("Grade", options, index=curr_idx, key=f"s_res_{orig_idx}", label_visibility="collapsed")
                    new_results[orig_idx] = res
                    
            if st.button("💾 Save All Single Grades", type="primary", use_container_width=True):
                for orig_idx, res in new_results.items(): ledger_df.at[orig_idx, 'Result'] = res
                overwrite_sheet("ROI_Ledger", ledger_df)
                st.success("Ledger Updated in Cloud!"); time.sleep(1); st.rerun()

    elif roi_mode == "🎟️ Parlay Tracker":
        st.markdown("### 🎟️ Parlay Tracker")
        st.caption("Calculate profit based on custom stakes and boosted odds.")
        single_df = load_ledger()
        pick_options, pick_odds_map, pick_prob_map = [], {}, {}
        
        if not single_df.empty:
            pending_singles = single_df[single_df['Result'] == 'Pending']
            for idx, row in pending_singles.iterrows():
                desc_str = f"{row['Player']} - {row['Stat']} {row['Vote']} {row['Line']}"
                pick_options.append(desc_str)
                pick_odds_map[desc_str] = float(row['Odds'])
                pick_prob_map[desc_str] = float(row.get('Win_Prob', 0.55))
        
        selected_picks = st.multiselect("🔗 Link Pending Picks into a Ticket", pick_options)
        default_desc = " + ".join(selected_picks) if selected_picks else ""
        
        calc_dec, c_prob = 1.0, 1.0 
        for p in selected_picks:
            o = pick_odds_map[p]
            calc_dec *= ((o / 100.0) + 1.0) if o > 0 else ((100.0 / abs(o)) + 1.0)
            c_prob *= pick_prob_map.get(p, 0.55)
                
        if selected_picks:
            true_american = int(round((calc_dec - 1.0) * 100)) if calc_dec >= 2.0 else int(round(-100.0 / (calc_dec - 1.0)))
            def_prob = min(99.9, c_prob * 100) 
        else:
            true_american, def_prob = 150, 55.0 

        with st.expander("📘 Open Bankroll Advisor (Calculate Bet Size)", expanded=False):
            liq_bal = get_liquid_balance()
            st.markdown(f"**Live Bankroll:** ${liq_bal:.2f}")
            strat_toggle = st.radio("Select Strategy", ["🔥 Micro-Aggressor (Tiered)", "🤖 True Kelly (AI Math)"], horizontal=True, label_visibility="collapsed")
            
            if "Micro" in strat_toggle:
                if liq_bal < 100: s_rec, p_rec = 5.00, 2.00
                elif liq_bal <= 200: s_rec, p_rec = 10.00, 4.00
                elif liq_bal <= 500: s_rec, p_rec = 15.00, 5.00
                else: s_rec, p_rec = liq_bal * 0.03, liq_bal * 0.01
                st.markdown(f"""<div style="background-color: #0f172a; padding: 15px; border-radius: 8px; border-left: 4px solid #ff0055; margin-top: 10px;"><div style="display: flex; justify-content: space-around; text-align: center;"><div><div style="font-size: 12px; color: #94a3b8;">Standard Single Wager</div><div style="font-size: 24px; font-weight: 900; color: #ff0055;">${s_rec:.2f}</div></div><div><div style="font-size: 12px; color: #94a3b8;">Standard Parlay Risk</div><div style="font-size: 24px; font-weight: 900; color: #00E5FF;">${p_rec:.2f}</div></div></div></div>""", unsafe_allow_html=True)
            else:
                kc1, kc2 = st.columns(2)
                k_prob = kc1.number_input("Est. Win Prob (%)", min_value=0.1, max_value=99.9, value=float(def_prob), step=1.0)
                k_odds = kc2.number_input("Bet Odds", value=true_american, step=10)
                win_prob_dec = k_prob / 100.0
                b_odds = (100 / abs(k_odds)) if k_odds < 0 else (k_odds / 100)
                k_pct = (b_odds * win_prob_dec - (1 - win_prob_dec)) / b_odds if b_odds > 0 else 0
                k_pct = max(0.0, k_pct)
                s_rec = liq_bal * (k_pct * 0.5) 
                st.markdown(f"""<div style="background-color: #0f172a; padding: 15px; border-radius: 8px; border-left: 4px solid #00E676; margin-top: 10px; text-align: center;"><div style="font-size: 12px; color: #94a3b8;">Recommended Kelly Stake (Half-Kelly)</div><div style="font-size: 24px; font-weight: 900; color: #00E676;">${s_rec:.2f}</div></div>""", unsafe_allow_html=True)
        
        p_col1, p_col2, p_col3, p_col4 = st.columns([2.5, 1, 1, 1.5])
        with p_col1: p_desc = st.text_area("Bet Description", value=default_desc, height=68)
        with p_col2: p_odds = st.number_input("Final Odds (w/ Boosts)", value=true_american, step=10)
        with p_col3: p_risk = st.number_input("Risk ($)", value=10.0, step=5.0)
        with p_col4: p_book = st.selectbox("Sportsbook", SPORTSBOOKS); p_free = st.checkbox("🆓 Free Bet")
            
        if p_odds != 0: proj_profit = p_risk * (p_odds / 100) if p_odds > 0 else p_risk / (abs(p_odds) / 100)
        else: proj_profit = 0.0
            
        st.info(f"💸 **Projected Payout:** ${(proj_profit if p_free else p_risk + proj_profit):.2f} (Profit: ${proj_profit:.2f})")
        
        if st.button("➕ Add Bet to Tracker", type="primary"):
            if p_desc:
                save_to_parlay_ledger(p_desc, p_odds, p_risk, p_book, p_free)
                st.success("Bet Added to Cloud DB!"); time.sleep(1.0); st.rerun()
            else: st.error("Please enter a description or link your picks above.")

        parlay_df = load_parlay_ledger()
        if not parlay_df.empty:
            st.markdown("---")
            graded_p = parlay_df[parlay_df['Result'].isin(['Win', 'Loss'])]
            p_wins = len(graded_p[graded_p['Result'] == 'Win'])
            p_total = len(graded_p)
            
            p_profit, total_staked = 0.0, 0.0
            for _, row in graded_p.iterrows():
                o = pd.to_numeric(row['Odds'], errors='coerce')
                r = pd.to_numeric(row['Risk'], errors='coerce')
                is_f = row.get('Is_Free_Bet', False)
                if not is_f: total_staked += r
                
                if row['Result'] == 'Win': p_profit += (r * (o / 100)) if o > 0 else (r / (abs(o) / 100))
                else: p_profit -= (0 if is_f else r)
            
            p_roi = (p_profit / total_staked * 100) if p_total > 0 and total_staked > 0 else 0.0
            p_win_pct = (p_wins / p_total * 100) if p_total > 0 else 0.0
            
            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Total Graded Live/Parlays", f"{p_total}")
            pm2.metric("Win Rate", f"{p_win_pct:.1f}%")
            pm3.metric("Net Profit", f"${p_profit:+.2f}")
            pm4.metric("ROI (%)", f"{p_roi:+.1f}%")

            st.markdown("#### 🎫 Your Live / Parlay Slips")
            new_p_results = {}
            reversed_p_df = parlay_df.iloc[::-1].reset_index()
            for i, row in reversed_p_df.iterrows():
                orig_idx = row['index']
                o = int(pd.to_numeric(row['Odds'], errors='coerce')); r = float(pd.to_numeric(row['Risk'], errors='coerce'))
                is_f = row.get('Is_Free_Bet', False)
                to_win = (r * (o / 100)) if o > 0 else (r / (abs(o) / 100))
                total_payout = to_win if is_f else r + to_win
                
                status_color = "#94a3b8"
                if row['Result'] == "Win": status_color = "#00E676"
                elif row['Result'] == "Loss": status_color = "#ff0055"
                elif row['Result'] == "Push": status_color = "#FFD700"
                
                legs = str(row['Description']).split(" + ")
                legs_html = "".join([f"<div style='margin-bottom: 4px;'>🎟️ {leg}</div>" for leg in legs])
                risk_txt = f"🆓 FREE BET: ${r:.2f}" if is_f else f"Risk: ${r:.2f}"
                book_lbl = row.get('Sportsbook', 'LIVE BET').upper()
                
                html = f"""<div style="background-color: #0f172a; border-radius: 8px; border: 1px solid #334155; border-left: 6px solid {status_color}; padding: 12px; margin-bottom: 5px;"><div style="display: flex; justify-content: space-between; margin-bottom: 8px;"><span style="font-size: 12px; color: #94a3b8; font-weight: bold; letter-spacing: 1px;">{book_lbl} • {row['Date']}</span><span style="font-size: 14px; color: #fff; font-weight: bold;">{o:+d}</span></div><div style="font-size: 13px; color: #f8fafc; margin-bottom: 10px; line-height: 1.5;">{legs_html}</div><div style="margin-top: 10px; border-top: 1px dashed #334155; padding-top: 8px; display: flex; justify-content: space-between;"><span style="font-size: 12px; color: #94a3b8;">{risk_txt}</span><span style="font-size: 12px; font-weight: bold; color: {status_color};">Payout: ${total_payout:.2f}</span></div></div>"""
                pc1, pc2 = st.columns([4, 1])
                with pc1: st.markdown(html, unsafe_allow_html=True)
                with pc2:
                    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
                    options = ["Pending", "Win", "Loss", "Push"]
                    curr_idx = options.index(row['Result']) if row['Result'] in options else 0
                    res = st.selectbox("Grade", options, index=curr_idx, key=f"p_res_{orig_idx}", label_visibility="collapsed")
                    new_p_results[orig_idx] = res
                    
            if st.button("💾 Save All Live/Parlay Grades", type="primary", use_container_width=True):
                for orig_idx, res in new_p_results.items(): parlay_df.at[orig_idx, 'Result'] = res
                overwrite_sheet("Parlay_Ledger", parlay_df)
                st.success("Tracker Updated in Cloud!"); time.sleep(1); st.rerun()
