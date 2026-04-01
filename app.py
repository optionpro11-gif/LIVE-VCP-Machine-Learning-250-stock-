import os
import pickle
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VCP ML Unified Command Center",
    page_icon="🏛️",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(__file__)
MODEL_FILE    = os.path.join(BASE_DIR, "vcp_xgb_model.pkl")
PARQUET_FILE  = os.path.join(BASE_DIR, "vcp_features_labeled.parquet")
PLAYBOOK_FILE = os.path.join(BASE_DIR, "SWING_STRATEGY_PLAYBOOK.md")

TICKERS_250 = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","TCS.NS","INFY.NS",
    "BHARTIARTL.NS","SBIN.NS","LICI.NS","HINDUNILVR.NS","ITC.NS",
    "TATAMOTORS.NS","ADANIENT.NS","TITAN.NS","BAJFINANCE.NS","SUNPHARMA.NS",
    "AXISBANK.NS","KOTAKBANK.NS","MARUTI.NS","ULTRACEMCO.NS","POWERGRID.NS",
    "NTPC.NS","ONGC.NS","LT.NS","ASIANPAINT.NS","NESTLEIND.NS",
    "WIPRO.NS","COALINDIA.NS","ADANIPORTS.NS","HCLTECH.NS","BAJAJFINSV.NS",
    "IOC.NS","DMART.NS","GRASIM.NS","CIPLA.NS","JSWSTEEL.NS",
    "DRREDDY.NS","TATACONSUM.NS","BAJAJ-AUTO.NS","BPCL.NS","TECHM.NS",
    "SIEMENS.NS","HAVELLS.NS","PIDILITIND.NS","GODREJCP.NS","DABUR.NS",
    "COLPAL.NS","MARICO.NS","BERGEPAINT.NS","AMBUJACEM.NS","SHREECEM.NS",
    "GLAND.NS","BIOCON.NS","AUROPHARMA.NS","DIVISLAB.NS","ALKEM.NS",
    "TORNTPHARM.NS","LUPIN.NS","MPHASIS.NS","LTIM.NS","PERSISTENT.NS",
    "COFORGE.NS","OFSS.NS","HDFCLIFE.NS","SBILIFE.NS","ICICIGI.NS",
    "BAJAJHLDNG.NS","CHOLAFIN.NS","MUTHOOTFIN.NS","RECLTD.NS","PFC.NS",
    "IRFC.NS","NHPC.NS","SJVN.NS","TATAPOWER.NS","TORNTPOWER.NS",
    "JINDALSTEL.NS","SAIL.NS","NMDC.NS","HINDALCO.NS","VEDL.NS",
    "NATIONALUM.NS","HINDCOPPER.NS","BALKRISIND.NS","MRF.NS","APOLLOTYRE.NS",
    "ESCORTS.NS","MOTHERSON.NS","BOSCHLTD.NS","TVSMOTOR.NS","HEROMOTOCO.NS",
    "EICHERMOT.NS","ASHOKLEY.NS","INDUSINDBK.NS","FEDERALBNK.NS","IDFCFIRSTB.NS",
    "RBLBANK.NS","BANDHANBNK.NS","PIIND.NS","UPL.NS","DEEPAKNTR.NS",
    "NAVINFLUOR.NS","SRF.NS","ATUL.NS","ANGELONE.NS","BSE.NS","MCX.NS",
    "CDSL.NS","IEX.NS","IRCTC.NS","HUDCO.NS","RVNL.NS","BEL.NS",
    "HAL.NS","BDL.NS","COCHINSHIP.NS","HAPPSTMNDS.NS","LTTS.NS",
    "KPITTECH.NS","TATAELXSI.NS","CYIENT.NS","SONACOMS.NS","IPCALAB.NS",
    "GLENMARK.NS","METROPOLIS.NS","APOLLOHOSP.NS","INDIAMART.NS","NAUKRI.NS",
    "ZOMATO.NS","JUBLFOOD.NS","TRENT.NS","PAGEIND.NS","MANYAVAR.NS",
    "BATAINDIA.NS","KPRMILL.NS","RAYMOND.NS","POLYCAB.NS","FINOLEX.NS",
    "KEI.NS","PRESTIGE.NS","OBEROIRLTY.NS","GODREJPROP.NS","DLF.NS",
    "HGINFRA.NS","KNRCON.NS","PNCINFRA.NS","IRCON.NS","NBCC.NS",
    "APLAPOLLO.NS","WELCORP.NS","GRINDWELL.NS","CUMMINSIND.NS","THERMAX.NS",
    "BHEL.NS","ABB.NS","VOLTAS.NS","BLUESTARCO.NS","DIXON.NS",
    "CROMPTON.NS","RAMCOCEM.NS","JKCEMENT.NS","ACC.NS","LAURUSLABS.NS",
    "ERIS.NS","GRANULES.NS","GAIL.NS","PETRONET.NS","IGL.NS","MGL.NS",
    "GUJARATGAS.NS","ATGL.NS"
]
TICKERS_250 = list(dict.fromkeys(TICKERS_250))


# ─────────────────────────────────────────────────────────────────────────────
# SHARED RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_shared_resources():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(PARQUET_FILE):
        return None, None, None, None
        
    with open(MODEL_FILE, "rb") as f:
        bundle = pickle.load(f)
    
    df = pd.read_parquet(PARQUET_FILE)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    unique_dates = sorted(df["date"].unique())
    
    return bundle["model"], bundle["features"], df, unique_dates

model_obj, features_obj, hist_df, all_dates = load_shared_resources()

if model_obj is None:
    st.error("Resources missing. Ensure ml_vcp data and models exist.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# LIVE MACHINE LEARNING ENGINE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def safe_last(series):
    vals = series.dropna()
    return float(vals.iloc[-1]) if len(vals) > 0 else np.nan

def compute_live_features(raw: pd.DataFrame, bench: pd.Series) -> dict:
    c = raw["Close"].squeeze()
    h = raw["High"].squeeze()
    l = raw["Low"].squeeze()
    v = raw["Volume"].squeeze()
    bench = bench.reindex(c.index).ffill()

    r1  = c.pct_change(1).iloc[-1]  * 100
    r5  = c.pct_change(5).iloc[-1]  * 100
    r21 = c.pct_change(21).iloc[-1] * 100
    r63 = c.pct_change(63).iloc[-1] * 100

    b100 = bench.shift(100).replace(0, np.nan)
    c100 = c.shift(100).replace(0, np.nan)
    rs   = safe_last((c / bench) / (c100 / b100) * 100)

    sma20    = c.rolling(20).mean()
    std20    = c.rolling(20).std()
    bb_upper = sma20 + 2*std20
    bb_lower = sma20 - 2*std20
    bbw      = ((bb_upper - bb_lower) / sma20.replace(0, np.nan) * 100)
    bbw_pctl = safe_last(bbw.rolling(50).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]*100, raw=False))

    period   = 14
    hdiff    = h.diff(); ldiff = -l.diff()
    plus_dm  = np.where((hdiff > ldiff) & (hdiff > 0), hdiff, 0.0)
    minus_dm = np.where((ldiff > hdiff) & (ldiff > 0), ldiff, 0.0)
    tr_val   = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr14    = tr_val.ewm(alpha=1/period, adjust=False).mean()
    plus_di  = 100*pd.Series(plus_dm, index=c.index).ewm(alpha=1/period, adjust=False).mean()/atr14.replace(0, np.nan)
    minus_di = 100*pd.Series(minus_dm, index=c.index).ewm(alpha=1/period, adjust=False).mean()/atr14.replace(0, np.nan)
    dx       = 100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0, np.nan)
    adx      = safe_last(dx.ewm(alpha=1/period, adjust=False).mean())

    vol_ma20 = v.rolling(20).mean()
    vol_r    = safe_last(v / vol_ma20.replace(0, np.nan))

    e20  = c.ewm(span=20,  adjust=False).mean()
    e50  = c.ewm(span=50,  adjust=False).mean()
    e150 = c.ewm(span=150, adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()
    last_c = float(c.iloc[-1])
    trend  = 1.0 if (last_c>e20.iloc[-1] and e20.iloc[-1]>e50.iloc[-1] and e50.iloc[-1]>e150.iloc[-1] and e150.iloc[-1]>e200.iloc[-1]) else 0.5 if last_c>e50.iloc[-1] else 0.0

    atr_pct = safe_last(atr14 / c.replace(0, np.nan) * 100)
    hi52   = h.rolling(252).max()
    dist52 = safe_last((1 - c/hi52.replace(0, np.nan)) * 100)

    r1_vol = (h.rolling(10).max() - l.rolling(10).min()) / c.replace(0, np.nan) * 100
    r2_vol = (h.shift(10).rolling(10).max() - l.shift(10).rolling(10).min()) / c.replace(0, np.nan) * 100
    r3_vol = (h.shift(20).rolling(10).max() - l.shift(20).rolling(10).min()) / c.replace(0, np.nan) * 100
    r1v, r2v, r3v = r1_vol.iloc[-1], r2_vol.iloc[-1], r3_vol.iloc[-1]
    tight  = 3.0 if (r1v < r2v and r2v < r3v) else 2.0 if r1v < r2v else 1.0

    score  = (100 - bbw_pctl) * 0.35
    score += 20.0 if rs > 105 else 10.0 if rs > 100 else 0.0
    score += 15.0 if trend == 1.0 else 5.0 if trend == 0.5 else 0.0
    score += 10.0 if vol_r < 0.8 else 0.0
    score += 10.0 if dist52 < 5 else 5.0 if dist52 < 15 else 0.0
    score += 10.0 if tight >= 2 else 0.0

    hi52_val = float(hi52.iloc[-1]) if not np.isnan(hi52.iloc[-1]) else last_c
    in_base = (c >= hi52_val * 0.85).astype(int)
    consec = in_base * (in_base.groupby((in_base != in_base.shift()).cumsum()).cumcount() + 1)
    wbase = float(consec.iloc[-1]) / 5.0

    kc_upper = sma20 + 1.5*atr14; kc_lower = sma20 - 1.5*atr14
    sqz = 1.0 if (bb_lower.iloc[-1] > kc_lower.iloc[-1] and bb_upper.iloc[-1] < kc_upper.iloc[-1]) else 0.0

    e200_rising = float(e200.iloc[-1]) > float(e200.iloc[-11]) if len(e200) > 11 else False
    stage = 2.0 if last_c>e150.iloc[-1] and last_c>e200.iloc[-1] and e200_rising else 3.0 if last_c<e150.iloc[-1] and last_c>e200.iloc[-1] else 4.0 if last_c<e150.iloc[-1] and last_c<e200.iloc[-1] else 1.0

    vol_dry = (v < vol_ma20).astype(int)
    vdry = float(vol_dry.groupby((vol_dry != vol_dry.shift()).cumsum()).cumcount().iloc[-1] * vol_dry.iloc[-1])

    high_10 = h.shift(1).rolling(10).max()
    low_10  = l.rolling(10).min()
    hndl    = safe_last((high_10 - low_10) / high_10.replace(0, np.nan) * 100)

    chk  = int(rs>100) + int(bbw_pctl<25) + int(vol_r<1.0) + int(tight>=2) + int(dist52<15) + int(adx>20) + int(trend==1.0)
    signal = 2 if score>=70 else 1 if score>=50 else 0
    spark_dir = np.sign(last_c - float(c.iloc[-6])) if len(c) >= 6 else 0.0
    price_e20_pct = (last_c - e20.iloc[-1]) / e20.iloc[-1] * 100 if e20.iloc[-1] != 0 else 0.0
    pdh = float(h.iloc[-2]) if len(h) >= 2 else last_c

    return {
        "r1": float(r1), "r5": float(r5), "r21": float(r21), "r63": float(r63),
        "rs": float(rs), "rs_rank": np.nan, "bbw_pctl": float(bbw_pctl), "adx": float(adx), 
        "vol_r": float(vol_r), "trend": trend, "atr_pct": float(atr_pct), "dist52": float(dist52),
        "tight": tight, "score": float(score), "wbase": float(wbase), "sqz": sqz, "stage": stage, 
        "vdry": float(vdry), "hndl": float(hndl), "chk": float(chk), "signal": float(signal),
        "spark_dir": float(spark_dir), "price_e20_pct": float(price_e20_pct),
        "_price": last_c, "_pdh": pdh, "_atr_abs": float(atr14.iloc[-1])
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR - UNIVERSAL RISK CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎮 Universal Controls")
    st.divider()
    
    st.subheader("🛡️ Risk Parameters")
    sl_pct = st.slider("Stop Loss (%)", 2.0, 15.0, 5.0, 0.5)
    target_pct = st.slider("Profit Target (%)", 5.0, 50.0, 10.0, 1.0)
    time_limit = st.slider("Time Exit (Days)", 5, 63, 12)
    
    st.divider()
    
    st.subheader("🤖 ML Settings")
    ml_thresh = st.slider("ML Probability Minimum", 0.40, 0.90, 0.65, 0.01)
    max_slots = st.number_input("Max Portfolio Slots", 5, 40, 10)
    
    st.divider()
    if st.button("🔄 Reset Global State"):
        st.session_state.clear()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE (SIMULATOR)
# ─────────────────────────────────────────────────────────────────────────────
if "sim_init" not in st.session_state:
    st.session_state.sim_init = True
    st.session_state.cash = 1000000.0
    st.session_state.holdings = []
    st.session_state.closed_trades = []
    st.session_state.equity_curve = []
    st.session_state.step_idx = 0
    st.session_state.pos_size = 100000.0 
    st.session_state.snapshots = []
    st.session_state.daily_events = []
    st.session_state.sim_running = False

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_live, tab_hist, tab_playbook = st.tabs(["🚀 Live Scanner", "🕒 Portfolio Simulator", "📘 Strategy Playbook"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: LIVE SCANNER
# ─────────────────────────────────────────────────────────────────────────────
with tab_live:
    st.subheader("🚀 Live Nifty 250 Breakout Scanner")
    st.write("Detects high-conviction VCP setups using real-time Yahoo Finance data.")
    
    if st.button("🔥 Run Live Scan Now", type="primary"):
        end   = datetime.now()
        start = end - timedelta(days=400)
        start_str, end_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

        with st.spinner("Fetching Live Data & Calculating 27-Feature ML Array..."):
            try:
                bench_df = flatten_df(yf.download("^NSEI", start=start_str, end=end_str, progress=False, auto_adjust=True))
                bench_close = bench_df["Close"].squeeze()
            except Exception as e:
                st.error(f"Failed to fetch benchmark: {e}")
                bench_close = pd.Series()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            rows = []
            for i, sym in enumerate(TICKERS_250):
                status_text.text(f"Scanning {sym} ({i+1}/{len(TICKERS_250)})...")
                progress_bar.progress((i + 1) / len(TICKERS_250))
                try:
                    raw = flatten_df(yf.download(sym, start=start_str, end=end_str, progress=False, auto_adjust=True))
                    if len(raw) > 200:
                        feats = compute_live_features(raw, bench_close)
                        feats["symbol"] = sym.replace(".NS", "")
                        rows.append(feats)
                except:
                    pass
                
            status_text.text("Scan complete! Processing ML inference...")
            
            if rows:
                df_live = pd.DataFrame(rows)
                df_live["rs_rank"] = df_live["rs"].rank(pct=True) * 100
                
                feat_vals = df_live[features_obj].values
                col_medians = pd.DataFrame(feat_vals, columns=features_obj).median()
                feat_vals = pd.DataFrame(feat_vals, columns=features_obj).fillna(col_medians).values
                
                df_live["ml_prob"] = model_obj.predict_proba(feat_vals)[:, 1]
                df_live["pdh_break"] = df_live["_price"] > df_live["_pdh"]
                df_live["stop_loss"] = df_live["_price"] - 1.5 * df_live["_atr_abs"]
                df_live["target"] = df_live["_price"] * 1.15
                df_live["rr_ratio"] = (df_live["target"] - df_live["_price"]) / (df_live["_price"] - df_live["stop_loss"]).replace(0, np.nan)
                
                picks = df_live[
                    (df_live["ml_prob"] >= ml_thresh) & (df_live["score"] >= 60) & (df_live["stage"] == 2.0)
                ].sort_values("ml_prob", ascending=False).head(10)
                
                st.markdown("---")
                if picks.empty:
                    st.warning(f"No high-conviction setups found matching ML threshold {ml_thresh}.")
                else:
                    for rank, (_, row) in enumerate(picks.iterrows(), 1):
                        with st.container():
                            c1, c2, c3, c4 = st.columns(4)
                            pdh = "🔥 YES" if row["pdh_break"] else "⏳ Wait"
                            c1.metric(f"#{rank} {row['symbol']}", f"₹{row['_price']:.1f}", f"Target: ₹{row['target']:.1f}")
                            c2.metric("ML Probability", f"{row['ml_prob']*100:.1f}%", f"Score: {row['score']:.0f}/100")
                            c3.metric("Tightness Base", f"{int(row['tight'])}T", f"BBW: {row['bbw_pctl']:.0f}%")
                            c4.metric("Entry Quality", f"R:R = {row['rr_ratio']:.1f}x", f"PDH Break: {pdh}")
                            st.caption(f"**RS:** {row['rs']:.1f} | **ADX:** {row['adx']:.0f} | **CHK:** {int(row['chk'])}/7 | **Stop Loss:** ₹{row['stop_loss']:.1f}")
                            st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: PORTFOLIO SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
with tab_hist:
    st.subheader("🕒 10L Portfolio Time-Stepper")
    
    # --- LOGIC ---
    def run_sim_step(direction="forward"):
        if direction == "backward":
            if st.session_state.snapshots:
                snap = st.session_state.snapshots.pop()
                st.session_state.cash = snap["cash"]
                st.session_state.holdings = snap["holdings"]
                st.session_state.closed_trades = snap["closed_trades"]
                st.session_state.equity_curve = snap["equity_curve"]
                st.session_state.step_idx = snap["step_idx"]
                st.session_state.pos_size = snap["pos_size"]
                st.session_state.daily_events = ["⏮️ Reversed to previous day."]
            return

        if st.session_state.step_idx >= len(all_dates):
            st.session_state.sim_running = False
            return
            
        # Snapshot
        snapshot = {
            "cash": st.session_state.cash,
            "holdings": [h.copy() for h in st.session_state.holdings],
            "closed_trades": [t.copy() for t in st.session_state.closed_trades],
            "equity_curve": [e.copy() for e in st.session_state.equity_curve],
            "step_idx": st.session_state.step_idx,
            "pos_size": st.session_state.pos_size
        }
        st.session_state.snapshots.append(snapshot)
        if len(st.session_state.snapshots) > 100: st.session_state.snapshots.pop(0)

        curr_dt = all_dates[st.session_state.step_idx]
        day_df = hist_df[hist_df["date"] == curr_dt]
        st.session_state.daily_events = []
        
        # 1. Check Exits
        new_h = []
        total_market_val = 0
        for h in st.session_state.holdings:
            ticker_data = day_df[day_df["symbol"] == h["symbol"]]
            if ticker_data.empty:
                new_h.append(h)
                continue
            row = ticker_data.iloc[0]
            cur_p = row["price"]
            low_p = row["low"]
            high_p = row["high"]
            
            days_held = (curr_dt - h["entry_dt"]).days
            low_pnl_pct = (low_p - h["entry_p"]) / h["entry_p"] * 100
            high_pnl_pct = (high_p - h["entry_p"]) / h["entry_p"] * 100
            
            exit_p = None
            exit_reason = ""
            if low_pnl_pct <= -sl_pct: 
                exit_p = h["entry_p"] * (1 - sl_pct/100)
                exit_reason = f"STOP LOSS (-{sl_pct}%)"
            elif high_pnl_pct >= target_pct: 
                exit_p = h["entry_p"] * (1 + target_pct/100)
                exit_reason = f"TARGET (+{target_pct}%)"
            elif days_held >= time_limit: 
                exit_p = row["price"]
                exit_reason = f"TIME EXIT ({time_limit}d)"
            
            if exit_p:
                sell_val = h["qty"] * exit_p
                pnl_val = sell_val - (h["qty"] * h["entry_p"])
                st.session_state.cash += sell_val
                st.session_state.closed_trades.append({
                    "symbol": h["symbol"], "entry_dt": h["entry_dt"], "exit_dt": curr_dt,
                    "entry_p": h["entry_p"], "exit_p": exit_p, "pnl": pnl_val, "reason": exit_reason
                })
                st.session_state.daily_events.append(f"🔴 SOLD {h['symbol']} @ {exit_p:.1f} ({exit_reason})")
            else:
                total_market_val += h["qty"] * cur_p
                new_h.append(h)
                
        st.session_state.holdings = new_h
        
        # 2. Monthly Rebalance Calculation
        if st.session_state.step_idx > 0:
            prev_dt = all_dates[st.session_state.step_idx - 1]
            if curr_dt.month != prev_dt.month:
                total_equity = st.session_state.cash + total_market_val
                st.session_state.pos_size = total_equity / max_slots
                st.session_state.daily_events.append(f"📅 MONTH END - Resetting Pos Size: ₹{st.session_state.pos_size:,.0f}")
        
        # 3. Daily Entry
        if len(st.session_state.holdings) < max_slots:
            feat_vals = day_df[features_obj].values
            if len(feat_vals) > 0:
                col_medians = pd.DataFrame(feat_vals, columns=features_obj).median()
                feat_vals = pd.DataFrame(feat_vals, columns=features_obj).fillna(col_medians).values
                day_df["ml_prob"] = model_obj.predict_proba(feat_vals)[:, 1]
                
                picks = day_df[
                    (day_df["ml_prob"] >= ml_thresh) & (day_df["score"] >= 60) & (day_df["stage"] == 2.0)
                ].sort_values("ml_prob", ascending=False)
                
                held_symbols = [h["symbol"] for h in st.session_state.holdings]
                for _, p in picks.iterrows():
                    if len(st.session_state.holdings) >= max_slots: break
                    if p["symbol"] in held_symbols: continue
                    qty = int(st.session_state.pos_size / p["price"])
                    
                    if qty > 0 and st.session_state.cash >= (qty * p["price"]):
                        cost = qty * p["price"]
                        st.session_state.cash -= cost
                        st.session_state.holdings.append({
                            "symbol": p["symbol"], "qty": qty, "entry_p": p["price"], 
                            "entry_dt": curr_dt, "reason": f"ML:{p['ml_prob']:.1%}, Score:{p['score']:.0f}"
                        })
                        total_market_val += cost
                        st.session_state.daily_events.append(f"🟢 BOUGHT {p['symbol']} @ {p['price']:.1f}")
        
        # 4. Save Equity Point
        st.session_state.equity_curve.append({
            "date": curr_dt, "total": st.session_state.cash + total_market_val,
            "cash": st.session_state.cash, "holdings": total_market_val
        })
        
        st.session_state.step_idx += 1

    # --- UI CONTROLS ---
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("⏭️ Next Day"):
            run_sim_step(direction="forward")
    with col2:
        if st.button("⏮️ Prev Day"):
            run_sim_step(direction="backward")
    with col3:
        if st.button("▶️ Auto Play"):
            st.session_state.sim_running = True
    with col4:
        if st.button("⏸️ Stop"):
            st.session_state.sim_running = False
    with col5:
        jump_date = st.date_input("Jump To Date", all_dates[0].date(), min_value=all_dates[0].date(), max_value=all_dates[-1].date())
        if st.button("🚀 Jump"):
            st.session_state.step_idx = next(i for i, d in enumerate(all_dates) if d >= pd.to_datetime(jump_date))
            st.session_state.snapshots = []
            st.toast(f"Jumped to {jump_date}")
            
    # --- METRICS & ALERTS ---
    curr_date_val = all_dates[min(st.session_state.step_idx, len(all_dates)-1)]
    st.write(f"Date: **{curr_date_val.date()}** | Position Size: **₹{st.session_state.pos_size:,.0f}**")
    
    current_date_dt = curr_date_val
    day_df_now = hist_df[hist_df["date"] == current_date_dt]
    
    holdings_val = 0
    for h in st.session_state.holdings:
        ticker_now = day_df_now[day_df_now["symbol"] == h["symbol"]]
        price_now = ticker_now["price"].iloc[0] if not ticker_now.empty else h["entry_p"]
        holdings_val += h["qty"] * price_now
        
    closed_pnl = sum([t["pnl"] for t in st.session_state.closed_trades])
    total_equity = st.session_state.cash + holdings_val
    
    if st.session_state.daily_events:
        with st.expander(f"🔔 Activity Log - {curr_date_val.date()}", expanded=True):
            for ev in st.session_state.daily_events:
                st.write(ev)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Value", f"₹{total_equity:,.0f}", f"{(total_equity/1000000 - 1)*100:.2f}%")
    m2.metric("Cash Balance", f"₹{st.session_state.cash:,.0f}")
    m3.metric("Closed P&L", f"₹{closed_pnl:,.0f}", delta_color="normal")
    m4.metric("Active Slots", f"{len(st.session_state.holdings)} / {max_slots}")
    
    # --- SIMULATOR TABS ---
    sim_tab1, sim_tab2, sim_tab3 = st.tabs(["📂 Active Portfolio", "📈 Equity Curve", "📝 Closed Trades"])
    
    with sim_tab1:
        if st.session_state.holdings:
            rows = []
            for h in st.session_state.holdings:
                ticker_now = hist_df[(hist_df["date"] == current_date_dt) & (hist_df["symbol"] == h["symbol"])]
                price_now = ticker_now["price"].iloc[0] if not ticker_now.empty else h["entry_p"]
                pnl_val = (price_now - h["entry_p"]) * h["qty"]
                pnl_pct = (price_now - h["entry_p"]) / h["entry_p"] * 100
                rows.append({
                    "Symbol": h["symbol"],
                    "Qty": h["qty"],
                    "Entry Price": h["entry_p"],
                    "Current Price": price_now,
                    "P&L (₹)": pnl_val,
                    "P&L %": pnl_pct,
                    "Reason/Stats": h["reason"]
                })
            st.dataframe(pd.DataFrame(rows).style.format({
                "Entry Price": "{:.1f}",
                "Current Price": "{:.1f}",
                "P&L (₹)": "{:,.0f}",
                "P&L %": "{:.2f}%"
            }), use_container_width=True)
        else:
            st.info("No active holdings. Start the simulation to buy stocks.")

    with sim_tab2:
        if st.session_state.equity_curve:
            df_curve = pd.DataFrame(st.session_state.equity_curve)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_curve["date"], y=df_curve["total"], mode='lines', name='Total Equity', line=dict(color='cyan')))
            fig.update_layout(title="Portfolio Growth (Initial 10L)", template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Equity curve will appear once the simulation starts.")

    with sim_tab3:
        if st.session_state.closed_trades:
            st.dataframe(pd.DataFrame(st.session_state.closed_trades).sort_values("exit_dt", ascending=False), use_container_width=True)
        else:
            st.info("No closed trades yet.")

    # Simulation Loop Trigger - Need to handle auto play re-run nicely without duplicate calls
    # but Streamlit runs top-down. Auto Play toggle sets session_state.sim_running
    if st.session_state.sim_running:
        run_sim_step(direction="forward")
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: STRATEGY PLAYBOOK
# ─────────────────────────────────────────────────────────────────────────────
with tab_playbook:
    st.subheader("📘 Official VCP Strategy Playbook")
    if os.path.exists(PLAYBOOK_FILE):
        with open(PLAYBOOK_FILE, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.error("Playbook file not found.")

st.sidebar.markdown("---")
st.sidebar.caption("Institutional Trading Engine V3.0 | Unified Command Center")
