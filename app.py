import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from agent import QLearningAgent
from rtp_model import RTPGenerator
from utils import StateUtils, APPLIANCE_MAPPING, ApplianceCategory

# IBM Watson ML Integration
try:
    from ibm_integration import is_ibm_available, get_ibm_recommendation
    IBM_AVAILABLE = is_ibm_available()
except ImportError:
    IBM_AVAILABLE = False
    def get_ibm_recommendation(state): return None

# Page Config
st.set_page_config(page_title="Smart Energy Optimizer", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
MODEL_PATH = "q_table.pkl"

# Custom CSS
st.markdown("""
<style>
    .main .block-container { padding: 0.5rem 1.5rem; }
    [data-testid="stSidebar"] { display: none; }
    .header-box { background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%); padding: 0.8rem 1.5rem; border-radius: 10px; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; }
    .header-title { color: white; font-size: 1.4rem; font-weight: bold; margin: 0; }
    .header-sub { color: #6c757d; font-size: 0.75rem; margin: 0; }
    .card { background: #1e1e2e; border-radius: 8px; padding: 0.8rem; margin-bottom: 0.5rem; border: 1px solid #2d2d44; }
    .card-title { color: #a0a0a0; font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.3rem; }
    .card-value { color: white; font-size: 1.3rem; font-weight: bold; }
    .card-value.green { color: #38ef7d; }
    .card-value.red { color: #f5576c; }
    .section-title { font-size: 0.9rem; font-weight: bold; margin-bottom: 0.5rem; color: #ddd; }
</style>
""", unsafe_allow_html=True)

# Helpers
def get_slot_label(idx): return ["🌙 Night", "🌅 Early", "☀️ Morning", "🌤️ Afternoon", "🌇 Evening", "🌃 Late"][idx]
def get_slot_short(idx): return ["0-4h", "4-8h", "8-12h", "12-16h", "16-20h", "20-24h"][idx]

# Session State
if 'appliances' not in st.session_state: st.session_state['appliances'] = []
if 'results' not in st.session_state: st.session_state['results'] = None

# ===== HEADER =====
st.markdown("""
<div class="header-box">
    <div>
        <p class="header-title">⚡ Smart Energy Optimizer</p>
        <p class="header-sub">AI-Powered Household Energy Management</p>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="color: #38ef7d; font-size: 0.7rem;">Powered by</span>
        <img src="https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg" style="height: 18px; filter: brightness(0) invert(1);">
        <span style="color: white; font-size: 0.8rem; font-weight: bold;">Watson ML</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Generate RTP
date_input = st.date_input("", value=pd.to_datetime("2023-12-01"), label_visibility="collapsed", key="date_hidden")
date_str = date_input.strftime("%Y-%m-%d")
rtp_gen = RTPGenerator()
rtp_profile = rtp_gen.get_prices(date_str)
min_p, max_p = min(rtp_profile), max(rtp_profile)
household_size = 4
temp_input = 20.0
use_ibm = IBM_AVAILABLE

# ===== ROW 1: Add Appliance + Appliances List + Results =====
col1, col2, col3 = st.columns([1.2, 1.5, 1.3])

# ----- COLUMN 1: Add Appliance -----
with col1:
    st.markdown('<p class="section-title">➕ Add Appliance</p>', unsafe_allow_html=True)
    with st.form("add_form", clear_on_submit=True):
        app_name = st.selectbox("Type", list(APPLIANCE_MAPPING.keys()))
        c1, c2 = st.columns(2)
        power = c1.number_input("Power (W)", min_value=10, value=1000, step=100)
        duration = c2.number_input("Hours", min_value=0.1, value=1.0, step=0.5)
        base_time = st.time_input("Usual Time", value=pd.to_datetime("18:00").time())
        
        if st.form_submit_button("➕ Add", use_container_width=True):
            energy = StateUtils.calculate_kwh(power, duration)
            slot = StateUtils.time_to_slot(base_time.hour)
            is_flex = APPLIANCE_MAPPING.get(app_name) == ApplianceCategory.ELASTIC
            st.session_state['appliances'].append({
                "name": app_name, "energy": energy, "base_time": base_time,
                "base_slot": slot, "is_flexible": is_flex
            })
            st.rerun()

# ----- COLUMN 2: Appliances List -----
with col2:
    st.markdown('<p class="section-title">🔌 Your Appliances</p>', unsafe_allow_html=True)
    if st.session_state['appliances']:
        df_apps = pd.DataFrame([{
            "Appliance": a['name'], "kWh": f"{a['energy']:.1f}", 
            "Time": a['base_time'].strftime("%H:%M"),
            "Flex": "✅" if a['is_flexible'] else "🔒"
        } for a in st.session_state['appliances']])
        st.dataframe(df_apps, use_container_width=True, hide_index=True, height=180)
        
        bc1, bc2 = st.columns(2)
        run_opt = bc1.button("🚀 OPTIMIZE", type="primary", use_container_width=True)
        if bc2.button("🗑️ Clear", use_container_width=True):
            st.session_state['appliances'] = []
            st.session_state['results'] = None
            st.rerun()
        
        # Optimization Logic
        if run_opt:
            agent = QLearningAgent()
            if os.path.exists(MODEL_PATH): agent.load_model(MODEL_PATH)
            
            total_base, total_opt, details = 0, 0, []
            for item in st.session_state['appliances']:
                s_slot = 0 if item['is_flexible'] else item['base_slot']
                e_slot = 5 if item['is_flexible'] else item['base_slot']
                state = StateUtils.discretize_state(item['name'], item['energy'], temp_input, household_size, rtp_profile, item['is_flexible'], s_slot, e_slot)
                
                rec = get_ibm_recommendation(state) if use_ibm else None
                if rec is None: rec = agent.choose_action(state, force_greedy=True) if os.path.exists(MODEL_PATH) else rtp_profile.index(min_p)
                
                b_cost = item['energy'] * rtp_profile[item['base_slot']]
                o_cost = item['energy'] * rtp_profile[rec]
                total_base += b_cost
                total_opt += o_cost
                details.append({"Appliance": item['name'], "From": get_slot_short(item['base_slot']), "To": get_slot_short(rec), "b_cost": b_cost, "o_cost": o_cost, "save": b_cost - o_cost})
            
            st.session_state['results'] = {'base': total_base, 'opt': total_opt, 'details': details}
            st.rerun()
    else:
        st.info("Add appliances to get started")
        run_opt = False

# ----- COLUMN 3: Results -----
with col3:
    st.markdown('<p class="section-title">📈 Optimization Results</p>', unsafe_allow_html=True)
    if st.session_state['results']:
        res = st.session_state['results']
        savings = res['base'] - res['opt']
        pct = (savings / res['base'] * 100) if res['base'] > 0 else 0
        
        # Metric Cards
        mc1, mc2, mc3 = st.columns(3)
        mc1.markdown(f'<div class="card"><div class="card-title">Traditional</div><div class="card-value red">₹{res["base"]:.0f}</div></div>', unsafe_allow_html=True)
        mc2.markdown(f'<div class="card"><div class="card-title">Optimized</div><div class="card-value">₹{res["opt"]:.0f}</div></div>', unsafe_allow_html=True)
        mc3.markdown(f'<div class="card"><div class="card-title">Savings</div><div class="card-value green">₹{savings:.0f}</div></div>', unsafe_allow_html=True)
        
        # Schedule Table 
        df_res = pd.DataFrame(res['details'])
        if 'From' in df_res.columns:
            display_df = df_res[['Appliance', 'From', 'To', 'save']].rename(columns={'save': 'Savings ₹'})
        else:
            display_df = df_res
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=150)
    else:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 2rem;">
            <div style="font-size: 2rem;">📊</div>
            <p style="color: #666; margin: 0.5rem 0 0 0;">Click OPTIMIZE to see results</p>
        </div>
        """, unsafe_allow_html=True)

# ===== ROW 2: CHARTS (RTP + Comparison) =====
st.markdown("---")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown('<p class="section-title">📊 Today\'s RTP Prices (₹/kWh)</p>', unsafe_allow_html=True)
    fig_rtp = go.Figure(go.Bar(
        x=[get_slot_short(i) for i in range(6)], y=rtp_profile,
        marker_color=['#38ef7d' if p==min_p else '#f5576c' if p==max_p else '#667eea' for p in rtp_profile],
        text=[f"₹{p:.1f}" for p in rtp_profile], textposition='outside'
    ))
    fig_rtp.update_layout(height=200, margin=dict(l=10,r=10,t=20,b=30), 
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        yaxis=dict(title="₹/kWh", gridcolor='#333'), xaxis=dict(title="Time Slot"))
    st.plotly_chart(fig_rtp, use_container_width=True, config={'displayModeBar': False})

with chart_col2:
    st.markdown('<p class="section-title">📈 Cost Comparison (Original vs Optimized)</p>', unsafe_allow_html=True)
    if st.session_state['results']:
        res = st.session_state['results']
        df_res = pd.DataFrame(res['details'])
        
        if 'b_cost' in df_res.columns:
            orig_costs = df_res['b_cost'].tolist()
            opt_costs = df_res['o_cost'].tolist()
        elif 'Original Cost' in df_res.columns:
            orig_costs = df_res['Original Cost'].tolist()
            opt_costs = df_res['Optimized Cost'].tolist()
        else:
            orig_costs = [10] * len(res['details'])
            opt_costs = [5] * len(res['details'])
        
        fig = go.Figure(data=[
            go.Bar(name='Original', x=df_res['Appliance'], y=orig_costs, marker_color='#f5576c'),
            go.Bar(name='Optimized', x=df_res['Appliance'], y=opt_costs, marker_color='#38ef7d')
        ])
        fig.update_layout(barmode='group', height=200, margin=dict(l=10,r=10,t=20,b=30),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
            yaxis=dict(title="Cost ₹", gridcolor='#333'), xaxis=dict(title="Appliance"))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 1.5rem; height: 180px; display: flex; align-items: center; justify-content: center;">
            <p style="color: #666; margin: 0;">Optimize to see comparison</p>
        </div>
        """, unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #555; font-size: 0.7rem;">
    🔋 Smart Energy Optimizer | Built with <img src="https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg" style="height: 12px; vertical-align: middle; opacity: 0.6;"> IBM Watson ML
</div>
""", unsafe_allow_html=True)
