
import streamlit as st
import numpy as np
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

# --- Page Config (NO SIDEBAR) ---
st.set_page_config(
    page_title="Smart Energy Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = "q_table.pkl"

# --- Minimal CSS ---
st.markdown("""
<style>
    .main .block-container { padding: 1rem 2rem; max-width: 100%; }
    [data-testid="stSidebar"] { display: none; }
    .dashboard-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 1rem; margin-bottom: 0.5rem;
        border: 1px solid #2a2a4a;
    }
    .stat-value { font-size: 1.8rem; font-weight: bold; color: #38ef7d; }
    .stat-label { font-size: 0.75rem; color: #888; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# --- Helpers ---
def get_slot_label(slot_idx):
    return ["🌙 Night", "🌅 Early AM", "☀️ Morning", "🌤️ Afternoon", "🌇 Evening", "🌃 Late"][slot_idx]

# --- Session State ---
if 'appliances' not in st.session_state: st.session_state['appliances'] = []
if 'results' not in st.session_state: st.session_state['results'] = None

# ==================== HEADER ====================
st.markdown("# ⚡ Smart Energy Optimizer")

# ==================== ROW 1: Add Appliance + Context ====================
col_add, col_context, col_rtp = st.columns([2, 1, 1])

with col_add:
    st.markdown("##### ➕ Add Appliance")
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    with c1:
        app_name = st.selectbox("Type", list(APPLIANCE_MAPPING.keys()), label_visibility="collapsed")
    with c2:
        power_watts = st.number_input("W", min_value=10, value=1000, step=50, label_visibility="collapsed")
    with c3:
        duration_hours = st.number_input("h", min_value=0.1, value=1.0, step=0.5, label_visibility="collapsed")
    with c4:
        base_time = st.time_input("Time", value=pd.to_datetime("18:00").time(), label_visibility="collapsed")
    with c5:
        if st.button("➕ Add", use_container_width=True):
            energy_kwh = StateUtils.calculate_kwh(power_watts, duration_hours)
            baseline_slot = StateUtils.time_to_slot(base_time.hour)
            cat = APPLIANCE_MAPPING.get(app_name)
            is_flexible = (cat == ApplianceCategory.ELASTIC)
            st.session_state['appliances'].append({
                "name": app_name, "energy": energy_kwh, "base_time": base_time,
                "base_slot": baseline_slot, "is_flexible": is_flexible
            })
            st.rerun()

with col_context:
    st.markdown("##### 🏠 Context")
    cx1, cx2 = st.columns(2)
    with cx1:
        date_input = st.date_input("Date", value=pd.to_datetime("2023-12-01"), label_visibility="collapsed")
    with cx2:
        household_size = st.selectbox("HH", [1,2,3,4,5,6], index=3, label_visibility="collapsed")
    temp_input = 20.0

with col_rtp:
    st.markdown("##### 📊 RTP Prices")
    date_str = date_input.strftime("%Y-%m-%d")
    rtp_gen = RTPGenerator()
    rtp_profile = rtp_gen.get_prices(date_str)
    min_p, max_p = min(rtp_profile), max(rtp_profile)
    
    # Mini sparkline
    fig = go.Figure(go.Bar(x=list(range(6)), y=rtp_profile, 
        marker_color=['#38ef7d' if p==min_p else '#f5576c' if p==max_p else '#667eea' for p in rtp_profile]))
    fig.update_layout(height=60, margin=dict(l=0,r=0,t=0,b=0), 
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

use_ibm = IBM_AVAILABLE

st.markdown("---")

# ==================== ROW 2: Appliances + Results ====================
col_apps, col_results = st.columns([1, 1])

with col_apps:
    st.markdown("##### 🔌 Your Appliances")
    if st.session_state['appliances']:
        app_data = [{
            "Appliance": i['name'],
            "kWh": f"{i['energy']:.1f}",
            "Time": i['base_time'].strftime("%H:%M"),
            "Flex": "✅" if i['is_flexible'] else "🔒"
        } for i in st.session_state['appliances']]
        st.dataframe(pd.DataFrame(app_data), use_container_width=True, hide_index=True, height=200)
        
        b1, b2 = st.columns(2)
        with b1:
            run_opt = st.button("🚀 OPTIMIZE", type="primary", use_container_width=True)
        with b2:
            if st.button("🗑️ CLEAR", use_container_width=True):
                st.session_state['appliances'] = []
                st.session_state['results'] = None
                st.rerun()
    else:
        st.info("Add appliances above to get started")
        run_opt = False

with col_results:
    st.markdown("##### 📊 Optimization Results")
    if st.session_state['results']:
        res = st.session_state['results']
        savings = res['base'] - res['opt']
        pct = (savings / res['base'] * 100) if res['base'] > 0 else 0
        
        # Stats row
        s1, s2, s3 = st.columns(3)
        s1.metric("Traditional", f"₹{res['base']:.0f}", delta=None)
        s2.metric("Optimized", f"₹{res['opt']:.0f}", delta=None)
        s3.metric("Savings", f"₹{savings:.0f}", delta=f"{pct:.0f}%")
        
        # Schedule table
        df = pd.DataFrame(res['details'])
        df = df[['Appliance', 'Original Slot', 'Optimized Slot', 'Savings']]
        df['Savings'] = df['Savings'].apply(lambda x: f"₹{x:.1f}")
        st.dataframe(df, use_container_width=True, hide_index=True, height=140)
    else:
        st.markdown("""
        <div class="dashboard-card" style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem;">📈</div>
            <p style="color: #888;">Click <b>OPTIMIZE</b> to see savings</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== OPTIMIZATION LOGIC ====================
if 'run_opt' in dir() and run_opt and st.session_state['appliances']:
    with st.spinner("🧠 Optimizing..."):
        agent = None
        if os.path.exists(MODEL_PATH):
            agent = QLearningAgent()
            agent.load_model(MODEL_PATH)
        
        total_base, total_opt, results = 0, 0, []
        
        for item in st.session_state['appliances']:
            start_slot, end_slot = (0, 5) if item['is_flexible'] else (item['base_slot'], item['base_slot'])
            
            state_key = StateUtils.discretize_state(
                item['name'], item['energy'], temp_input, household_size, rtp_profile,
                is_flexible=item['is_flexible'], start_slot=start_slot, end_slot=end_slot
            )
            
            rec_slot = None
            if use_ibm: rec_slot = get_ibm_recommendation(state_key)
            if rec_slot is None and agent: rec_slot = agent.choose_action(state_key, force_greedy=True)
            if rec_slot is None: rec_slot = rtp_profile.index(min(rtp_profile))
            
            b_cost = item['energy'] * rtp_profile[item['base_slot']]
            o_cost = item['energy'] * rtp_profile[rec_slot]
            total_base += b_cost
            total_opt += o_cost
            
            results.append({
                "Appliance": item['name'],
                "Original Slot": get_slot_label(item['base_slot']),
                "Optimized Slot": get_slot_label(rec_slot),
                "Savings": b_cost - o_cost
            })
        
        st.session_state['results'] = {'base': total_base, 'opt': total_opt, 'details': results}
        st.rerun()
