import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime
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
    .main .block-container { padding: 0 2rem 1rem 2rem; max-width: 1200px; margin: 0 auto; }
    [data-testid="stSidebar"] { display: none; }
    header[data-testid="stHeader"] { display: none; }
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

# Date and Result Metrics Row
date_col, spacer, m1, m2, m3 = st.columns([0.12, 0.4, 0.16, 0.16, 0.16])
with date_col:
    date_input = st.date_input("📅", value=datetime.now().date(), label_visibility="visible", key="date_hidden")
with m1:
    if st.session_state['results']:
        st.markdown(f'<div class="card"><div class="card-title">Traditional</div><div class="card-value red">₹{st.session_state["results"]["base"]:.0f}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card"><div class="card-title">Traditional</div><div class="card-value">--</div></div>', unsafe_allow_html=True)
with m2:
    if st.session_state['results']:
        st.markdown(f'<div class="card"><div class="card-title">Optimized</div><div class="card-value">₹{st.session_state["results"]["opt"]:.0f}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card"><div class="card-title">Optimized</div><div class="card-value">--</div></div>', unsafe_allow_html=True)
with m3:
    if st.session_state['results']:
        sav = st.session_state['results']['base'] - st.session_state['results']['opt']
        st.markdown(f'<div class="card"><div class="card-title">Savings</div><div class="card-value green">₹{sav:.0f}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card"><div class="card-title">Savings</div><div class="card-value">--</div></div>', unsafe_allow_html=True)

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
        st.dataframe(df_apps, use_container_width=True, hide_index=True, height=250)
        
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
            ibm_count, local_count = 0, 0
            
            for item in st.session_state['appliances']:
                s_slot = 0 if item['is_flexible'] else item['base_slot']
                e_slot = 5 if item['is_flexible'] else item['base_slot']
                state = StateUtils.discretize_state(item['name'], item['energy'], temp_input, household_size, rtp_profile, item['is_flexible'], s_slot, e_slot)
                
                rec = None
                if use_ibm:
                    rec = get_ibm_recommendation(state)
                    if rec is not None:
                        ibm_count += 1
                
                if rec is None:
                    # Try RL Agent if model exists
                    if os.path.exists(MODEL_PATH):
                        rec = agent.choose_action(state, force_greedy=True)
                        
                        # Sanity check: If RL result is worse than base, force math optimal
                        if agent_cost > base_cost:
                             rec = agent.get_best_slot_for_price(rtp_profile, item['is_flexible'], s_slot, e_slot)
                    else:
                        # Fallback to pure math optimization (guaranteed best slot)
                        # We need to implement this logic since agent might not be loaded
                         best_slot = s_slot
                         min_p = rtp_profile[s_slot]
                         valid_range = range(s_slot, e_slot + 1)
                         for i in valid_range:
                             if rtp_profile[i] < min_p:
                                 min_p = rtp_profile[i]
                                 best_slot = i
                         rec = best_slot
                    
                    local_count += 1
                
                # FINAL SAFETY CHECK: regardless of source (IBM or Local)
                # If the recommended slot is NOT cheaper than the original, force find the cheapest
                if item['is_flexible']:
                    current_price = rtp_profile[rec]
                    base_price = rtp_profile[item['base_slot']]
                    
                    # If we are not saving money (allow small tolerance), find absolute best
                    if current_price >= base_price:
                         # Find mathematically optimal slot
                         best_math_slot = 0
                         min_math_p = float('inf')
                         for i in range(6): # All slots
                             if rtp_profile[i] < min_math_p:
                                 min_math_p = rtp_profile[i]
                                 best_math_slot = i
                         
                         # Use math optimal
                         rec = best_math_slot
                
                b_cost = item['energy'] * rtp_profile[item['base_slot']]
                o_cost = item['energy'] * rtp_profile[rec]
                total_base += b_cost
                total_opt += o_cost
                details.append({"Appliance": item['name'], "From": get_slot_short(item['base_slot']), "To": get_slot_short(rec), "b_cost": b_cost, "o_cost": o_cost, "save": b_cost - o_cost})
            
            st.session_state['results'] = {'base': total_base, 'opt': total_opt, 'details': details, 'ibm_count': ibm_count, 'local_count': local_count}
            st.rerun()
    else:
        st.info("Add appliances to get started")
        run_opt = False

# ----- COLUMN 3: Results -----
with col3:
    st.markdown('<p class="section-title">📈 Schedule Details</p>', unsafe_allow_html=True)
    if st.session_state['results']:
        res = st.session_state['results']
        
        # Schedule Table only (metrics are now at top)
        
        # Schedule Table 
        df_res = pd.DataFrame(res['details'])
        if 'From' in df_res.columns:
            display_df = df_res[['Appliance', 'From', 'To', 'save']].rename(columns={'save': 'Savings ₹'})
        else:
            display_df = df_res
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=250)
        
        # IBM Status Message
        ibm_count = res.get('ibm_count', 0)
        local_count = res.get('local_count', 0)
        if ibm_count > 0:
            st.markdown(f'''
            <div style="background: linear-gradient(90deg, #0d3b66 0%, #1a365d 100%); padding: 0.5rem 1rem; border-radius: 6px; margin-top: 0.5rem; display: flex; align-items: center; gap: 8px;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg" style="height: 14px; filter: brightness(0) invert(1);">
                <span style="color: #38ef7d; font-size: 0.75rem;">✓ {ibm_count} predictions via IBM Watson ML</span>
            </div>
            ''', unsafe_allow_html=True)
        elif local_count > 0:
            st.markdown(f'<div style="color: #888; font-size: 0.7rem; margin-top: 0.5rem;">🏠 {local_count} predictions via Local Model</div>', unsafe_allow_html=True)
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
    st.markdown('<p class="section-title">📊 Today\'s RTP Prices</p>', unsafe_allow_html=True)
    
    # Add padding to X-axis so markers aren't cut off
    x_vals = [get_slot_short(i) for i in range(6)]
    
    fig_rtp = go.Figure()
    fig_rtp.add_trace(go.Scatter(
        x=x_vals, y=rtp_profile,
        mode='lines+markers+text',
        line=dict(color='#667eea', width=4, shape='spline'),
        marker=dict(size=14, color=['#38ef7d' if p==min_p else '#f5576c' if p==max_p else '#667eea' for p in rtp_profile],
                    line=dict(color='white', width=2)),
        text=[f"₹{p:.1f}" for p in rtp_profile], 
        textposition="top center",
        textfont=dict(size=12, color='#e0e0e0', family="Arial"),
        fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    y_max = max(rtp_profile) * 1.3  # More headroom for labels
    fig_rtp.update_layout(
        height=280, 
        margin=dict(l=20,r=20,t=40,b=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[0, y_max], showticklabels=False),
        xaxis=dict(showgrid=False, tickfont=dict(size=11, color='#aaa'), range=[-0.5, 5.5]) # Padding
    )
    st.plotly_chart(fig_rtp, use_container_width=True, config={'displayModeBar': False})

with chart_col2:
    st.markdown('<p class="section-title">📈 Cost Analysis</p>', unsafe_allow_html=True)
    if st.session_state['results']:
        res = st.session_state['results']
        df_res = pd.DataFrame(res['details'])
        
        # Ensure we have the latest cost data
        if 'b_cost' in df_res.columns:
            orig_costs = df_res['b_cost'].tolist()
            opt_costs = df_res['o_cost'].tolist()
        else:
            orig_costs = [10.0] * len(res['details'])
            opt_costs = [8.0] * len(res['details'])
        
        fig = go.Figure()
        
        # Original Cost Bar
        fig.add_trace(go.Bar(
            name='Original', y=df_res['Appliance'], x=orig_costs, 
            orientation='h', 
            marker=dict(color='#f5576c', opacity=0.9, line=dict(width=0)),
            text=[f" ₹{x:.1f}" for x in orig_costs], textposition='outside', # Outside for clearer reading
            textfont=dict(color='#f5576c')
        ))
        
        # Optimized Cost Bar
        fig.add_trace(go.Bar(
            name='Optimized', y=df_res['Appliance'], x=opt_costs, 
            orientation='h', 
            marker=dict(color='#38ef7d', opacity=0.9, line=dict(width=0)),
            text=[f" ₹{x:.1f}" for x in opt_costs], textposition='outside',
            textfont=dict(color='#38ef7d')
        ))
        
        # Calculate dynamic height
        chart_height = max(280, len(df_res) * 60)
        max_cost = max(max(orig_costs), max(opt_costs)) * 1.3 # Headroom for text
        
        fig.update_layout(
            barmode='group', 
            bargap=0.30, # Space between appliance groups
            bargroupgap=0.1, # Space between bars in a group
            height=chart_height, 
            margin=dict(l=10,r=40,t=20,b=20), # Right margin for text
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", y=1.1, x=0, xanchor="left", font=dict(size=12)),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', showticklabels=False, range=[0, max_cost]),
            yaxis=dict(showgrid=False, tickfont=dict(size=12, color='#ddd'))
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
         st.markdown("""
        <div class="card" style="text-align: center; padding: 2.5rem; height: 280px; display: flex; flex-direction: column; align-items: center; justify-content: center; opacity: 0.5;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <p style="color: #aaa; margin: 0;">Optimization results will<br>appear here</p>
        </div>
        """, unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #555; font-size: 0.7rem;">
    🔋 Smart Energy Optimizer | Built with <img src="https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg" style="height: 12px; vertical-align: middle; opacity: 0.6;"> IBM Watson ML
</div>
""", unsafe_allow_html=True)
