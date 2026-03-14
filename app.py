"""
=============================================================================
VECV Supply Chain — Interactive Sensitivity Analysis Dashboard
Built with Streamlit
=============================================================================

Run with:  streamlit run vecv_dashboard.py

This dashboard provides real-time interactive sensitivity analysis of the
VECV Pithampur assembly plant, allowing users to explore how changes in
key operational parameters affect throughput, utilization, WIP, and
bottleneck identification — based on the supply chain optimization report.
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="VECV Supply Chain Dashboard",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1a3a5c;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .kpi-box {
        background: #f0f4fa;
        border-left: 4px solid #1f77b4;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .kpi-label { font-size: 0.75rem; color: #666; font-weight: 600; text-transform: uppercase; }
    .kpi-value { font-size: 1.6rem; font-weight: bold; color: #1a3a5c; }
    .kpi-delta { font-size: 0.82rem; color: #2ca02c; }
    .kpi-delta-neg { font-size: 0.82rem; color: #d62728; }
    .bottleneck-badge {
        display: inline-block;
        background: #d62728;
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .ok-badge {
        display: inline-block;
        background: #2ca02c;
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .warn-badge {
        display: inline-block;
        background: #ff7f0e;
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a3a5c;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 4px;
        margin: 1rem 0 0.8rem 0;
    }
    .insight-box {
        background: #fffbe6;
        border-left: 4px solid #ff7f0e;
        border-radius: 4px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        color: #4a3800;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CORE COMPUTATION (inline, no external import)
# =============================================================================

WORKING_DAYS = 300
HOURS_PER_DAY = 16
WORKING_HOURS = WORKING_DAYS * HOURS_PER_DAY   # 4800
BASE_DEMAND = 90_000
BASE_TAKT = (WORKING_HOURS * 60) / BASE_DEMAND  # 3.20 min

STAGE_NAMES_FULL = [
    "Robotic Welding",
    "Cab Assembly",
    "Painting (CED/PU)",
    "Step 14: Wheel Mount",
    "Wheel Alignment",
    "Brake Bleeding",
    "End-of-Line Testing"
]

BASE_CYCLE_TIMES = [3.00, 3.10, 3.05, 5.43, 4.44, 3.96, 1.50]
BASE_STD_DEVS    = [0.20, 0.25, 0.20, 0.60, 0.50, 0.40, 0.15]


def compute_metrics(
    annual_demand,
    wheel_mount_ct,
    num_wheel_machines,
    num_welding_machines,
    num_painting_machines,
    processing_time_factor,
    inventory_buffer,
    n_sim=300,
    seed=42
):
    """Compute all KPIs for the given parameter set via Monte Carlo."""
    rng = np.random.default_rng(seed)

    takt = (WORKING_HOURS * 60) / annual_demand
    demand_ph = annual_demand / WORKING_HOURS

    # Scale cycle times by processing time factor, override wheel mount directly
    raw_cts = [
        BASE_CYCLE_TIMES[0] * processing_time_factor,  # Welding
        BASE_CYCLE_TIMES[1] * processing_time_factor,  # Cab
        BASE_CYCLE_TIMES[2] * processing_time_factor,  # Painting
        wheel_mount_ct,                                  # Wheel Mount (direct control)
        BASE_CYCLE_TIMES[4] * processing_time_factor,  # Alignment
        BASE_CYCLE_TIMES[5] * processing_time_factor,  # Brake Bleeding
        BASE_CYCLE_TIMES[6] * processing_time_factor,  # Testing
    ]
    raw_stds = BASE_STD_DEVS[:]
    n_stations = [
        num_welding_machines, 1, num_painting_machines,
        num_wheel_machines, 1, 1, 1
    ]

    throughputs, utils, waits, wips, bn_indices = [], [], [], [], []

    for _ in range(n_sim):
        sampled = [max(0.5, rng.normal(ct, sd)) for ct, sd in zip(raw_cts, raw_stds)]
        eff_cts = [ct / ns for ct, ns in zip(sampled, n_stations)]
        bn_idx = int(np.argmax(eff_cts))
        bn_eff_ct = eff_cts[bn_idx]

        tp = 60.0 / bn_eff_ct
        buf_boost = min(0.15, inventory_buffer * 0.01)
        eff_tp = tp * (1 + buf_boost)

        cap = 60.0 / bn_eff_ct
        util = demand_ph / cap
        util = min(util, 2.0)

        if util < 1.0:
            wt = (util / (cap * (1 - util))) * 60
        else:
            wt = 9999.0

        wip = eff_tp * 24.0

        throughputs.append(eff_tp)
        utils.append(util)
        waits.append(wt)
        wips.append(wip)
        bn_indices.append(bn_idx)

    valid_waits = [w for w in waits if w < 9000]

    return {
        "takt_min": takt,
        "demand_ph": demand_ph,
        "throughput_ph": float(np.mean(throughputs)),
        "throughput_year": float(np.mean(throughputs)) * WORKING_HOURS,
        "throughput_std": float(np.std(throughputs)),
        "utilization": float(np.mean(utils)),
        "utilization_std": float(np.std(utils)),
        "wait_min": float(np.mean(valid_waits)) if valid_waits else float("inf"),
        "wip": float(np.mean(wips)),
        "bottleneck_idx": int(np.median(bn_indices)),
        "bottleneck_name": STAGE_NAMES_FULL[int(np.median(bn_indices))],
        "eff_cts": [ct / ns for ct, ns in zip(raw_cts, n_stations)],
        "throughputs_dist": throughputs,
        "utils_dist": utils,
    }


# =============================================================================
# SIDEBAR — PARAMETER CONTROLS
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:12px;">
        <img src="data:image/png;base64,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" style="width:200px; display:block;"/>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 🎛 Parameter Controls")
    st.markdown("Adjust parameters to explore sensitivity of the Pithampur assembly system.")

    st.markdown("---")
    st.markdown("#### 📦 Demand & Capacity")
    annual_demand = st.slider(
        "Annual Demand (vehicles)",
        min_value=40_000, max_value=180_000, value=90_000, step=5_000,
        help="FY25 actual demand: 90,161 vehicles (report section 1.4)"
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Bottleneck Stage")
    wheel_mount_ct = st.slider(
        "Step 14: Wheel Mount Cycle Time (min)",
        min_value=1.5, max_value=8.0, value=5.43, step=0.05,
        help="Base: 5.43 min. Process redesign target: 4.30 min (-20%). Report sec. 3.1"
    )
    num_wheel_machines = st.slider(
        "Parallel Wheel Mount Stations",
        min_value=1, max_value=5, value=2,
        help="Report sec 3.2: 2 stations brings effective CT to 2.15 min (below takt)"
    )

    st.markdown("---")
    st.markdown("#### 🏭 Processing & Resources")
    processing_time_factor = st.slider(
        "Processing Time Factor (all stages)",
        min_value=0.6, max_value=1.3, value=1.0, step=0.05,
        help="1.0 = baseline. 0.8 = 20% improvement via tech adoption (report sec 3.3)"
    )
    num_welding_machines = st.slider(
        "Welding/Cab Assembly Stations", min_value=1, max_value=4, value=1
    )
    num_painting_machines = st.slider(
        "Painting Line Stations", min_value=1, max_value=4, value=1
    )

    st.markdown("---")
    st.markdown("#### 📦 Inventory Buffer")
    inventory_buffer = st.slider(
        "WIP Buffer Upstream of Bottleneck (vehicles)",
        min_value=0, max_value=50, value=10,
        help="Report sec 3.8: Buffer of 5 absorbs 15-min upstream AGV fault"
    )

    st.markdown("---")
    run_button = st.button("🔄 Recompute Analysis", use_container_width=True, type="primary")


# =============================================================================
# MAIN PANEL — HEADER
# =============================================================================

st.markdown('<div class="main-header">🚛 VECV Supply Chain Sensitivity Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Pithampur Assembly Plant · Volvo Eicher Commercial Vehicles · Interactive Operational Analysis</div>', unsafe_allow_html=True)

# Compute results
with st.spinner("Running Monte Carlo simulation..."):
    current = compute_metrics(
        annual_demand, wheel_mount_ct, num_wheel_machines,
        num_welding_machines, num_painting_machines,
        processing_time_factor, inventory_buffer, n_sim=400
    )
    baseline = compute_metrics(
        BASE_DEMAND, 5.43, 2, 1, 1, 1.0, 10, n_sim=200
    )


# =============================================================================
# KPI ROW
# =============================================================================

def delta_str(current_val, base_val, fmt=".1f", higher_is_better=True, unit=""):
    diff = current_val - base_val
    pct = (diff / base_val * 100) if base_val != 0 else 0
    sign = "▲" if diff > 0 else "▼"
    color = "green" if (diff > 0) == higher_is_better else "red"
    return f"<span style='color:{color}'>{sign} {abs(pct):.1f}% vs baseline</span>"


st.markdown('<div class="section-title">📊 Key Performance Indicators</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-label">Throughput Rate</div>
        <div class="kpi-value">{current['throughput_ph']:.1f}</div>
        <div style="font-size:0.75rem;color:#888">vehicles / hour</div>
        {delta_str(current['throughput_ph'], baseline['throughput_ph'], higher_is_better=True)}
    </div>""", unsafe_allow_html=True)

with k2:
    annual_cap = current['throughput_year']
    meets = annual_cap >= annual_demand
    badge = '<span class="ok-badge">✓ Meets Demand</span>' if meets else '<span class="bottleneck-badge">✗ Deficit</span>'
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-label">Annual Capacity</div>
        <div class="kpi-value">{annual_cap/1000:.1f}k</div>
        <div style="font-size:0.75rem;color:#888">vehicles / year &nbsp; {badge}</div>
        {delta_str(annual_cap, baseline['throughput_year'], higher_is_better=True)}
    </div>""", unsafe_allow_html=True)

with k3:
    util = current['utilization']
    u_badge = ('<span class="ok-badge">✓ Stable</span>' if util < 0.85
               else '<span class="warn-badge">⚠ High</span>' if util < 1.0
               else '<span class="bottleneck-badge">✗ Overloaded</span>')
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-label">Bottleneck Utilization</div>
        <div class="kpi-value">{util:.1%}</div>
        <div style="font-size:0.75rem;color:#888">ρ = {util:.3f} &nbsp; {u_badge}</div>
        {delta_str(util, baseline['utilization'], higher_is_better=False)}
    </div>""", unsafe_allow_html=True)

with k4:
    wt = current['wait_min']
    wt_str = f"{wt:.1f} min" if wt < 9000 else "∞ (Queue Unstable)"
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-label">Avg Waiting Time</div>
        <div class="kpi-value">{wt_str if wt < 9000 else "∞"}</div>
        <div style="font-size:0.75rem;color:#888">at bottleneck (M/M/1)</div>
        {'<span style="color:red">Queue is unstable (ρ ≥ 1)</span>' if wt >= 9000 else delta_str(wt, baseline['wait_min'], higher_is_better=False)}
    </div>""", unsafe_allow_html=True)

with k5:
    wip = current['wip']
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-label">WIP Inventory</div>
        <div class="kpi-value">{wip:.0f}</div>
        <div style="font-size:0.75rem;color:#888">vehicles in process (Little's Law)</div>
        {delta_str(wip, baseline['wip'], higher_is_better=False)}
    </div>""", unsafe_allow_html=True)


# =============================================================================
# BOTTLENECK INDICATOR
# =============================================================================

bn = current['bottleneck_name']
takt = current['takt_min']
eff_cts = current['eff_cts']

st.markdown('<div class="section-title">🔴 Bottleneck & Stage Analysis</div>', unsafe_allow_html=True)

bn_col, takt_col = st.columns([2, 1])

with bn_col:
    st.markdown(f"""
    <div style="padding:0.8rem 1rem; background:#fff3f3; border:1.5px solid #d62728; border-radius:8px;">
        <span style="font-size:0.8rem; color:#888; font-weight:600;">CURRENT BOTTLENECK STAGE</span><br>
        <span style="font-size:1.3rem; font-weight:bold; color:#d62728;">⚠ {bn}</span>
        <span style="font-size:0.85rem; color:#555; margin-left:1rem;">
            Effective CT: {eff_cts[current['bottleneck_idx']]:.2f} min  |  
            Takt Time: {takt:.2f} min  |  
            Excess: +{max(0, eff_cts[current['bottleneck_idx']] - takt):.2f} min
        </span>
    </div>""", unsafe_allow_html=True)

with takt_col:
    demand_ph = annual_demand / WORKING_HOURS
    st.markdown(f"""
    <div style="padding:0.8rem 1rem; background:#f0f7ff; border:1.5px solid #1f77b4; border-radius:8px;">
        <span style="font-size:0.8rem; color:#888; font-weight:600;">TAKT TIME</span><br>
        <span style="font-size:1.3rem; font-weight:bold; color:#1f77b4;">{takt:.2f} min/unit</span>
        <span style="font-size:0.85rem; color:#555; display:block;">Required rate: {demand_ph:.2f} veh/hr</span>
    </div>""", unsafe_allow_html=True)


# =============================================================================
# CHARTS ROW 1: Stage comparison + Throughput distribution
# =============================================================================

st.markdown('<div class="section-title">📈 Visual Analysis</div>', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("**Stage Cycle Times vs Takt Time**")
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    colors = ["#d62728" if ect > takt else "#2ca02c" for ect in eff_cts]
    bars = ax1.barh(STAGE_NAMES_FULL, eff_cts, color=colors, alpha=0.85,
                    edgecolor='white', height=0.55)
    ax1.axvline(x=takt, color='#ff7f0e', lw=2, ls='--',
                label=f"Takt Time ({takt:.2f} min)")
    for bar, v in zip(bars, eff_cts):
        ax1.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                 f"{v:.2f}", va='center', fontsize=8)
    ax1.set_xlabel("Effective Cycle Time (min/unit)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2, axis='x')
    ax1.invert_yaxis()
    ax1.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

with chart_col2:
    st.markdown("**Throughput Distribution (Monte Carlo)**")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    tp_dist = current["throughputs_dist"]
    ax2.hist(tp_dist, bins=30, color='#1f77b4', alpha=0.7, edgecolor='white')
    ax2.axvline(x=np.mean(tp_dist), color='#d62728', lw=2,
                label=f"Mean: {np.mean(tp_dist):.1f} veh/hr")
    ax2.axvline(x=demand_ph, color='#ff7f0e', lw=2, ls='--',
                label=f"Required: {demand_ph:.1f} veh/hr")
    ax2.set_xlabel("Throughput (vehicles/hour)")
    ax2.set_ylabel("Frequency")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# =============================================================================
# CHARTS ROW 2: Demand sweep + Capacity impact
# =============================================================================

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.markdown("**Utilization vs Demand Rate (Sweep)**")
    demands_sweep = np.arange(40000, 185000, 5000)
    utils_sweep, wts_sweep = [], []
    for d in demands_sweep:
        d_ph = d / WORKING_HOURS
        cap_ph = 60.0 / (wheel_mount_ct / num_wheel_machines)
        u = min(d_ph / cap_ph, 2.0)
        utils_sweep.append(u)
        wts_sweep.append((u / (cap_ph * (1 - u))) * 60 if u < 1.0 else None)

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(demands_sweep / 1000, utils_sweep, color='#1f77b4', lw=2.5)
    ax3.axhline(y=1.0, color='#d62728', ls='--', lw=1.5, label="Overload (ρ=1.0)")
    ax3.axhline(y=0.85, color='#ff7f0e', ls=':', lw=1.5, label="High Load (ρ=0.85)")
    ax3.axvline(x=annual_demand / 1000, color='#2ca02c', ls='-', lw=1.5,
                label=f"Current Demand ({annual_demand/1000:.0f}k)")
    ax3.fill_between(demands_sweep / 1000, utils_sweep, 1.0,
                     where=[u > 1.0 for u in utils_sweep],
                     color='#d62728', alpha=0.15)
    ax3.set_xlabel("Annual Demand (thousands)")
    ax3.set_ylabel("Bottleneck Utilization (ρ)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)
    ax3.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

with chart_col4:
    st.markdown("**Capacity vs Number of Wheel Mount Stations**")
    stations_range = list(range(1, 6))
    annual_caps = [(60.0 / (wheel_mount_ct / n)) * WORKING_HOURS for n in stations_range]
    eff_cts_range = [wheel_mount_ct / n for n in stations_range]

    fig4, ax4 = plt.subplots(figsize=(7, 4))
    bar_colors = ['#2ca02c' if c >= annual_demand else '#d62728' for c in annual_caps]
    bars4 = ax4.bar(stations_range, [c / 1000 for c in annual_caps],
                    color=bar_colors, alpha=0.85, edgecolor='white', width=0.5)
    ax4.axhline(y=annual_demand / 1000, color='#ff7f0e', ls='--', lw=2,
                label=f"Demand Target ({annual_demand/1000:.0f}k)")
    ax4.set_xlabel("Number of Parallel Wheel Mount Stations")
    ax4.set_ylabel("Annual Capacity (thousand vehicles)")
    ax4.set_xticks(stations_range)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2, axis='y')
    for bar, cap in zip(bars4, annual_caps):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{cap/1000:.0f}k", ha='center', fontsize=8)
    ax4.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()


# =============================================================================
# INSIGHT BOX
# =============================================================================

gap = annual_demand - current['throughput_year']
insight_parts = []

if current['utilization'] >= 1.0:
    insight_parts.append(f"⚠ Queue is UNSTABLE (ρ={current['utilization']:.2f}). "
                         f"Annual deficit of {gap:,.0f} vehicles. Immediate capacity addition required.")
elif current['utilization'] > 0.85:
    insight_parts.append(f"⚠ High utilization ({current['utilization']:.1%}) at '{bn}'. "
                         f"Waiting times increase exponentially beyond ρ=0.85 (M/M/1 dynamics).")
else:
    insight_parts.append(f"✓ System is stable with {current['utilization']:.1%} utilization at '{bn}'.")

if current['throughput_year'] >= annual_demand:
    insight_parts.append(f"Annual capacity ({current['throughput_year']/1000:.1f}k) meets demand ({annual_demand/1000:.0f}k) ✓")
else:
    insight_parts.append(f"Annual capacity shortfall: {gap:,.0f} vehicles. "
                         f"Consider adding {int(np.ceil(gap/53040))+1} parallel station(s).")

if eff_cts[current['bottleneck_idx']] > takt:
    insight_parts.append(f"Bottleneck CT ({eff_cts[current['bottleneck_idx']]:.2f} min) exceeds "
                         f"Takt Time ({takt:.2f} min) — WIP will pool upstream, "
                         f"creating ~{wip:.0f} vehicles of in-process inventory.")

st.markdown(f'<div class="insight-box">💡 <b>System Insight:</b> {" &nbsp;|&nbsp; ".join(insight_parts)}</div>',
            unsafe_allow_html=True)


# =============================================================================
# DATA TABLE
# =============================================================================

with st.expander("📋 View Detailed Metrics Table"):
    metrics_df = pd.DataFrame({
        "Stage": STAGE_NAMES_FULL,
        "Base CT (min)": [f"{BASE_CYCLE_TIMES[i]:.2f}" for i in range(len(STAGE_NAMES_FULL))],
        "Effective CT (min)": [f"{ect:.2f}" for ect in eff_cts],
        "Takt Time (min)": [f"{takt:.2f}"] * len(STAGE_NAMES_FULL),
        "Bottleneck?": ["🔴 YES" if i == current['bottleneck_idx'] else "✅ No"
                        for i in range(len(STAGE_NAMES_FULL))],
        "Excess over Takt": [f"+{max(0, ect - takt):.2f} min" if ect > takt else "—"
                             for ect in eff_cts],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    summary_df = pd.DataFrame({
        "Metric": ["Annual Demand", "Takt Time", "Throughput/hr", "Annual Capacity",
                   "Utilization", "Waiting Time", "WIP (Little's Law)", "Bottleneck Stage"],
        "Value": [
            f"{annual_demand:,} vehicles",
            f"{takt:.2f} min/unit",
            f"{current['throughput_ph']:.2f} veh/hr",
            f"{current['throughput_year']:,.0f} vehicles",
            f"{current['utilization']:.3f} ({current['utilization']:.1%})",
            f"{current['wait_min']:.2f} min" if current['wait_min'] < 9000 else "Unstable (∞)",
            f"{current['wip']:.0f} vehicles",
            current['bottleneck_name']
        ]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# =============================================================================
# FOOTER — METHODOLOGY
# =============================================================================

with st.expander("📚 Methodology & Model Documentation"):
    st.markdown("""
    ### Model Architecture

    This simulation is built on **Monte Carlo methods** combined with **analytical queueing theory**, 
    grounded in the VECV Pithampur operational data from the supply chain report.

    #### Key Equations

    | Concept | Formula | Source |
    |---|---|---|
    | **Takt Time** | T_takt = Available Time / Demand | Report Section 1.4 |
    | **Throughput** | X = 60 / (CT_bottleneck / N_stations) | Report Section 2.2 |
    | **Utilization** | ρ = λ / μ | Report Section 2.3 |
    | **Waiting Time (M/M/1)** | W_q = ρ / (μ(1−ρ)) | Report Section 2.3 |
    | **WIP (Little's Law)** | L = λ × W | Report Section 2.4 |

    #### Stochastic Modeling
    - **400 Monte Carlo samples** per parameter set
    - Normal distributions for cycle times (calibrated to empirical std devs from report)
    - Floor of 0.5 min applied to prevent negative cycle times
    - Buffer protection factor: min(15%, buffer_size × 1%) boost to throughput

    #### Baseline Parameters (from Report)
    - Pithampur plant capacity: 90,000 vehicles/year
    - Working calendar: 300 days × 16 hrs/day = 4,800 hrs/year
    - Primary bottleneck: Step 14 (Wheel Mount) — 5.43 min/unit
    - Secondary constraints: Wheel Alignment (4.44 min), Brake Bleeding (3.96 min)
    - Average WIP: ~444 vehicles (Little's Law with 24-hr cycle)
    - Inventory holding: 41 days (financial disclosures FY25)
    """)


st.markdown("---")
st.markdown(
    "<center style='color:#aaa; font-size:0.8rem'>VECV Supply Chain Dashboard · "
    "Monte Carlo Simulation with M/M/1 Queueing · "
    "Based on Exhaustive Operational Analysis Report</center>",
    unsafe_allow_html=True
)
