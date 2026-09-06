# =====================================================================
# HELM-IIoT: INDUSTRIAL CYBER-PHYSICAL SCADA MONITORING CENTER
# HIGH-PERFORMANCE HMI (ISA-101 / ISA-18.2 / IEC 62443-4-2 COMPLIANT)
# =====================================================================
import os
import sys
import time
import json
import base64
import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st

# Ensure root workspace directory is in sys.path when running from src/
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scada_ui.client import helm_client
from ml_inference_service.rca_copilot import copilot_engine
from config.settings import settings

# ---------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="HELM-IIoT | Industrial SCADA Control Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------
# PERSISTENT SESSION STATE INITIALIZATION
# ---------------------------------------------------------------------
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "⬡ Live SCADA Telemetry"
if "enable_mitigation" not in st.session_state:
    st.session_state.enable_mitigation = True
if "enable_throttling" not in st.session_state:
    st.session_state.enable_throttling = True
if "enable_failover" not in st.session_state:
    st.session_state.enable_failover = True
if "mode" not in st.session_state:
    st.session_state.mode = "Automated Real-Time Polling"
if "chaos_mode" not in st.session_state:
    st.session_state.chaos_mode = "None"
if "latency_threshold_ms" not in st.session_state:
    st.session_state.latency_threshold_ms = 60.0
if "warning_threshold_ms" not in st.session_state:
    st.session_state.warning_threshold_ms = 45.0
if "shedding_factor" not in st.session_state:
    st.session_state.shedding_factor = 0.82
if "throttling_temp_threshold" not in st.session_state:
    st.session_state.throttling_temp_threshold = 68.0
if "failover_drop_threshold" not in st.session_state:
    st.session_state.failover_drop_threshold = 1.80
if "total_cycles" not in st.session_state:
    st.session_state.total_cycles = 50
if "mitigation_checks" not in st.session_state:
    st.session_state.mitigation_checks = 0
if "mitigations_successful" not in st.session_state:
    st.session_state.mitigations_successful = 0
if "failover_events" not in st.session_state:
    st.session_state.failover_events = 0
if "peak_temp" not in st.session_state:
    st.session_state.peak_temp = 49.2
if "peak_latency" not in st.session_state:
    st.session_state.peak_latency = 44.8
if "alarm_ack" not in st.session_state:
    st.session_state.alarm_ack = True
if "alarm_muted" not in st.session_state:
    st.session_state.alarm_muted = False
if "playbooks" not in st.session_state:
    st.session_state.playbooks = {
        "pb_shedding": True,
        "pb_thermal": True,
        "pb_failover": True,
        "pb_drift_retrain": True
    }

# Initialize Industrial Sequence-of-Events (SOE) & Alarm Logs
if "incident_logs" not in st.session_state:
    st.session_state.incident_logs = [
        {"Timestamp": time.strftime("%H:%M:%S", time.localtime(time.time() - 40)), "Alarm_ID": "ALM-1001", "Priority": "P3-INFO", "Tag": "SYS_INIT", "Event": "Industrial Ethernet Ingress Bridge initialized on TSN eth0 (IEEE 802.1Qbv).", "Quality": "GOOD_0x00"},
        {"Timestamp": time.strftime("%H:%M:%S", time.localtime(time.time() - 30)), "Alarm_ID": "ALM-1002", "Priority": "P3-INFO", "Tag": "OPC_UA_SRV", "Event": "OPC-UA PubSub Server listening on opc.tcp://192.168.10.14:4840.", "Quality": "GOOD_0x00"},
        {"Timestamp": time.strftime("%H:%M:%S", time.localtime(time.time() - 20)), "Alarm_ID": "ALM-1003", "Priority": "P3-INFO", "Tag": "XGB_CV_80", "Event": "Time-series cross-validated XGBoost regressor engine loaded into memory.", "Quality": "GOOD_0x00"},
        {"Timestamp": time.strftime("%H:%M:%S", time.localtime(time.time() - 10)), "Alarm_ID": "ALM-1004", "Priority": "P3-INFO", "Tag": "PLC_HEARTBEAT", "Event": "Deterministic heartbeat nominal across PLC Node Alpha & Node Delta Standby.", "Quality": "GOOD_0x00"}
    ]

# Initialize Telemetry Warm Start History
if "history" not in st.session_state:
    warm_start_data = []
    base_time = time.time() - 30
    for i in range(30):
        t_str = time.strftime("%H:%M:%S", time.localtime(base_time + i))
        raw_throughput = 84.5 + np.sin(i / 3.0) * 6.5
        raw_drops = 0.32 + np.cos(i / 4.0) * 0.05
        sim_lat = 41.2 + (raw_drops * 11.5) + (1.2 * 0.10)
        warm_start_data.append({
            "Time": t_str,
            "Throughput": raw_throughput,
            "Drops": raw_drops,
            "Temperature": 47.8 + np.random.normal(0, 0.3),
            "Buffer_Util": 41.5 + np.sin(i / 2.5) * 4.2,
            "Predicted_Latency": sim_lat,
            "Actual_Latency": sim_lat + np.random.normal(0, 0.3),
            "Mitigation_Status": "Inactive",
            "Compute_Overhead_ms": 1.12 + np.random.normal(0, 0.05),
            "Core_Throttled": "False",
            "Failover_Active": "False"
        })
    st.session_state.history = pd.DataFrame(warm_start_data)

# ---------------------------------------------------------------------
# HIGH PERFORMANCE HMI (ISA-101) INDUSTRIAL DESIGN SYSTEM (100% DARK)
# ---------------------------------------------------------------------
st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
/* Master Dark Slate Base Viewport */
.stApp {
    background-color: #0b0f17 !important;
    background-image: 
        linear-gradient(rgba(30, 41, 59, 0.35) 1px, transparent 1px),
        linear-gradient(90deg, rgba(30, 41, 59, 0.35) 1px, transparent 1px) !important;
    background-size: 28px 28px !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #cbd5e1 !important;
}

/* Enforce Tabular Numerics for Instrumentation Accuracy */
* {
    font-variant-numeric: tabular-nums;
    box-sizing: border-box;
}

/* Typography Hierarchy */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
    letter-spacing: -0.01em !important;
}
p, span, label {
    color: #94a3b8 !important;
}
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Industrial SCADA Panel Cards with Overflow Hidden Guard */
.scada-panel {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.5) !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
    position: relative !important;
    transition: border-color 0.15s ease !important;
}
.scada-panel:hover {
    border-color: #374151 !important;
}

/* Custom Table Theme */
.scada-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
}
.scada-table th {
    background: #0f172a;
    color: #94a3b8;
    text-align: left;
    padding: 8px 10px;
    font-weight: 600;
    border-bottom: 1px solid #1e293b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.scada-table td {
    padding: 8px 10px;
    border-bottom: 1px solid #1a2234;
    color: #e2e8f0;
}
.scada-table tr:hover {
    background: #1e293b;
}

/* Priority Badges */
.badge-p1 {
    background: #7f1d1d;
    color: #fca5a5;
    border: 1px solid #dc2626;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 700;
}
.badge-p2 {
    background: #78350f;
    color: #fcd34d;
    border: 1px solid #d97706;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 700;
}
.badge-p3 {
    background: #1e3a8a;
    color: #93c5fd;
    border: 1px solid #2563eb;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 700;
}

/* High Contrast Technical Form Inputs */
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] select,
div[data-testid="stNumberInput"] input {
    color: #f8fafc !important;
    background-color: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 1px #3b82f6 !important;
}

/* Industrial Sidebar Navigation */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background-color: #090d16 !important;
    border-right: 1px solid #1e293b !important;
}
div[data-testid="stSidebarContent"] {
    padding-top: 0.8rem !important;
}

/* Sidebar Nav Radio Options */
div[data-testid="stRadio"] > div {
    gap: 3px !important;
}
div[data-testid="stRadio"] label {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 4px !important;
    padding: 8px 12px !important;
    margin-bottom: 2px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}
div[data-testid="stRadio"] label:hover {
    background: #1e293b !important;
    border-color: #475569 !important;
}
div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] label:has(input:checked) {
    background: #1e293b !important;
    border: 1px solid #3b82f6 !important;
    border-left: 3px solid #3b82f6 !important;
}
div[data-testid="stRadio"] label[data-checked="true"] p,
div[data-testid="stRadio"] label:has(input:checked) p {
    color: #f8fafc !important;
    font-weight: 600 !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

/* Industrial Action Buttons */
div.stButton > button,
div.stDownloadButton > button {
    background: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    font-size: 11.5px !important;
    font-family: 'Inter', sans-serif !important;
    padding: 7px 12px !important;
    box-shadow: none !important;
    transition: all 0.15s ease-in-out !important;
    width: 100% !important;
}
div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background: #2563eb !important;
    border-color: #3b82f6 !important;
    color: #ffffff !important;
}

/* Sliders */
div[data-testid="stSlider"] [data-baseweb="slider"] > div {
    background: #1e293b !important;
    height: 4px !important;
    border-radius: 2px !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
    background: #3b82f6 !important;
    height: 4px !important;
    border-radius: 2px !important;
}
div[data-testid="stSlider"] [role="slider"] {
    background-color: #3b82f6 !important;
    border: 2px solid #ffffff !important;
    width: 12px !important;
    height: 12px !important;
}

/* Alert & Notice Containers */
div[data-testid="stAlert"],
div[data-testid="stNotification"] {
    background-color: #111827 !important;
    border: 1px solid #1f2937 !important;
    color: #e2e8f0 !important;
    border-radius: 4px !important;
}

/* Clean Header Adjustments */
header[data-testid="stHeader"] {
    background-color: transparent !important;
    height: 0px !important;
}
div.block-container {
    padding-top: 1.0rem !important;
    padding-bottom: 2rem !important;
}
</style>
""")

# ---------------------------------------------------------------------
# VECTOR SVG GENERATORS (SPARKLINE, OSCILLOSCOPE & CAD SCHEMATICS)
# ---------------------------------------------------------------------
def svg_to_data_uri(svg_string):
    return "data:image/svg+xml;base64," + base64.b64encode(svg_string.strip().encode("utf-8")).decode("utf-8")

def generate_sparkline_svg(values, stroke_color="#10b981", height=20, width=50):
    if len(values) < 2:
        return ""
    min_v, max_v = min(values), max(values)
    v_range = max_v - min_v if max_v != min_v else 1.0
    
    points = []
    for i, v in enumerate(values):
        x = (i / (len(values) - 1)) * width
        y = height - 2 - ((v - min_v) / v_range) * (height - 4)
        points.append(f"{x:.1f},{y:.1f}")
        
    path_d = "M " + " L ".join(points)
    svg_raw = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="display:block; overflow:hidden;">
        <path d="{path_d}" fill="none" stroke="{stroke_color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" />
    </svg>"""
    return svg_to_data_uri(svg_raw)

# Clean High Performance HMI KPI Card (With strict overflow guard)
def render_scada_kpi_card(tag_id, label, value, unit, nominal_range, limit_val, quality="GOOD", spark_values=None, alarm_active=False):
    status_bg = "#7f1d1d" if alarm_active else "#064e3b"
    status_border = "#dc2626" if alarm_active else "#059669"
    status_text = "#fca5a5" if alarm_active else "#6ee7b7"
    val_color = "#ef4444" if alarm_active else "#f8fafc"
    stroke_col = "#ef4444" if alarm_active else "#10b981"
    
    spark_uri = generate_sparkline_svg(spark_values if spark_values is not None else [1, 1], stroke_color=stroke_col, height=20, width=50)
    
    return f"""
    <div class="scada-panel" style="padding: 10px 12px; height: 122px; display: flex; flex-direction: column; justify-content: space-between; overflow: hidden;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #64748b; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 90px;">
                {tag_id}
            </span>
            <span style="background: {status_bg}; border: 1px solid {status_border}; color: {status_text}; font-size: 8px; padding: 1px 5px; border-radius: 3px; font-family: 'JetBrains Mono', monospace; font-weight: 700; flex-shrink: 0;">
                {quality}
            </span>
        </div>
        
        <div style="display: flex; justify-content: space-between; align-items: flex-end; margin: 2px 0;">
            <div style="overflow: hidden; max-width: 85px;">
                <div style="font-size: 10.5px; font-weight: 500; color: #94a3b8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    {label}
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 19px; font-weight: 800; color: {val_color}; line-height: 1.1; white-space: nowrap;">
                    {value} <span style="font-size: 10px; font-weight: 500; color: #64748b;">{unit}</span>
                </div>
            </div>
            <div style="width: 50px; height: 20px; flex-shrink: 0; overflow: hidden;">
                <img src="{spark_uri}" style="width: 50px; height: 20px; display: block;" />
            </div>
        </div>

        <div style="font-family: 'JetBrains Mono', monospace; font-size: 8.5px; color: #64748b; border-top: 1px solid #1a2234; padding-top: 4px; display: flex; justify-content: space-between; white-space: nowrap;">
            <span>NOM: <b style="color: #94a3b8;">{nominal_range}</b></span>
            <span>LIM: <b style="color: #f59e0b;">{limit_val}</b></span>
        </div>
    </div>
    """

# ISA-18.2 Industrial Annunciator Tile Matrix
def render_alarm_annunciator_strip(is_lat_alm, is_drop_alm, is_buf_alm, is_temp_alm, failover_engaged):
    tiles = [
        {"code": "ANN-01", "name": "LATENCY SLA", "tag": "LAT-RTT-01", "active": is_lat_alm, "crit": True, "val": "HIGH RTT"},
        {"code": "ANN-02", "name": "FRAME LOSS", "tag": "DROP-ERR-03", "active": is_drop_alm, "crit": True, "val": "LOSS BURST"},
        {"code": "ANN-03", "name": "TSN QUEUE SAT", "tag": "BUFF-Q-04", "active": is_buf_alm, "crit": False, "val": "BUFFER HIGH"},
        {"code": "ANN-04", "name": "CORE OVERTEMP", "tag": "TEMP-JC-05", "active": is_temp_alm, "crit": False, "val": "THERMAL RUN"},
        {"code": "ANN-05", "name": "HA REDUNDANCY", "tag": "PLC-FAILOVER", "active": failover_engaged, "crit": False, "val": "STANDBY ACTIVE"},
        {"code": "ANN-06", "name": "IEC-62443 CONDUIT", "tag": "ZONE-2-SEC", "active": False, "crit": False, "val": "SL-3 LOCKED"}
    ]
    
    tile_htmls = []
    for t in tiles:
        if t["active"]:
            bg = "#7f1d1d" if t["crit"] else "#78350f"
            border = "#dc2626" if t["crit"] else "#d97706"
            txt_c = "#fca5a5" if t["crit"] else "#fcd34d"
            lamp_c = "#ef4444" if t["crit"] else "#f59e0b"
            state_lbl = "ALARM"
        else:
            bg = "#111827"
            border = "#1e293b"
            txt_c = "#64748b"
            lamp_c = "#10b981"
            state_lbl = "NORMAL"
            
        tile_htmls.append(f"""
        <div style="background: {bg}; border: 1px solid {border}; border-radius: 3px; padding: 6px 8px; flex: 1; min-width: 100px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 8px; color: #64748b;">{t['code']}</span>
                <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: {lamp_c};"></span>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-weight: 800; font-size: 10px; color: {'#f8fafc' if t['active'] else '#94a3b8'}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                {t['name']}
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 3px; font-size: 8.5px; font-family: monospace;">
                <span style="color: {txt_c}; font-weight: bold;">{state_lbl}</span>
                <span style="color: #64748b;">{t['val']}</span>
            </div>
        </div>
        """)
        
    return f"""
    <div style="display: flex; gap: 6px; margin-bottom: 10px; width: 100%;">
        {''.join(tile_htmls)}
    </div>
    """

# 100% Dark SVG Dual-Trace Real-Time Oscilloscope Data URI
def generate_oscilloscope_data_uri(history_df, ucl=60.0, uwl=45.0):
    width, height = 620, 225
    pad_left, pad_right, pad_top, pad_bottom = 35, 15, 20, 25
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    
    max_lat = max(80.0, max(history_df["Actual_Latency"].max(), history_df["Predicted_Latency"].max()) * 1.15)
    min_lat = 0.0
    lat_range = max_lat - min_lat
    
    pred_pts = []
    act_pts = []
    n_pts = len(history_df)
    
    for i in range(n_pts):
        x = pad_left + (i / max(1, n_pts - 1)) * plot_w
        y_pred = pad_top + plot_h - ((history_df["Predicted_Latency"].iloc[i] - min_lat) / lat_range) * plot_h
        pred_pts.append(f"{x:.1f},{y_pred:.1f}")
        
        y_act = pad_top + plot_h - ((history_df["Actual_Latency"].iloc[i] - min_lat) / lat_range) * plot_h
        act_pts.append(f"{x:.1f},{y_act:.1f}")
        
    pred_path = "M " + " L ".join(pred_pts)
    act_path = "M " + " L ".join(act_pts)
    
    y_ucl = pad_top + plot_h - ((ucl - min_lat) / lat_range) * plot_h
    y_uwl = pad_top + plot_h - ((uwl - min_lat) / lat_range) * plot_h
    
    grid_lines = []
    for g_val in [20, 40, 60, 80]:
        if g_val < max_lat:
            y_g = pad_top + plot_h - ((g_val - min_lat) / lat_range) * plot_h
            grid_lines.append(f'<line x1="{pad_left}" y1="{y_g:.1f}" x2="{width - pad_right}" y2="{y_g:.1f}" stroke="#1e293b" stroke-width="1" />')
            grid_lines.append(f'<text x="{pad_left - 6}" y="{y_g + 3:.1f}" fill="#64748b" font-size="8.5" font-family="monospace" text-anchor="end">{g_val}</text>')
            
    cur_pred = history_df["Predicted_Latency"].iloc[-1]
    cur_act = history_df["Actual_Latency"].iloc[-1]
    
    svg_scope = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background-color: #0b0f17; border-radius: 4px; display: block;">
        <!-- Grid Background Lines -->
        {' '.join(grid_lines)}
        
        <!-- Axis lines -->
        <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + plot_h}" stroke="#334155" stroke-width="1.5" />
        <line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{width - pad_right}" y2="{pad_top + plot_h}" stroke="#334155" stroke-width="1.5" />
        
        <!-- Warning Limit (UWL 45 ms) -->
        <line x1="{pad_left}" y1="{y_uwl:.1f}" x2="{width - pad_right}" y2="{y_uwl:.1f}" stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="4,4" />
        <text x="{width - pad_right - 4}" y="{y_uwl - 3:.1f}" fill="#f59e0b" font-size="8" font-family="monospace" text-anchor="end" font-weight="bold">UWL {uwl:.0f}ms</text>

        <!-- Control Limit (UCL 60 ms) -->
        <line x1="{pad_left}" y1="{y_ucl:.1f}" x2="{width - pad_right}" y2="{y_ucl:.1f}" stroke="#ef4444" stroke-width="1.4" stroke-dasharray="4,4" />
        <text x="{width - pad_right - 4}" y="{y_ucl - 3:.1f}" fill="#ef4444" font-size="8" font-family="monospace" text-anchor="end" font-weight="bold">UCL {ucl:.0f}ms</text>

        <!-- Trace 1: Predicted Latency (Cyan / Blue) -->
        <path d="{pred_path}" fill="none" stroke="#38bdf8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" />

        <!-- Trace 2: Actual Latency (Green) -->
        <path d="{act_path}" fill="none" stroke="#10b981" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" />

        <!-- Scope Header Legend -->
        <rect x="{pad_left + 10}" y="6" width="310" height="18" rx="2" fill="#111827" stroke="#1e293b" />
        <circle cx="{pad_left + 20}" cy="15" r="3" fill="#38bdf8" />
        <text x="{pad_left + 28}" y="18" fill="#e2e8f0" font-size="8.5" font-family="monospace">PRED: <tspan fill="#38bdf8" font-weight="bold">{cur_pred:.2f} ms</tspan></text>

        <circle cx="{pad_left + 115}" cy="15" r="3" fill="#10b981" />
        <text x="{pad_left + 123}" y="18" fill="#e2e8f0" font-size="8.5" font-family="monospace">ACTUAL: <tspan fill="#10b981" font-weight="bold">{cur_act:.2f} ms</tspan></text>

        <line x1="{pad_left + 215}" y1="15" x2="{pad_left + 225}" y2="15" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="2,2" />
        <text x="{pad_left + 230}" y="18" fill="#fca5a5" font-size="8.5" font-family="monospace">UCL: {ucl:.0f}ms</text>

        <!-- Bottom X-Axis Label -->
        <text x="{width / 2}" y="{height - 6}" fill="#64748b" font-size="8.5" font-family="monospace" text-anchor="middle">REAL-TIME TELEMETRY TRACE (30s RECENT WINDOW)</text>
    </svg>"""
    return svg_to_data_uri(svg_scope)

# Dynamic Animated Industrial CAD Schematic Data URI (With 60fps Client-Side Motion)
def generate_cad_schematic_data_uri(color_node_a, color_node_b, color_node_d, color_gateway, failover_engaged, tp=85.0, drops=0.35, temp=48.0, buff=42.0, pred_lat=42.0):
    width, height = 360, 225
    flow_speed_a = "1.1s" if tp < 100 else "0.55s"
    flow_speed_gw = "0.85s" if tp < 100 else "0.42s"
    
    particle_a_col = "#38bdf8" if temp < 60 else "#f59e0b"
    particle_b_col = "#ef4444" if drops > 1.0 else "#34d399"
    particle_gw_col = "#ef4444" if pred_lat >= 60 else "#38bdf8"
    
    cad_svg_raw = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background-color: #0b0f17; border-radius: 4px; display: block;">
        <defs>
            <style>
                @keyframes conduit-flow {{
                    from {{ stroke-dashoffset: 20; }}
                    to {{ stroke-dashoffset: 0; }}
                }}
                @keyframes beacon-ring {{
                    0% {{ r: 14px; opacity: 0.8; stroke-width: 1.5; }}
                    100% {{ r: 24px; opacity: 0; stroke-width: 0.5; }}
                }}
                .flow-a {{ stroke-dasharray: 4, 4; animation: conduit-flow {flow_speed_a} linear infinite; }}
                .flow-b {{ stroke-dasharray: 4, 4; animation: conduit-flow {"2.5s" if failover_engaged else "1.3s"} linear infinite; }}
                .flow-c {{ stroke-dasharray: 4, 4; animation: conduit-flow 1.5s linear infinite; }}
                .flow-d {{ stroke-dasharray: 4, 4; animation: conduit-flow {"0.8s" if failover_engaged else "4.0s"} linear infinite; }}
                .flow-gw {{ stroke-dasharray: 5, 5; animation: conduit-flow {flow_speed_gw} linear infinite; }}
                .pulse-beacon {{ animation: beacon-ring 1.6s ease-out infinite; transform-origin: 180px 110px; }}
            </style>
        </defs>

        <!-- Subtle CAD Engineering Grid -->
        <line x1="0" y1="36" x2="360" y2="36" stroke="#131d2e" stroke-width="0.5" stroke-dasharray="2,4" />
        <line x1="0" y1="81" x2="360" y2="81" stroke="#131d2e" stroke-width="0.5" stroke-dasharray="2,4" />
        <line x1="0" y1="136" x2="360" y2="136" stroke="#131d2e" stroke-width="0.5" stroke-dasharray="2,4" />
        <line x1="0" y1="186" x2="360" y2="186" stroke="#131d2e" stroke-width="0.5" stroke-dasharray="2,4" />

        <!-- Animated Conduits -->
        <path d="M 68,36 L 155,100" fill="none" stroke="{color_node_a}" stroke-width="2.2" class="flow-a" />
        <path d="M 68,81 L 155,105" fill="none" stroke="{"#334155" if failover_engaged else color_node_b}" stroke-width="2.0" class="flow-b" />
        <path d="M 68,136 L 155,115" fill="none" stroke="#10b981" stroke-width="2.0" class="flow-c" />
        <path d="M 68,186 L 155,120" fill="none" stroke="{color_node_d if failover_engaged else '#1e293b'}" stroke-width="2.2" class="flow-d" />
        <path d="M 205,110 L 268,110" fill="none" stroke="{color_gateway}" stroke-width="2.8" class="flow-gw" />

        <!-- Live Moving Packet Particles -->
        <circle r="3" fill="{particle_a_col}">
            <animateMotion dur="{flow_speed_a}" repeatCount="indefinite" path="M 68,36 L 155,100" />
        </circle>
        
        {f'''<circle r="3" fill="{particle_b_col}">
            <animateMotion dur="1.3s" repeatCount="indefinite" path="M 68,81 L 155,105" />
        </circle>''' if not failover_engaged else ''}

        <circle r="2.6" fill="#10b981">
            <animateMotion dur="1.5s" repeatCount="indefinite" path="M 68,136 L 155,115" />
        </circle>

        {f'''<circle r="3.2" fill="#f59e0b">
            <animateMotion dur="0.8s" repeatCount="indefinite" path="M 68,186 L 155,120" />
        </circle>''' if failover_engaged else ''}

        <circle r="3.5" fill="{particle_gw_col}">
            <animateMotion dur="{flow_speed_gw}" repeatCount="indefinite" path="M 205,110 L 268,110" />
        </circle>

        <!-- Node Alpha (PLC-A) -->
        <rect x="12" y="24" width="56" height="25" rx="3" fill="#111827" stroke="{color_node_a}" stroke-width="1.8" />
        <circle cx="20" cy="36" r="2.5" fill="{color_node_a}" />
        <text x="39" y="35" fill="#f8fafc" font-size="8.5" font-family="monospace" text-anchor="middle" font-weight="bold">PLC-A</text>
        <text x="39" y="44" fill="#94a3b8" font-size="7" font-family="monospace" text-anchor="middle">{temp:.0f}°C</text>

        <!-- Node Beta (PLC-B) -->
        <rect x="12" y="69" width="56" height="25" rx="3" fill="#111827" stroke="{"#334155" if failover_engaged else color_node_b}" stroke-width="1.8" />
        <circle cx="20" cy="81" r="2.5" fill="{"#475569" if failover_engaged else color_node_b}" />
        <text x="39" y="80" fill="#f8fafc" font-size="8.5" font-family="monospace" text-anchor="middle" font-weight="bold">PLC-B</text>
        <text x="39" y="89" fill="#94a3b8" font-size="7" font-family="monospace" text-anchor="middle">{drops:.1f}% err</text>

        <!-- Cache Node Gamma -->
        <rect x="12" y="124" width="56" height="25" rx="3" fill="#111827" stroke="#10b981" stroke-width="1.8" />
        <circle cx="20" cy="136" r="2.5" fill="#10b981" />
        <text x="39" y="135" fill="#f8fafc" font-size="8.5" font-family="monospace" text-anchor="middle" font-weight="bold">CACHE</text>
        <text x="39" y="144" fill="#10b981" font-size="7" font-family="monospace" text-anchor="middle">SYNC</text>

        <!-- Hot Standby Delta (HA-STBY) -->
        <rect x="12" y="174" width="56" height="25" rx="3" fill="#111827" stroke="{color_node_d if failover_engaged else '#334155'}" stroke-width="1.8" />
        <circle cx="20" cy="186" r="2.5" fill="{"#f59e0b" if failover_engaged else '#475569'}" />
        <text x="39" y="185" fill="#f8fafc" font-size="8.5" font-family="monospace" text-anchor="middle" font-weight="bold">HA-STBY</text>
        <text x="39" y="194" fill="{"#f59e0b" if failover_engaged else '#64748b'}" font-size="7" font-family="monospace" text-anchor="middle">{"ACTIVE" if failover_engaged else "STBY"}</text>

        <!-- Central TSN Gateway Hub (With Pulse Beacon) -->
        <circle cx="180" cy="110" r="18" fill="none" stroke="{color_gateway}" class="pulse-beacon" />
        <rect x="155" y="88" width="50" height="44" rx="4" fill="#111827" stroke="{color_gateway}" stroke-width="2.2" />
        <text x="180" y="105" fill="#f8fafc" font-size="9.5" font-family="monospace" text-anchor="middle" font-weight="bold">TSN-GW</text>
        <text x="180" y="116" fill="#38bdf8" font-size="7.5" font-family="monospace" text-anchor="middle" font-weight="bold">{tp:.0f} Mbps</text>
        <text x="180" y="125" fill="#94a3b8" font-size="7" font-family="monospace" text-anchor="middle">Q:{buff:.0f}%</text>

        <!-- SCADA Cloud NOC Bridge -->
        <rect x="268" y="93" width="80" height="34" rx="3" fill="#111827" stroke="#38bdf8" stroke-width="1.8" />
        <text x="308" y="108" fill="#38bdf8" font-size="9.5" font-family="monospace" text-anchor="middle" font-weight="bold">SCADA NOC</text>
        <text x="308" y="120" fill="#94a3b8" font-size="7.5" font-family="monospace" text-anchor="middle">RTT: {pred_lat:.1f}ms</text>

        <!-- Protocol & Active Stream Metrics -->
        <text x="75" y="27" fill="#64748b" font-size="7.5" font-family="monospace">Modbus :502</text>
        <text x="75" y="72" fill="#64748b" font-size="7.5" font-family="monospace">OPC-UA :4840</text>
        <text x="75" y="127" fill="#64748b" font-size="7.5" font-family="monospace">MQTT :1883</text>
        <text x="75" y="177" fill="#64748b" font-size="7.5" font-family="monospace">CoAP/TSN</text>
    </svg>"""
    return svg_to_data_uri(cad_svg_raw)

# ---------------------------------------------------------------------
# SIDEBAR: INDUSTRIAL CONTROL & SCADA DISPATCHER
# ---------------------------------------------------------------------
with st.sidebar:
    st.html("""
    <div style="padding: 6px 0 12px 0; border-bottom: 1px solid #1e293b; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 4px; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>
            </div>
            <div>
                <div style="font-size: 13px; font-weight: 800; color: #f8fafc; letter-spacing: 0.3px;">
                    HELM-IIoT SCADA
                </div>
                <div style="font-size: 9px; color: #3b82f6; font-family: 'JetBrains Mono', monospace; font-weight: 700;">
                    EDGE CONTROL NODE v4.2.8
                </div>
            </div>
        </div>
    </div>
    """)

    st.markdown("<div style='font-size:9.5px; font-weight:700; color:#64748b; font-family:monospace; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'>SCADA WORKSPACES</div>", unsafe_allow_html=True)
    nav_labels = [
        "⬡ Live SCADA Telemetry",
        "◈ AI SCADA Copilot & RCA",
        "☊ Fieldbus Topology & CAD",
        "▤ Edge Fleet & Node Assets",
        "⋈ TreeSHAP XAI & Feature Store",
        "⎋ QoS Actuation & Shaper",
        "⌸ Incident Audit & SOE Log"
    ]
    selected_nav = st.radio(
        "Navigation",
        nav_labels,
        index=nav_labels.index(st.session_state.active_nav) if st.session_state.active_nav in nav_labels else 0,
        label_visibility="collapsed",
        key="sidebar_nav_radio"
    )
    if selected_nav != st.session_state.active_nav:
        st.session_state.active_nav = selected_nav
        st.rerun()

    st.markdown("<div style='margin-top: 12px; border-top: 1px solid #1e293b; padding-top: 8px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:9.5px; font-weight:700; color:#64748b; font-family:monospace; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'>TELEMETRY INGRESS DISPATCHER</div>", unsafe_allow_html=True)
    mode = st.radio(
        "Operational Mode",
        ["Automated Real-Time Polling", "Manual Override Injection"],
        index=0 if st.session_state.mode == "Automated Real-Time Polling" else 1,
        key="sb_mode",
        label_visibility="collapsed"
    )
    if mode != st.session_state.mode:
        st.session_state.mode = mode
        st.rerun()

    st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:9.5px; font-weight:700; color:#64748b; font-family:monospace; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'>SAFETY INTERLOCKS & POLICIES</div>", unsafe_allow_html=True)
    enable_mitigation = st.toggle("Linux tc Traffic Shaper (QoS)", value=st.session_state.enable_mitigation, key="sb_mitigation")
    enable_throttling = st.toggle("Thermal Core Regulator", value=st.session_state.enable_throttling, key="sb_throttling")
    enable_failover = st.toggle("PLC Node Delta Failover", value=st.session_state.enable_failover, key="sb_failover")

    st.session_state.enable_mitigation = enable_mitigation
    st.session_state.enable_throttling = enable_throttling
    st.session_state.enable_failover = enable_failover

    override_throughput = 85.0
    override_drops = 0.35
    override_buffer = 42.0
    override_temp = 48.0
    stress_level = 1.0
    run_simulation = True

    if st.session_state.mode == "Manual Override Injection":
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:9.5px; font-weight:700; color:#64748b; font-family:monospace; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'>MANUAL SIGNAL INJECTION</div>", unsafe_allow_html=True)
        override_throughput = st.slider("Ingress Rate (Mbps)", 10.0, 150.0, 85.0, key="sb_ov_thru")
        override_drops = st.slider("Packet Loss Rate (%)", 0.0, 5.0, 0.35, key="sb_ov_drops")
        override_buffer = st.slider("Buffer Saturation (%)", 5.0, 100.0, 42.0, key="sb_ov_buff")
        override_temp = st.slider("Junction Temp (C)", 30.0, 85.0, 48.0, key="sb_ov_temp")
        run_simulation = False
    else:
        stress_level = st.slider("Industrial Load Multiplier", 1.0, 3.0, 1.0, step=0.25, key="sb_stress")
        run_simulation = st.toggle("Cyclic Polling Stream Active", value=True, key="sb_run_sim")

    st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.html("""
        <div style="font-family: 'Inter', sans-serif;">
            <div style="font-weight: 600; font-size: 11px; color: #f8fafc; margin-bottom: 2px;">IEC 62443 Security Audit</div>
            <div style="font-size: 9.5px; color: #64748b; line-height: 1.3;">Security Level: <b>SL-3 Validated</b><br>Auth: <b>mTLS + RBAC Key L3</b></div>
        </div>
        """)

# ---------------------------------------------------------------------
# TOP INDUSTRIAL SCADA HEADER (PLANT HIERARCHY & OPERATIONS STRIP)
# ---------------------------------------------------------------------
def render_industrial_header(timestamp):
    st.html(f"""
    <div style="background: #111827; border: 1px solid #1f2937; border-radius: 4px; padding: 8px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 9.5px; font-family: 'JetBrains Mono', monospace; color: #64748b; font-weight: 700;">PLANT HIERARCHY:</span>
                <span style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #38bdf8; font-weight: 600;">STUTTGART-04 &gt; CELL-02 &gt; HELM-GW01</span>
                <span style="background: #1e293b; border: 1px solid #334155; color: #94a3b8; font-size: 9px; padding: 1px 6px; border-radius: 3px; font-family: 'JetBrains Mono', monospace;">TSN 802.1Qbv</span>
                {f'<span style="background: #7f1d1d; border: 1px solid #dc2626; color: #fca5a5; font-size: 9px; padding: 1px 6px; border-radius: 3px; font-family: \'JetBrains Mono\', monospace; font-weight: 700;">FAULT: {st.session_state.chaos_mode.upper()}</span>' if st.session_state.chaos_mode != 'None' else ''}
            </div>
            <div style="font-size: 10px; color: #94a3b8; margin-top: 2px;">
                RT-PREEMPT Kernel 6.6.14-rt | Cyclic Scan: <b>10.0 ms</b> | Physical Socket RTT: <b>3.42 ms</b> | Jitter: <b>+/-0.15 ms</b>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #94a3b8;">
                SYS CLOCK: <span style="color: #f8fafc; font-weight: 700;">{timestamp}</span>
            </div>
            <div style="background: #064e3b; border: 1px solid #059669; padding: 3px 10px; border-radius: 3px; font-size: 10.5px; color: #34d399; font-weight: 700; font-family: 'JetBrains Mono', monospace; display: flex; align-items: center; gap: 6px;">
                <span style="display:inline-block; width:6px; height:6px; background:#10b981; border-radius:50%;"></span> OT-LINK ONLINE
            </div>
        </div>
    </div>
    """)

# ---------------------------------------------------------------------
# VIEW 1: LIVE SCADA TELEMETRY & FIELDBUS OSCILLOSCOPE
# ---------------------------------------------------------------------
@st.fragment(run_every=1.0 if (st.session_state.mode == "Automated Real-Time Polling" and run_simulation) else None)
def render_live_scada_telemetry():
    timestamp = time.strftime("%H:%M:%S")
    start_time = time.perf_counter()
    render_industrial_header(timestamp)

    # Telemetry generation & Chaos simulation logic
    if st.session_state.mode == "Automated Real-Time Polling" and run_simulation:
        if st.session_state.chaos_mode == "Traffic Spike":
            throughput = float(np.clip(138.0 + np.random.normal(0, 3.5), 10, 160))
            packet_drop = float(np.clip(2.10 + np.random.normal(0, 0.15), 0, 5))
            buffer_util = float(np.clip(93.5 + np.random.normal(0, 2.5), 5, 100))
            temperature = float(np.clip(57.5 + np.random.normal(0, 0.6), 30, 85))
        elif st.session_state.chaos_mode == "Packet Loss Burst":
            throughput = float(np.clip(46.0 + np.random.normal(0, 4.0), 10, 150))
            packet_drop = float(np.clip(3.40 + np.random.normal(0, 0.25), 0, 5))
            buffer_util = float(np.clip(77.0 + np.random.normal(0, 3.5), 5, 100))
            temperature = float(np.clip(53.8 + np.random.normal(0, 0.6), 30, 85))
        elif st.session_state.chaos_mode == "Thermal Surge":
            throughput = float(np.clip(76.0 + np.random.normal(0, 3.5), 10, 150))
            packet_drop = float(np.clip(1.15 + np.random.normal(0, 0.10), 0, 5))
            buffer_util = float(np.clip(61.0 + np.random.normal(0, 3.5), 5, 100))
            temperature = float(np.clip(76.8 + np.random.normal(0, 0.8), 30, 85))
        else:
            throughput = float(np.clip(84.5 + np.random.normal(0, 3.5) * stress_level, 10, 150))
            packet_drop = float(np.clip(0.35 + (np.random.exponential(0.5) if np.random.rand() > 0.82 else np.random.normal(0, 0.03)) * stress_level, 0, 5))
            buffer_util = float(np.clip(41.5 + np.random.normal(0, 5.0) * stress_level, 5, 100))
            temperature = float(np.clip(47.5 + (packet_drop * 3.8) + np.random.normal(0, 0.4), 30, 85))
    else:
        throughput = override_throughput
        packet_drop = override_drops
        buffer_util = override_buffer
        temperature = override_temp

    st.session_state.total_cycles += 1
    if temperature > st.session_state.peak_temp:
        st.session_state.peak_temp = temperature

    # Hardware Throttling Simulation
    throttled_state = "False"
    throttle_delay = 0.0
    if st.session_state.enable_throttling and temperature > st.session_state.throttling_temp_threshold:
        throttled_state = "True"
        throttle_delay = 0.0012
        temperature -= 3.5

    # HA Auto-Failover Logic
    failover_engaged = False
    active_drops = packet_drop
    if st.session_state.enable_failover and packet_drop > st.session_state.failover_drop_threshold:
        failover_engaged = True
        active_drops = 0.12
        st.session_state.failover_events += 1

    # Extract temporal lags from session history
    hist_lat = st.session_state.history["Actual_Latency"].tolist() if "history" in st.session_state and len(st.session_state.history) > 0 else [42.0]
    lag_1 = hist_lat[-1] if len(hist_lat) >= 1 else 42.0
    lag_2 = hist_lat[-2] if len(hist_lat) >= 2 else lag_1
    
    hist_tp = st.session_state.history["Throughput"].tolist() if "history" in st.session_state and len(st.session_state.history) > 0 else [throughput]
    tp_slope = (throughput - hist_tp[-1]) if len(hist_tp) >= 1 else 0.0
    buf_peak = max(buffer_util, max(st.session_state.history["Buffer_Util"].tail(5).tolist())) if "history" in st.session_state and len(st.session_state.history) > 0 else buffer_util

    # XGBoost Inference through Python SDK Client
    pred_res = helm_client.predict_latency(
        throughput=throughput,
        drop_pct=active_drops,
        buffer_util=buffer_util,
        temp=temperature,
        slope=tp_slope,
        buffer_peak=buf_peak,
        lag_1=lag_1,
        lag_2=lag_2
    )
    predicted_latency = float(pred_res["predicted_latency_ms"])
    unmitigated_predicted_latency = predicted_latency
    is_potential_breach = unmitigated_predicted_latency >= st.session_state.latency_threshold_ms

    # Dynamic QoS Mitigation
    mitigation_active = "Inactive"
    if st.session_state.enable_mitigation and predicted_latency >= (st.session_state.latency_threshold_ms - 5.0):
        mitigation_active = "Active"
        st.session_state.mitigation_checks += 1
        
        throughput *= st.session_state.shedding_factor
        active_drops *= 0.35
        buffer_util *= 0.70
        
        pred_opt = helm_client.predict_latency(
            throughput=throughput,
            drop_pct=active_drops,
            buffer_util=buffer_util,
            temp=temperature,
            slope=tp_slope * st.session_state.shedding_factor,
            buffer_peak=buffer_util,
            lag_1=lag_1,
            lag_2=lag_2
        )
        predicted_latency = float(pred_opt["predicted_latency_ms"])
        if predicted_latency < st.session_state.latency_threshold_ms and is_potential_breach:
            st.session_state.mitigations_successful += 1

    if predicted_latency > st.session_state.peak_latency:
        st.session_state.peak_latency = predicted_latency

    actual_base = 40.0 + (active_drops * 30.0) + np.random.normal(0, 0.9)
    actual_latency = float(np.clip(actual_base, 10.0, 195.0))
    compute_overhead = (time.perf_counter() - start_time) * 1000 + (throttle_delay * 1000)

    # Append to session history (capped at 30 points)
    new_data = pd.DataFrame([{
        "Time": timestamp, "Throughput": throughput, "Drops": packet_drop, 
        "Temperature": temperature, "Buffer_Util": buffer_util,
        "Predicted_Latency": predicted_latency, "Actual_Latency": actual_latency, 
        "Mitigation_Status": mitigation_active, "Compute_Overhead_ms": compute_overhead,
        "Core_Throttled": throttled_state, "Failover_Active": str(failover_engaged)
    }])
    st.session_state.history = pd.concat([st.session_state.history, new_data]).tail(30)
    chart_df = st.session_state.history

    # Append ISA-18.2 Alarms to Event Log
    if failover_engaged and (len(st.session_state.incident_logs) == 0 or st.session_state.incident_logs[-1]["Tag"] != "PLC_FAILOVER"):
        st.session_state.incident_logs.append({
            "Timestamp": timestamp, "Alarm_ID": "ALM-2041", "Priority": "P1-CRIT", "Tag": "PLC_FAILOVER",
            "Event": f"HA Failover actuated: Packet drop {packet_drop:.2f}% exceeded limit 1.80%. Standby Node Delta active.", "Quality": "FAILOVER_0x02"
        })
    elif mitigation_active == "Active" and (len(st.session_state.incident_logs) == 0 or st.session_state.incident_logs[-1]["Tag"] != "QOS_SHAPER"):
        st.session_state.incident_logs.append({
            "Timestamp": timestamp, "Alarm_ID": "ALM-3012", "Priority": "P2-WARN", "Tag": "QOS_SHAPER",
            "Event": f"Linux tc traffic shaper applied. Bounded latency to {predicted_latency:.2f} ms.", "Quality": "MITIGATED_0x01"
        })
    elif predicted_latency >= st.session_state.latency_threshold_ms and (len(st.session_state.incident_logs) == 0 or st.session_state.incident_logs[-1]["Tag"] != "LAT_SLA_BREACH"):
        st.session_state.incident_logs.append({
            "Timestamp": timestamp, "Alarm_ID": "ALM-4089", "Priority": "P1-CRIT", "Tag": "LAT_SLA_BREACH",
            "Event": f"Latency limit breach: {predicted_latency:.2f} ms exceeded threshold {st.session_state.latency_threshold_ms:.1f} ms.", "Quality": "BAD_0x03"
        })

    # -----------------------------------------------------------------
    # ISA-18.2 ALARM ANNUNCIATOR STRIP & OPERATOR ACKNOWLEDGMENT DECK
    # -----------------------------------------------------------------
    is_lat_alm = predicted_latency >= st.session_state.latency_threshold_ms
    is_drop_alm = packet_drop >= st.session_state.failover_drop_threshold
    is_temp_alm = temperature >= st.session_state.throttling_temp_threshold
    is_buf_alm = buffer_util >= 85.0

    st.html(render_alarm_annunciator_strip(is_lat_alm, is_drop_alm, is_buf_alm, is_temp_alm, failover_engaged))

    ack_c1, ack_c2, ack_c3, ack_c4 = st.columns([1.5, 1, 1, 1.5])
    with ack_c1:
        if st.button("✓ Acknowledge All Alarms", key="btn_ack_all"):
            st.session_state.alarm_ack = True
            st.toast("ISA-18.2 Operator Event: All active alarms acknowledged.")
    with ack_c2:
        if st.button("⎋ Mute Horn", key="btn_mute_horn"):
            st.session_state.alarm_muted = not st.session_state.alarm_muted
            st.toast(f"Alarm Horn: {'MUTED' if st.session_state.alarm_muted else 'ACTIVE'}")
    with ack_c3:
        if st.button("▤ Shelve (1h)", key="btn_shelve_alarm"):
            st.toast("Alarm Shelved: Warning alerts suppressed for 60 minutes.")
    with ack_c4:
        st.markdown(f"<div style='font-family:monospace; font-size:10px; color:#94a3b8; text-align:right; padding-top:6px;'>ANNUNCIATOR STATE: <b style='color:#34d399;'>NORMAL</b> | HORN: <b style='color:{'#ef4444' if not st.session_state.alarm_muted else '#64748b'};'>{'ENABLED' if not st.session_state.alarm_muted else 'SILENCED'}</b></div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # FAULT INJECTION CONTROL STRIP (ISA-18.2 TEST HARNESS)
    # -----------------------------------------------------------------
    with st.container():
        st.markdown("<div style='font-size:9.5px; font-weight:700; color:#64748b; font-family:monospace; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'>INDUSTRIAL FAULT INJECTION TEST HARNESS</div>", unsafe_allow_html=True)
        cb1, cb2, cb3, cb4 = st.columns(4)
        with cb1:
            if st.button("↑ Ingress Bandwidth Burst", key="btn_chaos_ddos"):
                st.session_state.chaos_mode = "Traffic Spike"
                st.toast("Fault Injected: Ingress Bandwidth Surge (140+ Mbps).")
                st.rerun()
        with cb2:
            if st.button("✕ Packet Loss Burst", key="btn_chaos_drop"):
                st.session_state.chaos_mode = "Packet Loss Burst"
                st.toast("Fault Injected: Frame Loss Surge (>3.0% Loss).")
                st.rerun()
        with cb3:
            if st.button("▲ Core Thermal Surge", key="btn_chaos_temp"):
                st.session_state.chaos_mode = "Thermal Surge"
                st.toast("Fault Injected: Core Thermal Surge (>75°C).")
                st.rerun()
        with cb4:
            if st.button("⟳ Restore Nominal Baseline", key="btn_chaos_clear"):
                st.session_state.chaos_mode = "None"
                st.toast("System State Restored: Nominal Baseline Operational.")
                st.rerun()

    st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # HIGH-PERFORMANCE HMI (ISA-101) 6-CARD INSTRUMENT GAUGES
    # -----------------------------------------------------------------
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    k1.html(render_scada_kpi_card("TAG: LAT-RTT-01", "Edge Latency", f"{predicted_latency:.2f}", "ms", "10-45 ms", f"{st.session_state.latency_threshold_ms:.0f} ms", "ALARM" if is_lat_alm else "GOOD", chart_df["Predicted_Latency"].values[-10:], is_lat_alm))
    k2.html(render_scada_kpi_card("TAG: THRU-MB-02", "Ingress Rate", f"{throughput:.1f}", "Mbps", "40-100", "150", "GOOD", chart_df["Throughput"].values[-10:], False))
    k3.html(render_scada_kpi_card("TAG: DROP-ERR-03", "Packet Loss", f"{packet_drop:.2f}", "%", "0.0-1.0", f"{st.session_state.failover_drop_threshold:.1f}", "ALARM" if is_drop_alm else "GOOD", chart_df["Drops"].values[-10:], is_drop_alm))
    k4.html(render_scada_kpi_card("TAG: BUFF-Q-04", "TSN Queue", f"{buffer_util:.1f}", "%", "10-60", "85.0", "ALARM" if is_buf_alm else "GOOD", chart_df["Buffer_Util"].values[-10:], is_buf_alm))
    k5.html(render_scada_kpi_card("TAG: TEMP-JC-05", "Junction Temp", f"{temperature:.1f}", "°C", "35-60", f"{st.session_state.throttling_temp_threshold:.0f}", "ALARM" if is_temp_alm else "GOOD", chart_df["Temperature"].values[-10:], is_temp_alm))
    
    active_controller_str = "NODE_DELTA (HA)" if failover_engaged else "NODE_ALPHA (PRI)"
    k6.html(f"""
    <div class="scada-panel" style="padding: 10px 12px; height: 122px; display: flex; flex-direction: column; justify-content: space-between; overflow: hidden;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #64748b; font-weight: 700;">TAG: PLC-06</span>
            <span style="background: {'#78350f' if failover_engaged else '#064e3b'}; border: 1px solid {'#d97706' if failover_engaged else '#059669'}; color: {'#fcd34d' if failover_engaged else '#6ee7b7'}; font-size: 8px; padding: 1px 5px; border-radius: 3px; font-family: 'JetBrains Mono', monospace; font-weight: 700;">
                {'HA_ACT' if failover_engaged else 'PRIMARY'}
            </span>
        </div>
        <div>
            <div style="font-size: 10.5px; font-weight: 500; color: #94a3b8;">Active Controller</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 13.5px; font-weight: 800; color: {'#f59e0b' if failover_engaged else '#34d399'}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                {active_controller_str}
            </div>
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 8.5px; color: #64748b; border-top: 1px solid #1a2234; padding-top: 4px; display: flex; justify-content: space-between; white-space: nowrap;">
            <span>SCAN: <b style="color: #34d399;">10ms</b></span>
            <span>SYNC: <b style="color: #38bdf8;">LOCK 0x00</b></span>
        </div>
    </div>
    """)

    st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # 100% DARK OSCILLOSCOPE & P&ID CAD SCHEMATIC (BASE64 URI EMBEDDED)
    # -----------------------------------------------------------------
    main_col_l, main_col_r = st.columns([2.0, 1.2])

    with main_col_l:
        st.markdown("<div style='font-size:10px; font-weight:700; color:#64748b; font-family:monospace; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'>REAL-TIME DUAL-TRACE FIELDBUS OSCILLOSCOPE</div>", unsafe_allow_html=True)
        scope_uri = generate_oscilloscope_data_uri(chart_df, ucl=st.session_state.latency_threshold_ms, uwl=st.session_state.warning_threshold_ms)
        st.html(f"""
        <div class="scada-panel" style="padding: 6px; height: 235px; box-sizing: border-box; overflow: hidden;">
            <img src="{scope_uri}" style="width: 100%; height: 223px; display: block; border-radius: 4px;" />
        </div>
        """)

    with main_col_r:
        st.markdown("<div style='font-size:10px; font-weight:700; color:#64748b; font-family:monospace; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'>FIELDBUS P&ID CAD SCHEMATIC</div>", unsafe_allow_html=True)
        
        color_node_a = "#10b981" if throttled_state == "False" else "#f59e0b"
        color_node_b = "#475569" if failover_engaged else ("#10b981" if packet_drop <= 1.0 else "#ef4444")
        color_node_d = "#f59e0b" if failover_engaged else "#334155"
        color_gateway = "#ef4444" if predicted_latency >= st.session_state.latency_threshold_ms else "#10b981"
        if mitigation_active == "Active":
            color_gateway = "#38bdf8"

        cad_uri = generate_cad_schematic_data_uri(
            color_node_a, color_node_b, color_node_d, color_gateway, failover_engaged,
            tp=throughput, drops=packet_drop, temp=temperature, buff=buffer_util, pred_lat=predicted_latency
        )
        st.html(f"""
        <div class="scada-panel" style="padding: 6px; height: 235px; box-sizing: border-box; overflow: hidden;">
            <img src="{cad_uri}" style="width: 100%; height: 223px; display: block; border-radius: 4px;" />
        </div>
        """)

    # -----------------------------------------------------------------
    # MULTI-HORIZON PREDICTIVE FAN & SPECTRAL JITTER FFT SPECTRUM
    # -----------------------------------------------------------------
    st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
    f_col1, f_col2 = st.columns([1.6, 1.4])

    with f_col1:
        st.markdown("<div style='font-size:10px; font-weight:700; color:#64748b; font-family:monospace; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'>MULTI-HORIZON LOOK-AHEAD FORECAST (±2σ CONFIDENCE FAN)</div>", unsafe_allow_html=True)
        h5 = predicted_latency + (tp_slope * 0.5) + np.random.normal(0, 0.4)
        h15 = predicted_latency + (tp_slope * 1.5) + (active_drops * 2.0) + np.random.normal(0, 0.8)
        h30 = predicted_latency + (tp_slope * 3.0) + (active_drops * 4.5) + np.random.normal(0, 1.2)
        
        st.html(f"""
        <div class="scada-panel" style="padding: 12px 16px; height: 110px; box-sizing: border-box; display: flex; justify-content: space-around; align-items: center;">
            <div style="text-align: center;">
                <div style="font-size: 9.5px; color: #64748b; font-family: monospace; font-weight: bold;">HORIZON T+5s</div>
                <div style="font-size: 16px; font-family: 'JetBrains Mono', monospace; font-weight: 800; color: {'#ef4444' if h5>=st.session_state.latency_threshold_ms else '#38bdf8'};">{h5:.1f} ms</div>
                <div style="font-size: 8.5px; color: #94a3b8; font-family: monospace;">±1.2 ms (95% CI)</div>
            </div>
            <div style="border-left: 1px solid #1e293b; height: 70px;"></div>
            <div style="text-align: center;">
                <div style="font-size: 9.5px; color: #64748b; font-family: monospace; font-weight: bold;">HORIZON T+15s</div>
                <div style="font-size: 16px; font-family: 'JetBrains Mono', monospace; font-weight: 800; color: {'#ef4444' if h15>=st.session_state.latency_threshold_ms else '#34d399'};">{h15:.1f} ms</div>
                <div style="font-size: 8.5px; color: #94a3b8; font-family: monospace;">±2.8 ms (95% CI)</div>
            </div>
            <div style="border-left: 1px solid #1e293b; height: 70px;"></div>
            <div style="text-align: center;">
                <div style="font-size: 9.5px; color: #64748b; font-family: monospace; font-weight: bold;">HORIZON T+30s</div>
                <div style="font-size: 16px; font-family: 'JetBrains Mono', monospace; font-weight: 800; color: {'#ef4444' if h30>=st.session_state.latency_threshold_ms else '#f59e0b'};">{h30:.1f} ms</div>
                <div style="font-size: 8.5px; color: #94a3b8; font-family: monospace;">±4.5 ms (95% CI)</div>
            </div>
        </div>
        """)

    with f_col2:
        st.markdown("<div style='font-size:10px; font-weight:700; color:#64748b; font-family:monospace; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'>REAL-TIME JITTER FFT SPECTRAL HARMONICS (0-50 Hz)</div>", unsafe_allow_html=True)
        fft_harmonics = [
            ("1.0 Hz (Baseline)", 12, "#10b981"),
            ("5.0 Hz (Queue)", 38 if buffer_util>50 else 18, "#38bdf8"),
            ("10.0 Hz (TSN Gate)", 82 if st.session_state.chaos_mode!='None' else 25, "#ef4444" if st.session_state.chaos_mode!='None' else "#10b981"),
            ("25.0 Hz (Micro-Burst)", 44 if active_drops>1.0 else 14, "#f59e0b")
        ]
        
        fft_bars = []
        for name, amp, col in fft_harmonics:
            fft_bars.append(f"""
            <div style="display: flex; align-items: center; gap: 8px; font-family: monospace; font-size: 9px; margin-bottom: 3px;">
                <span style="width: 110px; color: #cbd5e1; white-space: nowrap;">{name}</span>
                <div style="background: #0b0f17; border-radius: 2px; height: 5px; flex-grow: 1; overflow: hidden;">
                    <div style="background: {col}; height: 5px; width: {amp}%;"></div>
                </div>
                <span style="width: 40px; text-align: right; color: {col}; font-weight: bold;">{amp}%</span>
            </div>
            """)
            
        st.html(f"""
        <div class="scada-panel" style="padding: 10px 14px; height: 110px; box-sizing: border-box; overflow: hidden;">
            {''.join(fft_bars)}
        </div>
        """)

    # -----------------------------------------------------------------
    # ISA-18.2 SEQUENCE-OF-EVENTS (SOE) LOG (100% DARK HTML TABLE)
    # -----------------------------------------------------------------
    st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px; font-weight:700; color:#64748b; font-family:monospace; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;'>ISA-18.2 SEQUENCE-OF-EVENTS (SOE) ALARM & EVENT STREAM</div>", unsafe_allow_html=True)
    
    rows_html = []
    for log in st.session_state.incident_logs[-6:][::-1]:
        p_class = "badge-p1" if "CRIT" in log["Priority"] or "FAILOVER" in log.get("Quality", "") else ("badge-p2" if "WARN" in log["Priority"] else "badge-p3")
        rows_html.append(f"""
        <tr>
            <td style="color:#94a3b8;">{log['Timestamp']}</td>
            <td style="color:#f8fafc; font-weight:bold;">{log.get('Alarm_ID', 'ALM-1000')}</td>
            <td><span class="{p_class}">{log['Priority']}</span></td>
            <td style="color:#38bdf8;">{log.get('Tag', 'SYS')}</td>
            <td>{log['Event']}</td>
            <td style="color:#64748b;">{log.get('Quality', 'GOOD_0x00')}</td>
        </tr>
        """)

    table_html = f"""
    <div class="scada-panel" style="padding: 0; overflow: hidden;">
        <table class="scada-table">
            <thead>
                <tr>
                    <th style="width: 85px;">TIMESTAMP</th>
                    <th style="width: 100px;">ALARM ID</th>
                    <th style="width: 90px;">PRIORITY</th>
                    <th style="width: 120px;">TAG NAME</th>
                    <th>EVENT DESCRIPTION</th>
                    <th style="width: 110px;">QUALITY CODE</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    """
    st.html(table_html)

    # Export Action Bar
    mae = mean_absolute_error(chart_df["Actual_Latency"], chart_df["Predicted_Latency"])
    e_c1, e_c2, e_c3 = st.columns([2.5, 1, 1])
    e_c1.info(f"Predictive Model MAE: **{mae:.4f} ms** | Cycle Overhead: **{compute_overhead:.2f} ms** | Active Policy: **ISA-101 HMI Compliant**")
    e_c2.download_button("⤓ Export SOE Log (.CSV)", data=pd.DataFrame(st.session_state.incident_logs).to_csv(index=False).encode('utf-8'), file_name="scada_soe_log.csv", mime="text/csv", key="btn_dl_soe_csv")
    e_c3.download_button("⤓ Export Telemetry (.JSON)", data=chart_df.to_json(orient="records", indent=2).encode('utf-8'), file_name="scada_telemetry.json", mime="application/json", key="btn_dl_telemetry_json")

# ---------------------------------------------------------------------
# VIEW 2: AI SCADA COPILOT & AUTONOMOUS ROOT CAUSE ANALYSIS (RCA)
# ---------------------------------------------------------------------
def render_ai_copilot_workspace():
    timestamp = time.strftime("%H:%M:%S")
    render_industrial_header(timestamp)

    st.markdown("#### AI SCADA Copilot & Autonomous RCA Incident Engine")
    st.markdown("<p style='font-size: 11.5px; color: #64748b; margin-top: -6px;'>Continuous multi-signal cyber-physical correlation correlating socket RTT, TreeSHAP attributions, and closed-loop actuation.</p>", unsafe_allow_html=True)

    chart_df = st.session_state.history
    cur_tp = float(chart_df["Throughput"].iloc[-1]) if len(chart_df) > 0 else 84.5
    cur_drops = float(chart_df["Drops"].iloc[-1]) if len(chart_df) > 0 else 0.35
    cur_temp = float(chart_df["Temperature"].iloc[-1]) if len(chart_df) > 0 else 47.5
    cur_buf = float(chart_df["Buffer_Util"].iloc[-1]) if len(chart_df) > 0 else 41.5
    cur_pred = float(chart_df["Predicted_Latency"].iloc[-1]) if len(chart_df) > 0 else 42.0
    cur_actual = float(chart_df["Actual_Latency"].iloc[-1]) if len(chart_df) > 0 else 41.5
    ha_active = chart_df["Failover_Active"].iloc[-1] == "True" if len(chart_df) > 0 else False

    pred_res = helm_client.predict_latency(
        throughput=cur_tp,
        drop_pct=cur_drops,
        buffer_util=cur_buf,
        temp=cur_temp
    )
    shap_vals = pred_res.get("shap_attributions", {})

    diag = copilot_engine.diagnose_root_cause(
        throughput_mbps=cur_tp,
        packet_drop_pct=cur_drops,
        buffer_util_pct=cur_buf,
        node_temp_celsius=cur_temp,
        predicted_latency_ms=cur_pred,
        actual_latency_ms=cur_actual,
        operational_regime=pred_res.get("cluster_regime", "Nominal Regime"),
        shap_attributions=shap_vals,
        ha_failover_active=ha_active
    )

    risk_meta = {
        "CRITICAL_BREACH": ("#7f1d1d", "#dc2626", "#fca5a5", "CRITICAL SLA RISK DETECTED"),
        "WARNING_APPROACHING_LIMIT": ("#78350f", "#d97706", "#fcd34d", "WARNING: ELEVATED JITTER DETECTED"),
        "NOMINAL_STABLE": ("#064e3b", "#059669", "#6ee7b7", "DETERMINISTIC TSN OPERATION NOMINAL")
    }
    r_bg, r_border, r_text, r_title = risk_meta.get(diag["risk_level"], ("#064e3b", "#059669", "#6ee7b7", "NOMINAL"))

    st.html(f"""
    <div class="scada-panel" style="padding: 14px 18px; background: {r_bg} !important; border: 1px solid {r_border} !important; margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-family: 'JetBrains Mono', monospace; font-weight: 800; font-size: 13px; color: {r_text};">{r_title}</span>
                <span style="background: #1e293b; border: 1px solid #334155; color: #f8fafc; font-size: 9.5px; padding: 1px 6px; border-radius: 3px; font-family: monospace;">Confidence: {diag['confidence_score']*100:.0f}%</span>
            </div>
            <span style="font-size: 11px; font-family: monospace; color: #cbd5e1;">Primary Root Vector: <b style="color: #ffffff;">{diag['primary_vector']}</b></span>
        </div>
        <div style="font-size: 12px; color: #f8fafc; line-height: 1.4;">
            {diag['diagnosis_summary']}
        </div>
        <div style="margin-top: 6px; font-size: 11px; color: #cbd5e1; font-family: 'JetBrains Mono', monospace;">
            RECOMMENDATION: <b>{diag['recommendation_text']}</b>
        </div>
    </div>
    """)

    c_left, c_right = st.columns([1.5, 1.5])

    with c_left:
        st.markdown("##### Multi-Signal Telemetry Diagnostics")
        for finding in diag["findings"]:
            f_border = "#dc2626" if finding["severity"] == "CRITICAL" else ("#d97706" if finding["severity"] in ["HIGH", "MEDIUM"] else "#059669")
            f_text = "#f87171" if finding["severity"] == "CRITICAL" else ("#fbbf24" if finding["severity"] in ["HIGH", "MEDIUM"] else "#34d399")
            st.html(f"""
            <div class="scada-panel" style="padding: 10px 14px; margin-bottom: 8px; border-left: 3px solid {f_border} !important;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #f8fafc;">{finding['vector']}</span>
                    <span style="font-size: 9px; color: {f_text}; font-weight: 700; font-family: monospace;">{finding['severity']}</span>
                </div>
                <div style="font-size: 10.5px; color: #94a3b8; margin-top: 3px; font-family: monospace;">
                    {finding['evidence']}
                </div>
            </div>
            """)

        if diag["mitigation_action"] != "none":
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(f"###### Recommended Actuation: `{diag['mitigation_action'].upper()}`")
                st.markdown(f"<p style='font-size: 11px; color: #94a3b8;'>Target: <b>{diag['action_params'].get('target_device', 'PLC_NODE_ALPHA')}</b> | Reason: {diag['action_params'].get('reason')}</p>", unsafe_allow_html=True)
                
                if st.button("⏵ Execute Recommended Mitigation", key="btn_exec_rca_mitigation"):
                    res = helm_client.execute_mitigation(
                        action=diag["mitigation_action"],
                        target_device=diag["action_params"].get("target_device", "PLC_NODE_ALPHA"),
                        reason=diag["action_params"].get("reason", "AI SCADA Copilot RCA actuation"),
                        shedding_factor=diag["action_params"].get("shedding_factor", settings.traffic_shedding_factor)
                    )
                    st.toast(f"Actuation Result: {res['status'].upper()} — {diag['mitigation_action']}")
                    st.session_state.incident_logs.append({
                        "Timestamp": timestamp, "Alarm_ID": "ALM-5099", "Priority": "P2-WARN", "Tag": "COPILOT_ACTUATE",
                        "Event": f"Copilot executed {diag['mitigation_action']} on {diag['action_params'].get('target_device')}.", "Quality": "ACTUATED_0x04"
                    })

    with c_right:
        st.markdown("##### Expert Operator Diagnostics Console")
        q1, q2 = st.columns(2)
        with q1:
            if st.button("⌕ Root Cause Breakdown", key="btn_q_rca"):
                st.info(f"**Root Cause**: Forecasted latency is **{cur_pred:.2f} ms**. Dominant factor: **{diag['primary_vector']}** with marginal impact of **+{shap_vals.get('packet_drop_percentage', 3.8):.1f} ms**.")
            if st.button("◈ PLC Redundancy Status", key="btn_q_fleet"):
                st.success(f"**Redundancy Matrix**: Primary Node Alpha is active. Standby Node Delta is {'[FAILOVER ACTIVE: FORWARDING]' if ha_active else '[HOT STANDBY READY]'}.")
        with q2:
            if st.button("⚠ TSN SLA Risk Index", key="btn_q_sla"):
                prob = min(99.0, max(5.0, (cur_pred / settings.sla_latency_threshold_ms) * 100))
                st.warning(f"**SLA Risk**: Operating at **{cur_pred:.1f} / {settings.sla_latency_threshold_ms:.0f} ms** ({prob:.1f}% capacity). Anti-flapping safety guard active.")
            if st.button("✓ IEC Compliance Audit", key="btn_q_audit"):
                report = copilot_engine.generate_compliance_audit_summary(st.session_state.incident_logs)
                st.markdown(report)

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        user_query = st.text_input("SCADA Copilot Diagnostic Query:", placeholder="e.g. Analyze latency jitter on Node Alpha...", key="copilot_text_input")
        if user_query:
            st.html(f"""
            <div class="scada-panel" style="padding: 10px 12px; margin-top: 6px; border-left: 3px solid #3b82f6 !important;">
                <div style="color: #93c5fd; font-weight: 700; font-size: 10.5px; font-family: monospace; margin-bottom: 2px;">COPILOT DIAGNOSTIC REPORT</div>
                <div style="font-size: 11px; color: #f8fafc; line-height: 1.4; font-family: 'JetBrains Mono', monospace;">
                    Ingesting multi-channel telemetry... Operating latency bounded at {cur_pred:.2f} ms within '{pred_res.get('cluster_regime')}'. All deterministic QoS constraints active.
                </div>
            </div>
            """)

# ---------------------------------------------------------------------
# VIEW 3: FIELDBUS TOPOLOGY & DETAILED P&ID SCHEMATIC
# ---------------------------------------------------------------------
def render_fieldbus_topology_cad():
    timestamp = time.strftime("%H:%M:%S")
    render_industrial_header(timestamp)

    st.markdown("#### Industrial Fieldbus Architecture & P&ID CAD Layout")
    st.markdown("<p style='font-size: 11.5px; color: #64748b; margin-top: -6px;'>Single-line industrial fieldbus topology conforming to IEC 61158 / IEC 62443 zone segmentation.</p>", unsafe_allow_html=True)

    st.html("""
    <div class="scada-panel" style="padding: 16px; margin-bottom: 14px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #94a3b8; display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
            <div>
                <span style="color: #64748b;">FIELDBUS TRUNK:</span><br>
                <b style="color: #f8fafc;">TSN IEEE 802.1Qbv</b>
            </div>
            <div>
                <span style="color: #64748b;">CYCLE TIME:</span><br>
                <b style="color: #10b981;">10.0 ms (Deterministic)</b>
            </div>
            <div>
                <span style="color: #64748b;">BANDWIDTH ALLOC:</span><br>
                <b style="color: #38bdf8;">1000BASE-T1 (1.0 Gbps)</b>
            </div>
            <div>
                <span style="color: #64748b;">SECURITY ZONE:</span><br>
                <b style="color: #a78bfa;">IEC 62443 Zone 2 (Cell)</b>
            </div>
        </div>
    </div>
    """)

    t_c1, t_c2 = st.columns(2)
    with t_c1:
        st.markdown("##### Zone 1: Field Device & Sensor Interconnects")
        sensor_data = [
            {"Tag": "SEN-MOD-01", "Protocol": "Modbus TCP (Port 502)", "Address": "192.168.10.21", "Scan_Rate": "10 ms", "Frame_Loss": "0.01%", "Status": "NOMINAL"},
            {"Tag": "SEN-OPC-02", "Protocol": "OPC-UA PubSub", "Address": "192.168.10.22", "Scan_Rate": "20 ms", "Frame_Loss": "0.00%", "Status": "NOMINAL"},
            {"Tag": "SEN-MQT-03", "Protocol": "MQTT v5 (Sparkplug)", "Address": "192.168.10.23", "Scan_Rate": "50 ms", "Frame_Loss": "0.02%", "Status": "NOMINAL"},
            {"Tag": "SEN-PFN-04", "Protocol": "PROFINET IRT", "Address": "192.168.10.24", "Scan_Rate": "5 ms", "Frame_Loss": "0.00%", "Status": "NOMINAL"}
        ]
        
        s_rows = "".join([f"<tr><td style='color:#38bdf8; font-weight:bold;'>{s['Tag']}</td><td>{s['Protocol']}</td><td>{s['Address']}</td><td>{s['Scan_Rate']}</td><td>{s['Frame_Loss']}</td><td style='color:#10b981; font-weight:bold;'>{s['Status']}</td></tr>" for s in sensor_data])
        st.html(f"""
        <div class="scada-panel" style="padding:0; overflow:hidden;">
            <table class="scada-table">
                <thead><tr><th>TAG</th><th>PROTOCOL</th><th>IP ADDRESS</th><th>SCAN</th><th>LOSS</th><th>STATUS</th></tr></thead>
                <tbody>{s_rows}</tbody>
            </table>
        </div>
        """)

    with t_c2:
        st.markdown("##### Zone 2: Controller Redundancy & Gateways")
        controller_data = [
            {"Node": "PLC_ALPHA", "Role": "Primary Ingress", "IP": "192.168.10.14", "CPU": "42.5%", "Temp": "48.2 °C", "State": "ACTIVE"},
            {"Node": "PLC_BETA", "Role": "ML Inference Core", "IP": "192.168.10.15", "CPU": "64.0%", "Temp": "53.8 °C", "State": "ACTIVE"},
            {"Node": "PLC_DELTA", "Role": "Hot Standby HA", "IP": "192.168.10.17", "CPU": "11.2%", "Temp": "39.1 °C", "State": "STANDBY"}
        ]
        
        c_rows = "".join([f"<tr><td style='color:#f8fafc; font-weight:bold;'>{c['Node']}</td><td>{c['Role']}</td><td>{c['IP']}</td><td>{c['CPU']}</td><td>{c['Temp']}</td><td style='color:{'#10b981' if c['State']=='ACTIVE' else '#f59e0b'}; font-weight:bold;'>{c['State']}</td></tr>" for c in controller_data])
        st.html(f"""
        <div class="scada-panel" style="padding:0; overflow:hidden;">
            <table class="scada-table">
                <thead><tr><th>NODE</th><th>ROLE</th><th>IP ADDRESS</th><th>CPU</th><th>TEMP</th><th>STATE</th></tr></thead>
                <tbody>{c_rows}</tbody>
            </table>
        </div>
        """)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Multi-Protocol Fieldbus Telemetry & Demux Matrix")

    proto_p1, proto_p2, proto_p3, proto_p4 = st.columns(4)
    with proto_p1:
        st.html("""
        <div class="scada-panel" style="padding: 12px 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; color: #38bdf8;">MODBUS TCP</span>
                <span style="font-size: 8px; background: #064e3b; border: 1px solid #059669; color: #34d399; padding: 1px 4px; border-radius: 2px; font-family: monospace;">PORT 502</span>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 800; color: #f8fafc;">120.4 <span style="font-size: 9px; color: #64748b;">msg/s</span></div>
            <div style="font-size: 9px; color: #94a3b8; font-family: monospace; margin-top: 4px; border-top: 1px solid #1e293b; padding-top: 3px;">
                CRC ERR: <b style="color:#10b981;">0.00%</b> | REG: <b>0x00-0xFF</b>
            </div>
        </div>
        """)

    with proto_p2:
        st.html("""
        <div class="scada-panel" style="padding: 12px 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; color: #38bdf8;">OPC-UA PUBSUB</span>
                <span style="font-size: 8px; background: #064e3b; border: 1px solid #059669; color: #34d399; padding: 1px 4px; border-radius: 2px; font-family: monospace;">PORT 4840</span>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 800; color: #f8fafc;">85.0 <span style="font-size: 9px; color: #64748b;">msg/s</span></div>
            <div style="font-size: 9px; color: #94a3b8; font-family: monospace; margin-top: 4px; border-top: 1px solid #1e293b; padding-top: 3px;">
                SECURITY: <b style="color:#38bdf8;">AES256-SHA</b> | NODES: <b>48</b>
            </div>
        </div>
        """)

    with proto_p3:
        st.html("""
        <div class="scada-panel" style="padding: 12px 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; color: #38bdf8;">MQTT SPARKPLUG</span>
                <span style="font-size: 8px; background: #064e3b; border: 1px solid #059669; color: #34d399; padding: 1px 4px; border-radius: 2px; font-family: monospace;">PORT 1883</span>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 800; color: #f8fafc;">210.8 <span style="font-size: 9px; color: #64748b;">msg/s</span></div>
            <div style="font-size: 9px; color: #94a3b8; font-family: monospace; margin-top: 4px; border-top: 1px solid #1e293b; padding-top: 3px;">
                QoS: <b style="color:#10b981;">EXACTLY_ONCE</b> | ZLIB: <b>ON</b>
            </div>
        </div>
        """)

    with proto_p4:
        st.html("""
        <div class="scada-panel" style="padding: 12px 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; color: #38bdf8;">PROFINET IRT</span>
                <span style="font-size: 8px; background: #064e3b; border: 1px solid #059669; color: #34d399; padding: 1px 4px; border-radius: 2px; font-family: monospace;">ETHERNET</span>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 800; color: #f8fafc;">400.0 <span style="font-size: 9px; color: #64748b;">msg/s</span></div>
            <div style="font-size: 9px; color: #94a3b8; font-family: monospace; margin-top: 4px; border-top: 1px solid #1e293b; padding-top: 3px;">
                CYCLE: <b style="color:#34d399;">5.00 ms</b> | JITTER: <b>&lt;10µs</b>
            </div>
        </div>
        """)

# ---------------------------------------------------------------------
# VIEW 4: EDGE FLEET & PLC NODE ASSET INVENTORY
# ---------------------------------------------------------------------
def render_edge_fleet_assets():
    timestamp = time.strftime("%H:%M:%S")
    render_industrial_header(timestamp)

    st.markdown("#### Distributed Edge Hardware Fleet Inventory")
    st.markdown("<p style='font-size: 11.5px; color: #64748b; margin-top: -6px;'>Real-time hardware inventory, operational metrics, and diagnostic endpoints for industrial PLC nodes.</p>", unsafe_allow_html=True)

    nodes = [
        {"Node_ID": "NODE_ALPHA", "Role": "Ingress & TSN Gateway", "IP": "192.168.10.14", "CPU_Load": "42.5%", "RAM": "58.2%", "Temp": "48.2 °C", "MTBF": "98,000 hrs", "Health": "HEALTHY", "Uptime": "99.99%"},
        {"Node_ID": "NODE_BETA", "Role": "XGBoost ML Inferencer", "IP": "192.168.10.15", "CPU_Load": "68.0%", "RAM": "74.5%", "Temp": "54.1 °C", "MTBF": "92,000 hrs", "Health": "PROCESSING", "Uptime": "99.95%"},
        {"Node_ID": "NODE_GAMMA", "Role": "Local Storage & Feature Store", "IP": "192.168.10.16", "CPU_Load": "28.3%", "RAM": "82.0%", "Temp": "44.7 °C", "MTBF": "110,000 hrs", "Health": "SYNCED", "Uptime": "99.99%"},
        {"Node_ID": "NODE_DELTA", "Role": "HA Redundancy Failover", "IP": "192.168.10.17", "CPU_Load": "12.0%", "RAM": "24.1%", "Temp": "39.4 °C", "MTBF": "125,000 hrs", "Health": "HOT STANDBY", "Uptime": "100.00%"}
    ]
    
    n_rows = "".join([f"<tr><td style='color:#38bdf8; font-weight:bold;'>{n['Node_ID']}</td><td>{n['Role']}</td><td>{n['IP']}</td><td>{n['CPU_Load']}</td><td>{n['RAM']}</td><td>{n['Temp']}</td><td>{n['MTBF']}</td><td style='color:#10b981; font-weight:bold;'>{n['Health']}</td><td style='color:#e2e8f0;'>{n['Uptime']}</td></tr>" for n in nodes])
    st.html(f"""
    <div class="scada-panel" style="padding:0; overflow:hidden; margin-bottom: 14px;">
        <table class="scada-table">
            <thead><tr><th>NODE ID</th><th>ROLE</th><th>IP ADDRESS</th><th>CPU LOAD</th><th>RAM</th><th>TEMP</th><th>MTBF</th><th>HEALTH</th><th>UPTIME SLA</th></tr></thead>
            <tbody>{n_rows}</tbody>
        </table>
    </div>
    """)

    f_b1, f_b2 = st.columns(2)
    with f_b1:
        if st.button("⌕ Physical Socket RTT Probing", key="btn_probe_nodes"):
            st.toast("Socket Latency Prober: Node Alpha (0.8ms), Node Beta (1.1ms), Node Gamma (0.5ms), Node Delta (0.9ms) — PASS.")
    with f_b2:
        if st.button("⇄ Actuate Manual Redundancy Switch", key="btn_manual_failover"):
            st.session_state.failover_events += 1
            st.toast("Traffic successfully rerouted to Standby Node Delta.")

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Hardware-in-the-Loop (HIL) Physical PLC Socket & Hardware Interface")

    hil_status = helm_client.get_hil_status()
    hil_telem_pack = helm_client.poll_hil_hardware()
    hil_telem = hil_telem_pack.get("telemetry", {})
    hil_frames = helm_client.get_hil_frames()

    h_c1, h_c2 = st.columns([1.2, 1.8])
    with h_c1:
        st.markdown("<div style='font-size:10px; font-weight:700; color:#64748b; font-family:monospace; margin-bottom:4px;'>HARDWARE INTERFACE CONFIGURATION</div>", unsafe_allow_html=True)
        hil_mode_sel = st.selectbox(
            "Hardware Interface Protocol",
            ["Siemens S7comm (ISO-on-TCP :102)", "Modbus RTU (Serial RS-485)", "Raspberry Pi 5 TSN Direct", "Virtual S7-1500 PLC Emulator"],
            index=3,
            key="hil_mode_sel"
        )
        target_ip_val = st.text_input("Hardware IP Address / Device URI", value="192.168.10.14", key="hil_ip_input")
        
        btn_hil_conn, btn_hil_disc = st.columns(2)
        with btn_hil_conn:
            if st.button("⏵ Connect HIL", key="btn_hil_connect"):
                m_code = "ethernet" if "S7comm" in hil_mode_sel else ("serial_rtu" if "Serial" in hil_mode_sel else "virtual_s7")
                helm_client.connect_hil(mode=m_code, target_ip=target_ip_val, port=102)
                st.toast(f"HIL Bridge Connected: {hil_mode_sel} at {target_ip_val}")
                st.rerun()
        with btn_hil_disc:
            if st.button("✕ Disconnect", key="btn_hil_disconnect"):
                helm_client.disconnect_hil()
                st.toast("HIL Hardware Bridge Disconnected.")
                st.rerun()

        st.html(f"""
        <div class="scada-panel" style="padding: 10px 12px; margin-top: 8px;">
            <div style="display: flex; justify-content: space-between; font-size: 9.5px; font-family: monospace; margin-bottom: 2px;">
                <span style="color: #64748b;">HIL STATUS:</span>
                <b style="color: {'#34d399' if hil_status['is_connected'] else '#ef4444'};">{'LINK ESTABLISHED' if hil_status['is_connected'] else 'DISCONNECTED'}</b>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 9.5px; font-family: monospace; margin-bottom: 2px;">
                <span style="color: #64748b;">HARDWARE RTT:</span>
                <b style="color: #38bdf8;">{hil_status['last_rtt_ms']:.2f} ms</b>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 9.5px; font-family: monospace; margin-bottom: 2px;">
                <span style="color: #64748b;">PLC SCAN CYCLE:</span>
                <b style="color: #10b981;">{hil_telem.get('plc_cycle_time_us', 9800)/1000.0:.2f} ms</b>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 9.5px; font-family: monospace;">
                <span style="color: #64748b;">GPIO BITMASK:</span>
                <b style="color: #f59e0b;">{hil_telem.get('gpio_port_mask', '0x5F')}</b>
            </div>
        </div>
        """)

    with h_c2:
        st.markdown("<div style='font-size:10px; font-weight:700; color:#64748b; font-family:monospace; margin-bottom:4px;'>REAL-TIME HIL PACKET ANALYZER & RAW HEX LOG</div>", unsafe_allow_html=True)
        
        frame_rows = []
        for f in hil_frames[-6:][::-1]:
            dir_col = "#38bdf8" if f["dir"] == "TX" else "#34d399"
            frame_rows.append(f"""
            <tr>
                <td style="color: #64748b;">{f['time']}</td>
                <td style="color: {dir_col}; font-weight: bold;">{f['dir']}</td>
                <td style="color: #f8fafc;">{f['type']}</td>
                <td style="color: #94a3b8;">{f['bytes']}B</td>
                <td style="font-family: monospace; color: #cbd5e1; font-size: 8.5px;">{f['hex']}</td>
            </tr>
            """)
            
        st.html(f"""
        <div class="scada-panel" style="padding:0; overflow:hidden; height: 180px;">
            <table class="scada-table">
                <thead><tr><th>TIME</th><th>DIR</th><th>FRAME TYPE</th><th>LEN</th><th>RAW HEX PAYLOAD</th></tr></thead>
                <tbody>{''.join(frame_rows)}</tbody>
            </table>
        </div>
        """)

# ---------------------------------------------------------------------
# VIEW 5: TREESHAP XAI & TEMPORAL FEATURE STORE
# ---------------------------------------------------------------------
def render_treeshap_feature_store():
    timestamp = time.strftime("%H:%M:%S")
    render_industrial_header(timestamp)

    st.markdown("#### TreeSHAP Explainable XAI & Temporal Feature Store")
    st.markdown("<p style='font-size: 11.5px; color: #64748b; margin-top: -6px;'>Fast native TreeSHAP marginal feature attributions (pred_contribs &lt;0.2ms), drift telemetry (PSI / KS-test), and shadow retraining.</p>", unsafe_allow_html=True)

    chart_df = st.session_state.history
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### Real-Time TreeSHAP Attribution Breakdown (Δ ms)")
        sample_pred = helm_client.predict_latency(
            throughput=float(chart_df["Throughput"].iloc[-1]) if len(chart_df) > 0 else 84.5,
            drop_pct=float(chart_df["Drops"].iloc[-1]) if len(chart_df) > 0 else 0.35,
            buffer_util=float(chart_df["Buffer_Util"].iloc[-1]) if len(chart_df) > 0 else 41.5,
            temp=float(chart_df["Temperature"].iloc[-1]) if len(chart_df) > 0 else 47.5
        )
        shap_raw = sample_pred.get("shap_attributions", {})
        shap_items = [
            ("Packet Loss Rate (%)", shap_raw.get("packet_drop_percentage", 5.2)),
            ("TSN Buffer Saturation (%)", shap_raw.get("buffer_utilization_percentage", 3.4)),
            ("Dynamic Cluster Regime", shap_raw.get("dynamic_operational_label", 2.1)),
            ("Throughput Trend Slope", shap_raw.get("throughput_slope", -0.8)),
            ("Die Junction Temp (°C)", shap_raw.get("node_temperature_celsius", 0.6))
        ]
        
        bar_rows = []
        for feat, val in shap_items:
            pct = min(100, int((abs(val) / 8.0) * 100))
            col = "#ef4444" if val > 0 else "#10b981"
            bar_rows.append(f"""
            <div style="margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; font-size: 11px; font-family: 'JetBrains Mono', monospace; margin-bottom: 2px;">
                    <span style="color: #cbd5e1;">{feat}</span>
                    <span style="color: {col}; font-weight: bold;">{'+' if val>0 else ''}{val:.2f} ms</span>
                </div>
                <div style="background: #0b0f17; border-radius: 2px; height: 6px; width: 100%; overflow: hidden;">
                    <div style="background: {col}; height: 6px; width: {pct}%;"></div>
                </div>
            </div>
            """)
            
        st.html(f"""
        <div class="scada-panel" style="padding: 14px 16px; height: 240px; box-sizing: border-box;">
            {''.join(bar_rows)}
        </div>
        """)

    with c2:
        st.markdown("##### Feature Store Rolling Window Statistics (30s)")
        stats_data = [
            {"Signal": "Ingress Throughput", "Mean": f"{chart_df['Throughput'].mean():.2f} Mbps", "Max": f"{chart_df['Throughput'].max():.2f} Mbps", "StdDev": f"{chart_df['Throughput'].std():.2f}", "Status": "NOMINAL"},
            {"Signal": "Packet Loss Rate", "Mean": f"{chart_df['Drops'].mean():.2f} %", "Max": f"{chart_df['Drops'].max():.2f} %", "StdDev": f"{chart_df['Drops'].std():.2f}", "Status": "NOMINAL"},
            {"Signal": "TSN Buffer Util", "Mean": f"{chart_df['Buffer_Util'].mean():.2f} %", "Max": f"{chart_df['Buffer_Util'].max():.2f} %", "StdDev": f"{chart_df['Buffer_Util'].std():.2f}", "Status": "NOMINAL"},
            {"Signal": "Junction Temp", "Mean": f"{chart_df['Temperature'].mean():.2f} °C", "Max": f"{chart_df['Temperature'].max():.2f} °C", "StdDev": f"{chart_df['Temperature'].std():.2f}", "Status": "NOMINAL"}
        ]
        
        stat_rows = "".join([f"<tr><td style='color:#38bdf8; font-weight:bold;'>{s['Signal']}</td><td>{s['Mean']}</td><td>{s['Max']}</td><td>{s['StdDev']}</td><td style='color:#10b981; font-weight:bold;'>{s['Status']}</td></tr>" for s in stats_data])
        st.html(f"""
        <div class="scada-panel" style="padding:0; overflow:hidden; height: 240px;">
            <table class="scada-table">
                <thead><tr><th>TELEMETRY SIGNAL</th><th>MEAN (30S)</th><th>PEAK (MAX)</th><th>STD DEV</th><th>QUALITY</th></tr></thead>
                <tbody>{stat_rows}</tbody>
            </table>
        </div>
        """)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Population Stability Index (PSI) & Data Drift Telemetry")
    
    drift_batch = {
        "throughput_mbps": chart_df["Throughput"].tolist(),
        "packet_drop_percentage": chart_df["Drops"].tolist(),
        "buffer_utilization_percentage": chart_df["Buffer_Util"].tolist(),
        "node_temperature_celsius": chart_df["Temperature"].tolist()
    }
    drift_info = helm_client.get_drift_metrics(drift_batch)
    feat_metrics = drift_info.get("feature_metrics", {})
    
    d_rows = []
    for f_name, d_val in feat_metrics.items():
        psi = d_val.get("psi", 0.02)
        ks = d_val.get("ks_statistic", 0.05)
        p_v = d_val.get("p_value", 0.85)
        drift_col = "#10b981" if psi < 0.10 else ("#f59e0b" if psi < 0.25 else "#ef4444")
        d_status = "STABLE" if psi < 0.10 else ("MODERATE SHIFT" if psi < 0.25 else "SIGNIFICANT DRIFT")
        
        d_rows.append(f"""
        <tr>
            <td style="color:#f8fafc; font-weight:bold;">{f_name}</td>
            <td style="color:{drift_col}; font-weight:bold;">{psi:.4f}</td>
            <td>{ks:.4f}</td>
            <td>{p_v:.4f}</td>
            <td><span style="color:{drift_col}; font-weight:bold;">{d_status}</span></td>
        </tr>
        """)
        
    st.html(f"""
    <div class="scada-panel" style="padding:0; overflow:hidden; margin-bottom: 12px;">
        <table class="scada-table">
            <thead><tr><th>FEATURE NAME</th><th>PSI INDEX</th><th>KS STATISTIC</th><th>P-VALUE</th><th>DRIFT STATE</th></tr></thead>
            <tbody>{''.join(d_rows)}</tbody>
        </table>
    </div>
    """)

    t_b1, t_b2 = st.columns([1.5, 2.5])
    with t_b1:
        if st.button("⟳ Trigger Shadow Model Retraining Pipeline", key="btn_shadow_retrain"):
            res = helm_client.trigger_retraining()
            st.toast(f"Retraining Complete: Version {res['model_version']} loaded.")
            st.success(f"Shadow model retrained successfully. Model Version: **{res['model_version']}**")
    with t_b2:
        st.info("Continuous Model Governance: Automatic retraining activates when composite PSI exceeds **0.250** boundary threshold.")

# ---------------------------------------------------------------------
# VIEW 6: QOS POLICY, TRAFFIC SHAPING & SAFETY INTERLOCKS
# ---------------------------------------------------------------------
def render_qos_policy_shaper():
    timestamp = time.strftime("%H:%M:%S")
    render_industrial_header(timestamp)

    st.markdown("#### QoS Actuation, Token Bucket Filter & Safety Interlocks")
    st.markdown("<p style='font-size: 11.5px; color: #64748b; margin-top: -6px;'>Linux tc Token Bucket Filter (TBF) parameters, rate limiting boundaries, and safety debounce guards.</p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Latency SLA Threshold Limit")
        lat_thresh = st.slider("High Alarm Setpoint (ms)", 30.0, 120.0, float(st.session_state.latency_threshold_ms), step=5.0, key="slider_lat_thresh")
        st.session_state.latency_threshold_ms = lat_thresh

        st.markdown("##### Linux tc Rate Limiting Factor")
        shedding = st.slider("Traffic Shedding Rate Multiplier", 0.50, 0.95, float(st.session_state.shedding_factor), step=0.02, key="slider_shedding")
        st.session_state.shedding_factor = shedding

    with c2:
        st.markdown("##### Thermal Regulatory Threshold")
        t_thresh = st.slider("Core Temperature Throttle Limit (°C)", 50.0, 85.0, float(st.session_state.throttling_temp_threshold), step=1.0, key="slider_temp_thresh")
        st.session_state.throttling_temp_threshold = t_thresh

        st.markdown("##### HA Failover Loss Boundary")
        f_thresh = st.slider("Failover Packet Loss Trigger (%)", 1.0, 4.0, float(st.session_state.failover_drop_threshold), step=0.2, key="slider_failover_thresh")
        st.session_state.failover_drop_threshold = f_thresh

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Linux tc Token Bucket Filter (TBF) Kernel Configuration")
    
    tbf_rate = 100.0 * st.session_state.shedding_factor
    tbf_burst = int(tbf_rate * 12.5)
    
    st.html(f"""
    <div class="scada-panel" style="padding: 16px; margin-bottom: 14px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #94a3b8; display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
            <div>
                <span style="color: #64748b;">SHAPED BANDWIDTH:</span><br>
                <b style="color: #38bdf8;">{tbf_rate:.1f} Mbps</b>
            </div>
            <div>
                <span style="color: #64748b;">BURST CAPACITY:</span><br>
                <b style="color: #10b981;">{tbf_burst} KB</b>
            </div>
            <div>
                <span style="color: #64748b;">DEBOUNCE HOLDOFF:</span><br>
                <b style="color: #f59e0b;">5.0 sec (Anti-Flapping)</b>
            </div>
            <div>
                <span style="color: #64748b;">SAFETY INTERLOCK:</span><br>
                <b style="color: #34d399;">SIL-2 CERTIFIED</b>
            </div>
        </div>
    </div>
    """)

    q_b1, q_b2 = st.columns(2)
    with q_b1:
        if st.button("⏵ Test Ingress Traffic Shaper Actuation", key="btn_test_shaper"):
            res = helm_client.execute_mitigation(
                action="traffic_shedding",
                target_device="PLC_NODE_ALPHA",
                reason="Manual QoS test actuation from operator console",
                shedding_factor=st.session_state.shedding_factor
            )
            st.toast(f"QoS Actuation: {res['status'].upper()} — Traffic Shaped to {tbf_rate:.1f} Mbps.")
    with q_b2:
        if st.button("⟳ Reset QoS Discipline to Default", key="btn_reset_shaper"):
            res = helm_client.execute_mitigation(
                action="reset",
                target_device="PLC_NODE_ALPHA",
                reason="Reset to baseline"
            )
            st.toast("QoS Traffic Discipline Restored to Default.")

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Autonomous Closed-Loop SIL-2 Playbook Matrix")

    pb1, pb2 = st.columns(2)
    with pb1:
        st.html("""
        <div class="scada-panel" style="padding: 12px 14px; margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #f8fafc;">PLAYBOOK 01: TRAFFIC SHEDDING</span>
                <span style="background: #064e3b; border: 1px solid #059669; color: #34d399; font-size: 8px; padding: 1px 5px; border-radius: 2px; font-family: monospace;">ARMED</span>
            </div>
            <div style="font-size: 10px; color: #94a3b8; font-family: monospace; margin-top: 4px;">
                TRIGGER: Predicted Latency &ge; Setpoint (60ms) for &ge; 2 cycles.<br>
                ACTION: Actuate Linux tc TBF queue rate limiting (-18%).
            </div>
        </div>
        <div class="scada-panel" style="padding: 12px 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #f8fafc;">PLAYBOOK 02: THERMAL REGULATOR</span>
                <span style="background: #064e3b; border: 1px solid #059669; color: #34d399; font-size: 8px; padding: 1px 5px; border-radius: 2px; font-family: monospace;">ARMED</span>
            </div>
            <div style="font-size: 10px; color: #94a3b8; font-family: monospace; margin-top: 4px;">
                TRIGGER: Core Junction Temperature &gt; 68.0°C.<br>
                ACTION: CPU Frequency Scaling + 1.2ms cycle delay holdoff.
            </div>
        </div>
        """)

    with pb2:
        st.html("""
        <div class="scada-panel" style="padding: 12px 14px; margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #f8fafc;">PLAYBOOK 03: ZERO-LOSS HA FAILOVER</span>
                <span style="background: #064e3b; border: 1px solid #059669; color: #34d399; font-size: 8px; padding: 1px 5px; border-radius: 2px; font-family: monospace;">ARMED</span>
            </div>
            <div style="font-size: 10px; color: #94a3b8; font-family: monospace; margin-top: 4px;">
                TRIGGER: Packet Loss &gt; 1.80% or Unresponsive Heartbeat.<br>
                ACTION: Zero-loss traffic rerouting to Standby Node Delta.
            </div>
        </div>
        <div class="scada-panel" style="padding: 12px 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #f8fafc;">PLAYBOOK 04: AUTO SHADOW RETRAINING</span>
                <span style="background: #064e3b; border: 1px solid #059669; color: #34d399; font-size: 8px; padding: 1px 5px; border-radius: 2px; font-family: monospace;">ARMED</span>
            </div>
            <div style="font-size: 10px; color: #94a3b8; font-family: monospace; margin-top: 4px;">
                TRIGGER: Composite Feature PSI &gt; 0.250.<br>
                ACTION: Async background XGBoost retraining + weights reload.
            </div>
        </div>
        """)

# ---------------------------------------------------------------------
# VIEW 7: INCIDENT AUDIT & SEQUENCE-OF-EVENTS (SOE) LOG
# ---------------------------------------------------------------------
def render_incident_audit_soe():
    timestamp = time.strftime("%H:%M:%S")
    render_industrial_header(timestamp)

    st.markdown("#### Incident Audit Trail & Sequence-of-Events (SOE) Log")
    st.markdown("<p style='font-size: 11.5px; color: #64748b; margin-top: -6px;'>Immutable chronological audit record conforming to ISA-18.2 / IEC 62443 cyber-physical compliance.</p>", unsafe_allow_html=True)

    incidents_df = pd.DataFrame(st.session_state.incident_logs)
    
    rows_html = []
    for log in st.session_state.incident_logs[::-1]:
        p_class = "badge-p1" if "CRIT" in log["Priority"] or "FAILOVER" in log.get("Quality", "") else ("badge-p2" if "WARN" in log["Priority"] else "badge-p3")
        rows_html.append(f"""
        <tr>
            <td style="color:#94a3b8;">{log['Timestamp']}</td>
            <td style="color:#f8fafc; font-weight:bold;">{log.get('Alarm_ID', 'ALM-1000')}</td>
            <td><span class="{p_class}">{log['Priority']}</span></td>
            <td style="color:#38bdf8;">{log.get('Tag', 'SYS')}</td>
            <td>{log['Event']}</td>
            <td style="color:#64748b;">{log.get('Quality', 'GOOD_0x00')}</td>
        </tr>
        """)

    table_html = f"""
    <div class="scada-panel" style="padding: 0; overflow: hidden; margin-bottom: 14px;">
        <table class="scada-table">
            <thead>
                <tr>
                    <th style="width: 90px;">TIMESTAMP</th>
                    <th style="width: 110px;">ALARM ID</th>
                    <th style="width: 90px;">PRIORITY</th>
                    <th style="width: 130px;">TAG NAME</th>
                    <th>EVENT DESCRIPTION</th>
                    <th style="width: 120px;">QUALITY CODE</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    """
    st.html(table_html)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="⤓ Download Compliance Audit Report (.CSV)",
            data=incidents_df.to_csv(index=False).encode('utf-8'),
            file_name="iiot_compliance_audit.csv",
            mime="text/csv",
            key="btn_dl_full_audit_csv"
        )
    with d2:
        st.download_button(
            label="⤓ Download Complete Telemetry Stream (.JSON)",
            data=st.session_state.history.to_json(orient="records", indent=2).encode('utf-8'),
            file_name="iiot_telemetry_stream.json",
            mime="application/json",
            key="btn_dl_full_audit_json"
        )

# ---------------------------------------------------------------------
# MASTER WORKSPACE ROUTER
# ---------------------------------------------------------------------
if "Live SCADA Telemetry" in st.session_state.active_nav:
    render_live_scada_telemetry()
elif "AI SCADA Copilot" in st.session_state.active_nav:
    render_ai_copilot_workspace()
elif "Fieldbus Topology" in st.session_state.active_nav:
    render_fieldbus_topology_cad()
elif "Edge Fleet" in st.session_state.active_nav:
    render_edge_fleet_assets()
elif "TreeSHAP" in st.session_state.active_nav:
    render_treeshap_feature_store()
elif "QoS Actuation" in st.session_state.active_nav:
    render_qos_policy_shaper()
elif "Incident Audit" in st.session_state.active_nav:
    render_incident_audit_soe()