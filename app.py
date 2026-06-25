import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="NEET-Sighted Indonesia 2024",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── LOAD DATA ────────────────────────────────────────────────
@st.cache_data
def load_data():
    master = pd.read_csv("data/master_data_neet_clean.csv", encoding="utf-8-sig")
    cluster = pd.read_csv("data/hasil_clustering_neet.csv", encoding="utf-8-sig")
    with open("data/indonesia_38_provinsi_fixed.geojson", encoding="utf-8") as f:
        geojson = json.load(f)

    # Rename columns to be consistent
    master.columns = master.columns.str.strip()
    master["kode_provinsi"] = master["kode_provinsi"].astype(str)
    cluster["kode_provinsi"] = cluster["kode_provinsi"].astype(str)

    # Remove duplicate kode if any
    master = master.drop_duplicates(subset="kode_provinsi", keep="first")

    # Merge
    df = master.merge(cluster[["kode_provinsi", "cluster", "warna_klaster"]], on="kode_provinsi", how="left")
    df["cluster"] = df["cluster"].fillna(1).astype(int)

    # Clean numeric cols
    num_cols = ["Target_NEET", "TPT", "Miskin", "APK_PT", "IPM", "PDRB",
                "SMK_Komp", "SMK_Int", "Kerjasama_Ind", "Kap_LPK", "Peserta_PBK", "Total_Magang"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Island grouping
    pulau_map = {
        "ACEH": "Sumatera", "SUMATERA UTARA": "Sumatera", "SUMATERA BARAT": "Sumatera",
        "RIAU": "Sumatera", "JAMBI": "Sumatera", "SUMATERA SELATAN": "Sumatera",
        "BENGKULU": "Sumatera", "LAMPUNG": "Sumatera", "KEP. BANGKA BELITUNG": "Sumatera",
        "KEP. RIAU": "Sumatera",
        "DKI JAKARTA": "Jawa", "JAWA BARAT": "Jawa", "JAWA TENGAH": "Jawa",
        "DI YOGYAKARTA": "Jawa", "JAWA TIMUR": "Jawa", "BANTEN": "Jawa",
        "BALI": "Bali-Nusa", "NUSA TENGGARA BARAT": "Bali-Nusa", "NUSA TENGGARA TIMUR": "Bali-Nusa",
        "KALIMANTAN BARAT": "Kalimantan", "KALIMANTAN TENGAH": "Kalimantan",
        "KALIMANTAN SELATAN": "Kalimantan", "KALIMANTAN TIMUR": "Kalimantan",
        "KALIMANTAN UTARA": "Kalimantan",
        "SULAWESI UTARA": "Sulawesi", "SULAWESI TENGAH": "Sulawesi",
        "SULAWESI SELATAN": "Sulawesi", "SULAWESI TENGGARA": "Sulawesi",
        "GORONTALO": "Sulawesi", "SULAWESI BARAT": "Sulawesi",
        "MALUKU": "Maluku", "MALUKU UTARA": "Maluku",
        "PAPUA BARAT": "Papua", "PAPUA BARAT DAYA": "Papua", "PAPUA": "Papua",
        "PAPUA SELATAN": "Papua", "PAPUA TENGAH": "Papua", "PAPUA PEGUNUNGAN": "Papua",
    }
    df["Provinsi"] = df["Provinsi"].str.title().replace({"Di Yogyakarta": "DI Yogyakarta", "Dki Jakarta": "DKI Jakarta"})
    df["pulau"] = df["Provinsi"].map(pulau_map).fillna("Lainnya")

    # Add regression-based prediction
    def predict_neet(row):
        tpt = max(0.01, row["TPT"])
        ipm = row["IPM"]
        smk = row["SMK_Komp"]
        log_n = 5.218 + 0.659 * np.log(tpt) - 0.037 * ipm - 0.005 * smk
        return round(np.exp(log_n), 2)
    df["pred_neet"] = df.apply(predict_neet, axis=1)

    return df, geojson


df, geojson = load_data()

def hex_to_rgba(hex_str, alpha):
    hex_str = hex_str.lstrip('#')
    r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"

# Load logo assets as Base64 for custom HTML rendering
import base64
import os

def get_base64_image(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return ""

unair_base64 = get_base64_image("assets/unair.png")
ftmm_base64 = get_base64_image("assets/ftmm.png")
pemuda_neet_base64 = get_base64_image("assets/pemuda-neet.png")

# ── COLORS & CONSTANTS ───────────────────────────────────────
CLUSTER_COLORS = {1: "#059669", 2: "#EAB308", 3: "#DC2626"}
CLUSTER_LABELS = {1: "Provinsi Berkinerja Baik", 2: "Provinsi Transisi", 3: "Provinsi Prioritas Intervensi"}
CLUSTER_SHORT = {1: "Provinsi Berkinerja Baik", 2: "Provinsi Transisi", 3: "Provinsi Prioritas Intervensi"}
NAVY = "#1E3A8A"

CORR_MATRIX = {
    "vars": ["NEET", "TPT", "IPM", "Miskin", "SMK_Komp"],
    "matrix": [
        [1.00,  0.55, -0.72,  0.61, -0.44],
        [0.55,  1.00, -0.48,  0.53, -0.31],
        [-0.72, -0.48, 1.00, -0.69,  0.58],
        [0.61,  0.53, -0.69,  1.00, -0.39],
        [-0.44, -0.31, 0.58, -0.39,  1.00],
    ]
}

PCA_DATA = {
    1: [(-2.1,0.4),(-1.8,1.1),(-1.5,-0.3),(-2.3,0.8),(-1.9,1.5),(-1.6,-0.7),(-2.4,0.2),(-1.7,0.9),(-2.0,1.3),(-1.4,-0.5),(-2.2,0.6),(-1.6,1.1),(-1.9,-0.2),(-1.3,0.7),(-2.1,1.0),(-1.8,0.3)],
    2: [(0.3,0.8),(0.6,1.2),(-0.1,0.5),(0.4,-0.3),(0.8,0.9),(-0.2,1.1),(0.5,-0.6),(0.2,1.4),(-0.4,0.3),(0.7,-0.1),(0.1,0.7),(0.4,1.3),(-0.3,-0.4),(0.6,0.5),(0.0,1.0)],
    3: [(2.8,0.3),(2.4,1.1),(2.6,-0.5),(3.1,0.8),(2.9,1.4),(2.5,-0.2),(3.2,0.5)],
}

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

/* Main background */
.main .block-container {
    padding: 0rem 2rem 3rem 2rem !important;
    max-width: 100%;
    background: #F8FAFC;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #1e3a8a 0%, #6db5d8 100%) !important;
}
[data-testid="stSidebar"] > div {
    background: linear-gradient(135deg, #1e3a8a 0%, #6db5d8 100%) !important;
}

/* Sidebar Custom Logo and Description Boxes */
[data-testid="stSidebar"] .sidebar-logo-box {
    margin-top: 40px !important;
    padding: 12px !important;
    background: rgba(255, 255, 255, 0.15) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 16px !important;
    margin-bottom: 12px !important;
}
[data-testid="stSidebar"] .sidebar-desc-box {
    background: rgba(255, 255, 255, 0.15) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
    font-size: 13.5px !important;
    line-height: 1.6 !important;
    text-align: left !important;
}
[data-testid="stSidebar"] .sidebar-desc-box,
[data-testid="stSidebar"] .sidebar-desc-box div,
[data-testid="stSidebar"] .sidebar-desc-box p,
[data-testid="stSidebar"] .sidebar-desc-box strong {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .sidebar-desc-box span.label {
    color: rgba(255, 255, 255, 0.8) !important;
    font-weight: 500 !important;
}
[data-testid="stSidebarUserContent"] {
    padding-left: 14px !important;
    padding-right: 14px !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown span {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] * { color: #FFFFFF; }

/* Sidebar radio buttons */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > div {
    width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    color: #FFFFFF !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    transition: all 0.18s !important;
    cursor: pointer !important;
    white-space: normal !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.1) !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-checked="true"] {
    background: rgba(255, 255, 255, 0.15) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-checked="true"] p {
    font-weight: 700 !important;
}

/* Hover class for cards */
.hover-card {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.hover-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1) !important;
}

/* KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.kpi-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.05);
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-blue::after  { background: #1E3A8A; }
.kpi-red::after   { background: #DC2626; }
.kpi-green::after { background: #059669; }
.kpi-yellow::after{ background: #EAB308; }
.kpi-label { font-size: 13px; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; }
.kpi-value { font-size: 32px; font-weight: 800; color: #1E3A8A; letter-spacing: -0.03em; margin: 4px 0; line-height: 1; }
.kpi-sub   { font-size: 15.5px; color: #64748B; }
.kpi-icon  { position: absolute; right: 16px; top: 16px; font-size: 24px; opacity: 0.15; }

/* Cards */
div[data-testid="stVerticalBlockBorder"] {
    background: white !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.05) !important;
    padding: 20px !important;
    margin-bottom: 20px !important;
}
.card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.05);
    padding: 20px;
    margin-bottom: 20px;
}
.card-title {
    font-size: 16px;
    font-weight: 800;
    color: #1E3A8A;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.card-subtitle {
    font-size: 15.5px;
    color: #64748B;
    margin-bottom: 12px;
}

/* Insight box */
.insight-box {
    background: #EEF2FF;
    border-left: 3px solid #1E3A8A;
    border-radius: 0 8px 8px 0;
    padding: 12px 14px;
    margin-top: 14px;
    font-size: 16px;
    color: #152C6B;
    line-height: 1.55;
}
.insight-title {
    font-weight: 700;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 4px;
    color: #1E3A8A;
}

/* Section title */
.section-title {
    font-size: 11px;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 4px 10px;
    background: #F1F5F9;
    border-radius: 4px;
    margin: 24px 0 14px 0;
    display: inline-block;
}

/* Page header */
.page-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #6db5d8 100%) !important;
    border-radius: 12px;
    padding: 24px 32px;
    margin-top: 10px;
    margin-bottom: 24px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08), 0 2px 4px -1px rgba(0,0,0,0.04);
}
.page-title {
    font-family: 'Inter', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: #FFFFFF !important;
    letter-spacing: 0.03em;
    margin: 6px 0 4px 0;
}
.page-desc {
    color: rgba(255, 255, 255, 0.75) !important;
    font-size: 17px;
    margin: 0;
    padding: 0;
}
.badge-year {
    background: #EEF2FF;
    color: #1E3A8A;
    font-size: 13px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: 0.05em;
    display: inline-block;
}

/* Cluster cards */
.cluster-card {
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 4px;
}
.cluster-card.green  { background: #ECFDF5; border: 1px solid #A7F3D0; }
.cluster-card.yellow { background: #FFFBEB; border: 1px solid #FDE68A; }
.cluster-card.red    { background: #FEF2F2; border: 1px solid #FECACA; }
.cc-title { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 6px; }
.cc-stat  { font-size: 26px; font-weight: 800; letter-spacing: -0.02em; }
.cc-sub   { font-size: 13px; color: #64748B; margin-bottom: 10px; }
.cc-item  { font-size: 16px; margin-bottom: 4px; }

/* Rec card */
.rec-card {
    border-radius: 12px;
    padding: 18px 20px;
    margin-top: 12px;
}
.rec-card.green  { background: #ECFDF5; border: 1px solid #A7F3D0; }
.rec-card.yellow { background: #FFFBEB; border: 1px solid #FDE68A; }
.rec-card.red    { background: #FEF2F2; border: 1px solid #FECACA; }

/* Sim cards */
.sim-result {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.sim-label { font-size: 13px; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; }
.sim-value { font-size: 30px; font-weight: 800; color: #1E3A8A; margin: 4px 0; }
.sim-sub   { font-size: 13px; color: #64748B; }

/* Finding cards */
.finding-card {
    background: white;
    min-height: 145px;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.finding-num {
    font-size: 12px;
    font-weight: 700;
    color: #1E3A8A;
    background: #EEF2FF;
    padding: 2px 7px;
    border-radius: 10px;
    letter-spacing: 0.06em;
}
.finding-text { font-size: 14.5px; color: #0F172A; margin-top: 6px; line-height: 1.5; }

/* Regression table */
.reg-table { width: 100%; border-collapse: collapse; font-size: 16px; }
.reg-table th { background: #F8FAFC; padding: 8px 12px; text-align: left; font-weight: 600; color: #64748B; font-size: 13px; border-bottom: 1px solid #E2E8F0; }
.reg-table td { padding: 8px 12px; border-bottom: 1px solid #F1F5F9; }
.reg-table tr:hover td { background: #F8FAFC; }
.sig-star { color: #DC2626; font-weight: 700; }

/* Province table */
.prov-table { width: 100%; border-collapse: collapse; font-size: 16px; }
.prov-table thead tr { background: linear-gradient(90deg, #1e3a8a 0%, #6db5d8 100%); }
.prov-table th { background: transparent; padding: 8px 10px; text-align: left; font-weight: 600; color: #FFFFFF; font-size: 13px; border-bottom: 2px solid #E2E8F0; }
.prov-table td { padding: 7px 10px; border-bottom: 1px solid #F1F5F9; }
.prov-table tr:hover td { background: #F8FAFC; }
.cluster-dot { display: inline-flex; align-items: center; gap: 5px; font-size: 13px; font-weight: 600; }

/* Streamlit overrides */

div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background-color: #FFFFFF !important;
}

.stSelectbox, .stSlider { font-size: 17px; }
div[data-baseweb="select"] { border-radius: 8px !important; }
.stSlider [data-baseweb="slider"] { margin-top: 0; }

/* Hide default streamlit elements */
[data-testid="stHeader"] {background-color: transparent !important;}
#MainMenu {visibility: hidden !important;}
footer {display: none !important;}
[data-testid="stToolbarActionButton"] {display: none !important;}

/* Fix for sidebar toggle button visibility */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    opacity: 1 !important;
    visibility: visible !important;
    z-index: 999999 !important;
    color: #1e3a8a !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg {
    fill: #1e3a8a !important;
    color: #1e3a8a !important;
}

/* Specific Sidebar Overrides for Logo and Description Box */
[data-testid="stSidebar"] .stMarkdown .sidebar-desc-box,
[data-testid="stSidebar"] .stMarkdown .sidebar-desc-box *,
[data-testid="stSidebar"] .stMarkdown .sidebar-desc-box div,
[data-testid="stSidebar"] .stMarkdown .sidebar-desc-box p,
[data-testid="stSidebar"] .stMarkdown .sidebar-desc-box span,
[data-testid="stSidebar"] .stMarkdown .sidebar-desc-box strong,
[data-testid="stSidebar"] .stMarkdown .sidebar-desc-box span.label {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stMarkdown .sidebar-desc-box span.label {
    color: rgba(255, 255, 255, 0.8) !important;
}
[data-testid="stSidebar"] .stMarkdown .sidebar-desc-box .sidebar-desc-title {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    line-height: 1.4 !important;
    margin-bottom: 10px !important;
    letter-spacing: 0.02em !important;
}
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding: 12px 16px 8px; margin-bottom: 8px;">
    <div style="color: #FFFFFF !important; font-size: 10px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; padding-left: 8px;">Modul Analisis</div>
</div>
""", unsafe_allow_html=True)

    page = st.radio(
        "",
        options=[
            "Beranda",
            "Overview NEET",
            "Faktor Penentu NEET",
            "Clustering Risiko NEET",
            "Simulasi & Rekomendasi"
        ],
        label_visibility="collapsed"
    )

    st.markdown(f"""
<div class="sidebar-logo-box">
    <img src="data:image/png;base64,{unair_base64}" style="height: 38px; width: auto; object-fit: contain;" />
    <img src="data:image/png;base64,{ftmm_base64}" style="height: 28px; width: auto; object-fit: contain;" />
</div>
<div class="sidebar-desc-box">
    <div class="sidebar-desc-title">NEET-Sighted Indonesia: Dashboard Interaktif Analisis Spasial dan Simulator Kebijakan Pemuda NEET</div>
    <div style="margin-bottom: 4px;"><span class="label">Mata Kuliah:</span> <strong>Official Statistics</strong></div>
    <div style="margin-bottom: 4px;"><span class="label">Prodi:</span> <strong>Teknologi Sains Data</strong></div>
    <div style="margin-bottom: 8px;"><span class="label">Fakultas:</span> <strong>FTMM - Universitas Airlangga</strong></div>
    <div style="border-top: 1px dashed rgba(255, 255, 255, 0.3); padding-top: 8px;"><span class="label">Tim:</span> <strong>Kelompok F Official Statistics</strong></div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 0 — BERANDA
# ════════════════════════════════════════════════════════════
if page == "Beranda":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #6db5d8 100%); border-radius: 16px; padding: 48px 40px; color: white; margin-bottom: 40px; text-align: center; box-shadow: 0 10px 25px rgba(30, 58, 138, 0.2);">
        <h1 style="color: #ffffff; margin-bottom: 12px; font-size: 42px; font-weight: 800; font-family: 'Outfit', sans-serif; letter-spacing: 0.03em;">NEET-Sighted Indonesia</h1>
        <div style="font-size: 22px; font-weight: 400; opacity: 0.95; margin-bottom: 28px; max-width: 800px; margin-left: auto; margin-right: auto; line-height: 1.4;">Dashboard Interaktif Analisis Spasial dan Simulator Kebijakan Pemuda NEET</div>
        <div style="display: flex; gap: 12px; flex-wrap: wrap; justify-content: center;">
            <span style="background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.3); backdrop-filter: blur(4px); border-radius: 20px; padding: 6px 16px; font-size: 13px; font-weight: 600;">📍 38 Provinsi · 2015–2025</span>
            <span style="background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.3); backdrop-filter: blur(4px); border-radius: 20px; padding: 6px 16px; font-size: 13px; font-weight: 600;">📈 Regresi OLS</span>
            <span style="background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.3); backdrop-filter: blur(4px); border-radius: 20px; padding: 6px 16px; font-size: 13px; font-weight: 600;">🧩 K-Means Clustering</span>
            <span style="background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.3); backdrop-filter: blur(4px); border-radius: 20px; padding: 6px 16px; font-size: 13px; font-weight: 600;">📊 Data BPS x Kemendikdasmen</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="font-size: 12px; font-weight: 700; color: #F59E0B; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">PENGERTIAN</div>
    <div style="font-size: 28px; font-weight: 800; color: #0F172A; margin-bottom: 24px; margin-top: 0; font-family: 'Outfit', sans-serif; letter-spacing: 0.03em;">Apa itu Pemuda NEET?</div>
    
    <div style="background: white; border-radius: 16px; padding: 32px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05); border: 1px solid #F1F5F9; margin-bottom: 48px;">
        <div style="display: flex; gap: 24px; align-items: stretch;">
            <div style="flex: 1;">
                <div style="font-size: 17px; color: #475569; line-height: 1.7; margin-bottom: 24px;">
                    <b style="color:#0F172A;">NEET (Not in Employment, Education, or Training)</b> adalah kelompok <b>pemuda usia produktif (15–24 tahun)</b> yang <b>tidak sedang bekerja</b>, <b>tidak menempuh pendidikan formal</b>, dan <b>tidak mengikuti pelatihan keterampilan</b> pada periode tertentu.
                </div>
                <div style="display: flex; gap: 16px; flex-wrap: wrap;">
                    <div class="hover-card" style="background: #FEE2E2; border-left: 4px solid #EF4444; padding: 16px; border-radius: 8px; display: flex; align-items: center; gap: 12px; white-space: nowrap;">
                        <div style="font-size: 24px;">🎓</div>
                        <div style="font-weight: 700; color: #991B1B; font-size: 16px;">Tidak Sekolah</div>
                    </div>
                    <div class="hover-card" style="background: #FEF3C7; border-left: 4px solid #F59E0B; padding: 16px; border-radius: 8px; display: flex; align-items: center; gap: 12px; white-space: nowrap;">
                        <div style="font-size: 24px;">💼</div>
                        <div style="font-weight: 700; color: #B45309; font-size: 16px;">Tidak Bekerja</div>
                    </div>
                    <div class="hover-card" style="background: #E0E7FF; border-left: 4px solid #6366F1; padding: 16px; border-radius: 8px; display: flex; align-items: center; gap: 12px; white-space: nowrap;">
                        <div style="font-size: 24px;">🛠️</div>
                        <div style="font-weight: 700; color: #3730A3; font-size: 16px;">Tidak Mengikuti Pelatihan</div>
                    </div>
                </div>
            </div>
            <img src="data:image/png;base64,{pemuda_neet_base64}" style="width: 320px; border-radius: 16px; object-fit: cover; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 12px; font-weight: 700; color: #F59E0B; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">LATAR BELAKANG</div>
    <div style="font-size: 28px; font-weight: 800; color: #0F172A; margin-bottom: 24px; margin-top: 0; font-family: 'Outfit', sans-serif; letter-spacing: 0.03em;">Mengapa dashboard ini penting?</div>
    
    <div class="hover-card" style="background: white; border-radius: 16px; padding: 32px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05); border: 1px solid #F1F5F9; display: flex; gap: 24px; align-items: center; margin-bottom: 48px;">
        <div style="font-size: 48px;">💡</div>
        <div style="font-size: 17px; color: #475569; line-height: 1.7;">
            NEET merupakan indikator penting untuk menggambarkan <b>kesiapan dan keterlibatan pemuda</b> dalam pendidikan maupun dunia kerja. Dashboard ini membantu <b>mengeksplorasi kondisi NEET</b> di Indonesia, memahami <b>faktor-faktor yang memengaruhinya</b>, dan mengidentifikasi kelompok provinsi yang memerlukan <b>prioritas kebijakan</b> untuk meningkatkan kualitas <b>sumber daya manusia muda</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 12px; font-weight: 700; color: #F59E0B; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">FITUR UTAMA</div>
    <div style="font-size: 28px; font-weight: 800; color: #0F172A; margin-bottom: 24px; margin-top: 0; font-family: 'Outfit', sans-serif; letter-spacing: 0.03em;">Apa yang bisa kamu eksplorasi?</div>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 24px; margin-bottom: 48px;">
        <div onclick="const labels = document.querySelectorAll('[data-testid=\'stSidebar\'] div[role=\'radiogroup\'] label'); if(labels[1]) labels[1].click();" class="hover-card" style="cursor: pointer; background: white; border-radius: 16px; padding: 28px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05); border: 1px solid #F1F5F9; display: flex; flex-direction: column; gap: 16px;">
            <div style="font-size: 32px; background: #F8FAFC; width: 64px; height: 64px; display: flex; align-items: center; justify-content: center; border-radius: 16px;">📂</div>
            <div>
                <div style="font-weight: 800; color: #0F172A; font-size: 17px; margin-bottom: 8px; font-family: 'Outfit', sans-serif;">Overview & Distribusi</div>
                <div style="font-size: 16px; color: #64748B; line-height: 1.6;">Eksplorasi <b>sebaran angka NEET di 38 provinsi</b> serta <b>tren historis</b> secara nasional.</div>
            </div>
        </div>
        <div onclick="const labels = document.querySelectorAll('[data-testid=\'stSidebar\'] div[role=\'radiogroup\'] label'); if(labels[2]) labels[2].click();" class="hover-card" style="cursor: pointer; background: white; border-radius: 16px; padding: 28px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05); border: 1px solid #F1F5F9; display: flex; flex-direction: column; gap: 16px;">
            <div style="font-size: 32px; background: #F8FAFC; width: 64px; height: 64px; display: flex; align-items: center; justify-content: center; border-radius: 16px;">📈</div>
            <div>
                <div style="font-weight: 800; color: #0F172A; font-size: 17px; margin-bottom: 8px; font-family: 'Outfit', sans-serif;">Regresi & Korelasi</div>
                <div style="font-size: 16px; color: #64748B; line-height: 1.6;">Analisis <b>pengaruh indikator makroekonomi dan pendidikan</b> terhadap angka NEET.</div>
            </div>
        </div>
        <div onclick="const labels = document.querySelectorAll('[data-testid=\'stSidebar\'] div[role=\'radiogroup\'] label'); if(labels[3]) labels[3].click();" class="hover-card" style="cursor: pointer; background: white; border-radius: 16px; padding: 28px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05); border: 1px solid #F1F5F9; display: flex; flex-direction: column; gap: 16px;">
            <div style="font-size: 32px; background: #F8FAFC; width: 64px; height: 64px; display: flex; align-items: center; justify-content: center; border-radius: 16px;">📊</div>
            <div>
                <div style="font-weight: 800; color: #0F172A; font-size: 17px; margin-bottom: 8px; font-family: 'Outfit', sans-serif;">Clustering K-Means</div>
                <div style="font-size: 16px; color: #64748B; line-height: 1.6;"><b>Pengelompokan wilayah</b> berdasarkan <b>tingkat kerentanan</b> terhadap NEET.</div>
            </div>
        </div>
        <div onclick="const labels = document.querySelectorAll('[data-testid=\'stSidebar\'] div[role=\'radiogroup\'] label'); if(labels[4]) labels[4].click();" class="hover-card" style="cursor: pointer; background: white; border-radius: 16px; padding: 28px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05); border: 1px solid #F1F5F9; display: flex; flex-direction: column; gap: 16px;">
            <div style="font-size: 32px; background: #F8FAFC; width: 64px; height: 64px; display: flex; align-items: center; justify-content: center; border-radius: 16px;">🎯</div>
            <div>
                <div style="font-weight: 800; color: #0F172A; font-size: 17px; margin-bottom: 8px; font-family: 'Outfit', sans-serif;">Simulasi & Rekomendasi</div>
                <div style="font-size: 16px; color: #64748B; line-height: 1.6;"><b>Prediksi dampak kebijakan</b> dan <b>rekomendasi</b> per provinsi berbasis regresi.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 12px; font-weight: 700; color: #F59E0B; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">INFORMASI TEKNIS</div>
    <div style="font-size: 28px; font-weight: 800; color: #0F172A; margin-bottom: 24px; margin-top: 0; font-family: 'Outfit', sans-serif; letter-spacing: 0.03em;">Sumber data & variabel yang digunakan</div>
    
    <div style="background: white; border-radius: 16px; padding: 24px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05); border: 1px solid #F1F5F9; margin-bottom: 48px; overflow-x: auto;">
        <table class="prov-table" style="margin-bottom: 0; font-size: 15.5px; width: 100%; border-radius: 8px; overflow: hidden;">
            <thead style="background: linear-gradient(135deg, #1e3a8a 0%, #6db5d8 100%);">
                <tr>
                    <th style="padding: 16px; border: none; color: white; font-weight: 700; background: transparent; font-size: 17px;">Variabel</th>
                    <th style="padding: 16px; border: none; color: white; font-weight: 700; background: transparent; font-size: 17px;">Tahun</th>
                    <th style="padding: 16px; border: none; color: white; font-weight: 700; background: transparent; font-size: 17px;">Satuan</th>
                    <th style="padding: 16px; border: none; color: white; font-weight: 700; background: transparent; font-size: 17px;">Peran dalam Analisis</th>
                    <th style="padding: 16px; border: none; color: white; font-weight: 700; background: transparent; font-size: 17px;">Sumber</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;"><b>Angka Pemuda NEET</b></td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">2015–2025</td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">%</td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">Variabel respon & fitur clustering</td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">BPS</td>
                </tr>
                <tr>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;"><b>Tingkat Pengangguran Terbuka (TPT)</b></td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">2024</td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">%</td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">Variabel prediktor & fitur clustering</td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">BPS</td>
                </tr>
                <tr>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;"><b>Indeks Pembangunan Manusia (IPM)</b></td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">2024</td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">Indeks 1–100</td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">Variabel prediktor & fitur clustering</td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">BPS</td>
                </tr>
                <tr>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;"><b>Persentase Penduduk Miskin</b></td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">2024</td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">%</td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">Variabel prediktor & fitur clustering</td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">BPS</td>
                </tr>
                <tr>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;"><b>SMK dengan Fasilitas Komputer</b></td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">2024</td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">%</td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">Variabel prediktor & fitur clustering</td>
                    <td style="padding: 12px 16px; border-bottom: 1px solid #F1F5F9;">Kemendikdasmen</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 12px; font-weight: 700; color: #F59E0B; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">TARGET AUDIENS</div>
    <div style="font-size: 28px; font-weight: 800; color: #0F172A; margin-bottom: 24px; margin-top: 0; font-family: 'Outfit', sans-serif; letter-spacing: 0.03em;">Dashboard ini untuk siapa?</div>
    
    <div style="font-size: 17px; color: #475569; margin-bottom: 24px;">
        Dashboard ini ditujukan bagi berbagai pihak yang membutuhkan informasi terkait kondisi pemuda dan ketenagakerjaan, antara lain:
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 24px; margin-bottom: 40px;">
        <div class="hover-card" style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 32px 24px; border-radius: 16px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.05);">
            <div style="font-size: 48px; margin-bottom: 16px;">🏛️</div>
            <div style="font-weight: 800; color: #0F172A; font-size: 16px; margin-bottom: 12px; font-family: 'Outfit', sans-serif;">Pemerintah & Pembuat Kebijakan</div>
            <div style="font-size: 16px; color: #64748B; line-height: 1.6;">Sebagai bahan <b>pertimbangan strategis</b> dalam merancang program pendidikan, pelatihan vokasi, dan ketenagakerjaan.</div>
        </div>
        <div class="hover-card" style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 32px 24px; border-radius: 16px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.05);">
            <div style="font-size: 48px; margin-bottom: 16px;">🎓</div>
            <div style="font-weight: 800; color: #0F172A; font-size: 16px; margin-bottom: 12px; font-family: 'Outfit', sans-serif;">Akademisi & Peneliti</div>
            <div style="font-size: 16px; color: #64748B; line-height: 1.6;">Untuk <b>mengeksplorasi</b> faktor-faktor yang berkaitan dengan NEET <b>secara empiris</b> dan <b>mendukung riset lanjutan</b>.</div>
        </div>
        <div class="hover-card" style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 32px 24px; border-radius: 16px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.05);">
            <div style="font-size: 48px; margin-bottom: 16px;">👥</div>
            <div style="font-weight: 800; color: #0F172A; font-size: 16px; margin-bottom: 12px; font-family: 'Outfit', sans-serif;">Masyarakat & Mahasiswa</div>
            <div style="font-size: 16px; color: #64748B; line-height: 1.6;">Untuk memperoleh <b>pemahaman yang lebih baik</b> mengenai tantangan kondisi pemuda di berbagai provinsi di Indonesia.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW & DISTRIBUSI
# ════════════════════════════════════════════════════════════
elif page == "Overview NEET":
    st.markdown("""
<div class="page-header">
    <div style="font-size: 26px; margin-bottom: 4px;">📂</div>
    <div class="page-title">Overview & Distribusi Pemuda NEET</div>
    <div class="page-desc">Gambaran makro kondisi pemuda NEET (Not in Education, Employment, or Training) di 38 provinsi Indonesia</div>
</div>
""", unsafe_allow_html=True)

    # KPI Cards
    nat_avg  = df["Target_NEET"].mean()
    highest  = df.loc[df["Target_NEET"].idxmax()]
    lowest   = df.loc[df["Target_NEET"].idxmin()]
    n_kritis = len(df[df["cluster"] == 3])

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card kpi-blue">
            <div class="kpi-label">Rata-rata Nasional</div>
            <div class="kpi-value">{nat_avg:.2f}%</div>
            <div class="kpi-sub">38 Provinsi · 2024</div>
            <div class="kpi-icon">📊</div>
        </div>
        <div class="kpi-card kpi-red">
            <div class="kpi-label">NEET TERTINGGI</div>
            <div class="kpi-value">{highest["Target_NEET"]:.1f}%</div>
            <div class="kpi-sub">{highest["Provinsi"]}</div>
            <div class="kpi-icon">⬆️</div>
        </div>
        <div class="kpi-card kpi-green">
            <div class="kpi-label">NEET TERENDAH</div>
            <div class="kpi-value">{lowest["Target_NEET"]:.1f}%</div>
            <div class="kpi-sub">{lowest["Provinsi"]}</div>
            <div class="kpi-icon">⬇️</div>
        </div>
        <div class="kpi-card kpi-yellow">
            <div class="kpi-label">Jumlah Provinsi</div>
            <div class="kpi-value" style="font-size: 26px;">38 Provinsi</div>
            <div class="kpi-sub">Total Provinsi Analisis</div>
            <div class="kpi-icon">🗺️</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tren Nasional Line Chart
    with st.container(border=True):
        st.markdown('<div class="card-title">📈 TREN TINGKAT PEMUDA NEET NASIONAL (2015–2025)</div>', unsafe_allow_html=True)
        
        years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
        neet_trend = [24.77, 23.19, 21.41, 22.15, 21.77, 24.28, 22.4, 23.22, 22.25, 20.31, 19.44]
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=years, y=neet_trend,
            mode='lines+markers',
            name='NEET Nasional',
            line=dict(color="#1E3A8A", width=3),
            marker=dict(size=8, color="#1E3A8A"),
            hovertemplate="<b>Tahun %{x}</b><br>NEET: %{y:.2f}%<extra></extra>"
        ))
        
        fig_line.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(tickmode='linear', dtick=1, gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=12)),
            yaxis=dict(title="Angka NEET (%)", gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=12)),
            paper_bgcolor="white", plot_bgcolor="white",
            showlegend=False
        )
        st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})
        
        st.markdown("""
        <div class="insight-box" style="margin-top: 10px;">
            <div class="insight-title">💡 Insight Tren Nasional</div>
            Meskipun angka NEET nasional terus <b>menurun</b> sejak 2015, sebanyak <b>15,99%</b> pemuda Indonesia masih <b>berada dalam kondisi NEET pada tahun 2025</b>. Artinya, <b>hampir 1 dari 5 pemuda belum terhubung dengan pendidikan</b>, <b>pelatihan</b>, <b>maupun dunia kerja</b>.
        </div>
        """, unsafe_allow_html=True)

    # Filter & Map row
    

    if True:
        with st.container(border=True):
            st.markdown('<div class="card-title">🗺️ SEBARAN TINGKAT NEET PER PROVINSI (2024)</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-subtitle">Warna gradasi biru menunjukkan intensitas NEET (semakin gelap = semakin tinggi)</div>', unsafe_allow_html=True)

            def neet_color_val(neet):
                if neet > 30: return "#1E3A8A"
                if neet > 24: return "#2563EB"
                if neet > 18: return "#60A5FA"
                if neet > 12: return "#93C5FD"
                return "#EFF6FF"

            fig_map = go.Figure(go.Choropleth(
                geojson=geojson,
                locations=df["kode_provinsi"],
                z=df["Target_NEET"],
                featureidkey="properties.KODE_PROV",
                colorscale=[[0,"#EFF6FF"],[0.33,"#93C5FD"],[0.66,"#3B82F6"],[1,"#1E3A8A"]],
                zmin=df["Target_NEET"].min(),
                zmax=df["Target_NEET"].max(),
                colorbar=dict(
                    title=dict(text="NEET %", font=dict(size=13)),
                    thickness=12,
                    len=0.6,
                    tickfont=dict(size=12)
                ),
                hovertemplate="<b>%{text}</b><br>NEET: %{z:.2f}%<extra></extra>",
                text=df["Provinsi"],
            ))
            fig_map.update_geos(
                visible=False,
                fitbounds="locations",
                bgcolor="white",
            )
            fig_map.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=500,
                geo=dict(bgcolor="white"),
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

            st.markdown("""
            <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;margin-bottom:12px;">
                <span style="display:inline-flex;align-items:center;gap:5px;font-size:13px;"><span style="width:12px;height:12px;background:#EFF6FF;border-radius:2px;display:inline-block;"></span> ≤ 12%</span>
                <span style="display:inline-flex;align-items:center;gap:5px;font-size:13px;"><span style="width:12px;height:12px;background:#93C5FD;border-radius:2px;display:inline-block;"></span> 12–18%</span>
                <span style="display:inline-flex;align-items:center;gap:5px;font-size:13px;"><span style="width:12px;height:12px;background:#60A5FA;border-radius:2px;display:inline-block;"></span> 18–24%</span>
                <span style="display:inline-flex;align-items:center;gap:5px;font-size:13px;"><span style="width:12px;height:12px;background:#2563EB;border-radius:2px;display:inline-block;"></span> 24–30%</span>
                <span style="display:inline-flex;align-items:center;gap:5px;font-size:13px;"><span style="width:12px;height:12px;background:#1E3A8A;border-radius:2px;display:inline-block;"></span> > 30%</span>
            </div>
            <div class="insight-box" style="margin-top: 10px;">
                <div class="insight-title">💡 Insight Spasial</div>
                Sebaran NEET menunjukkan <b>kesenjangan antarwilayah</b>. Provinsi di <b>kawasan Papua</b> dan <b>Nusa Tenggara</b> memiliki tingkat <b>NEET tertinggi</b>, sedangkan sebagian besar provinsi di <b>Jawa dan Bali</b> berada pada kategori yang <b>lebih rendah</b>.
            </div>
            """, unsafe_allow_html=True)

    if True:
        with st.container(border=True):
            st.markdown('<div class="card-title">📊 TINGKAT NEET PER PROVINSI (2024)</div>', unsafe_allow_html=True)

            pulau_options = ["Semua Pulau"] + sorted(df["pulau"].unique().tolist())
            sel_pulau = st.selectbox("Filter Wilayah:", pulau_options, key="pulau_filter")

            if sel_pulau == "Semua Pulau":
                bar_df = df.copy()
            else:
                bar_df = df[df["pulau"] == sel_pulau].copy()

            bar_df = bar_df.sort_values("Target_NEET", ascending=True)

            fig_bar = go.Figure(go.Bar(
                x=bar_df["Target_NEET"],
                y=bar_df["Provinsi"],
                orientation="h",
                marker=dict(color=bar_df["Target_NEET"], colorscale=[[0, "#6db5d8"], [1, "#1e3a8a"]]),
                hovertemplate="<b>%{y}</b><br>NEET: %{x:.2f}%<extra></extra>",
            ))
            fig_bar.add_vline(
                x=nat_avg,
                line_dash="dash",
                line_color="#DC2626",
                line_width=2,
                annotation_text=f"Nasional: {nat_avg:.2f}%",
                annotation_position="top right",
                annotation_font_size=12,
                annotation_bgcolor="rgba(254,226,226,0.9)",
                annotation_font_color="#DC2626",
            )
            fig_bar.update_layout(
                height=480 if len(bar_df) > 15 else 320,
                margin=dict(l=0, r=10, t=10, b=0),
                xaxis=dict(range=[0,35], ticksuffix="%", gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=12)),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=12)),
                paper_bgcolor="white",
                plot_bgcolor="white",
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
            st.markdown("""
            <div class="insight-box" style="margin-top: 10px;">
                <div class="insight-title">💡 Insight Distribusi</div>
                Sebanyak <b>18 provinsi</b> di Indonesia masih mencatatkan <b>angka NEET di atas rata-rata nasional (20,46%)</b>. Sebagian besar provinsi di <b>Pulau Jawa</b> sudah menunjukkan <b>performa yang baik</b>, sedangkan <b>posisi dengan NEET tertinggi</b> didominasi oleh provinsi-provinsi di kawasan <b>Indonesia Timur</b>.
            </div>
            """, unsafe_allow_html=True)



# ════════════════════════════════════════════════════════════
# PAGE 2 — REGRESI & KORELASI
# ════════════════════════════════════════════════════════════
elif page == "Faktor Penentu NEET":
    st.markdown('''
<div class="page-header">
    <div style="font-size: 26px; margin-bottom: 4px;">📈</div>
    <div class="page-title">Analisis Faktor Penentu Tingkat NEET</div>
    <div class="page-desc">Hasil pengujian regresi OLS terhadap variabel makro ekonomi & pendidikan</div>
</div>
''', unsafe_allow_html=True)

    # Badges di bagian atas
    st.markdown('''
    <div style="display:flex; gap:16px; margin-bottom: 24px;">
        <div style="background:white; border:1px solid #E2E8F0; border-radius:8px; padding:12px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size:11px; color:#64748B; font-weight:600; text-transform:uppercase;">R² (R-Squared)</div>
            <div style="font-size:24px; color:#1E3A8A; font-weight:800;">70,7%</div>
        </div>
        <div style="background:white; border:1px solid #E2E8F0; border-radius:8px; padding:12px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size:11px; color:#64748B; font-weight:600; text-transform:uppercase;">Adjusted R-Squared</div>
            <div style="font-size:24px; color:#1E3A8A; font-weight:800;">68,11%</div>
        </div>
        <div style="background:#ECFDF5; border:1px solid #A7F3D0; border-radius:8px; padding:12px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size:11px; color:#059669; font-weight:600; text-transform:uppercase;">Variabel Paling Berpengaruh</div>
            <div style="font-size:18px; color:#059669; font-weight:800;">Indeks Pembangunan Manusia (IPM)</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Section 1 — Eksplorasi Awal
    st.markdown('<div class="section-title">Bagian 1 — Eksplorasi Hubungan Awal</div>', unsafe_allow_html=True)
    col_heat, col_corr = st.columns([1,1], gap="large")

    with col_heat:
        with st.container(border=True):
            st.markdown('<div class="card-title">🌡️ KORELASI ANTARVARIABEL</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-subtitle">&nbsp;</div>', unsafe_allow_html=True)

            heat_vars = CORR_MATRIX["vars"]
            mat = np.array(CORR_MATRIX["matrix"])
            colorscale = [
                [0.0, "#1E293B"], [0.25, "#475569"], [0.5, "#F1F5F9"], [0.75, "#60A5FA"], [1.0, "#1E3A8A"]
            ]
            fig_heat = go.Figure(go.Heatmap(
                z=mat, x=heat_vars, y=heat_vars,
                colorscale=colorscale, zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in mat],
                texttemplate="%{text}", textfont=dict(size=14),
                hoverongaps=False, colorbar=dict(thickness=12, len=0.7, tickfont=dict(size=12)),
            ))
            fig_heat.update_layout(
                height=300, margin=dict(l=0,r=0,t=10,b=0),
                xaxis=dict(tickfont=dict(size=13)),
                yaxis=dict(tickfont=dict(size=13), autorange="reversed"),
                paper_bgcolor="white", plot_bgcolor="white",
            )
            st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})
            st.markdown('''
            
            ''', unsafe_allow_html=True)

    with col_corr:
        with st.container(border=True):
            st.markdown('<div class="card-title">📏 KORELASI VARIABEL PREDIKTOR DENGAN TINGKAT NEET</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-subtitle">Kekuatan & arah hubungan tiap variabel</div>', unsafe_allow_html=True)

            vars_corr = ["IPM", "SMK_Int", "SMK_Komp", "Miskin", "TPT"]
            corr_vals = [-0.72, -0.52, -0.44, 0.61, 0.55]
            colors_div = ["#1E3A8A" if c < 0 else "#60A5FA" for c in corr_vals]

            fig_div = go.Figure(go.Bar(
                x=corr_vals, y=vars_corr, orientation="h",
                marker_color=colors_div, hovertemplate="%{y}: r = %{x:.2f}<extra></extra>",
                text=[f"r = {c:+.2f}" for c in corr_vals], textposition="outside", textfont=dict(size=13),
            ))
            fig_div.add_vline(x=0, line_color="#64748B", line_width=1.5)
            fig_div.update_layout(
                height=300, margin=dict(l=0,r=50,t=10,b=0),
                xaxis=dict(range=[-1,1], gridcolor="rgba(0,0,0,0.06)", tickfont=dict(size=12), zeroline=False),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=13)),
                paper_bgcolor="white", plot_bgcolor="white", showlegend=False,
            )
            st.plotly_chart(fig_div, use_container_width=True, config={"displayModeBar": False})

            st.markdown('''
            
            ''', unsafe_allow_html=True)

    st.markdown('''
                <div class="insight-box" style="margin-top: 10px;">
                    <div class="insight-title">💡 Insight Korelasi Utama</div>
                    <b>Indeks Pembangunan Manusia (IPM)</b> memiliki <b>hubungan negatif terkuat</b> dengan NEET (r = -0,72), menunjukkan bahwa provinsi dengan <b>kualitas pembangunan manusia yang lebih baik</b> cenderung memiliki <b>tingkat NEET yang lebih rendah</b>. Sebaliknya, <b>tingkat kemiskinan</b> (r = 0,61) dan <b>pengangguran terbuka</b> (r = 0,55) berkaitan dengan <b>peningkatan NEET</b>.
                </div>
                ''', unsafe_allow_html=True)

    # Section 2 — Scatter Plot Interaktif
    st.markdown('<div class="section-title">Bagian 2 — Pemodelan Inti</div>', unsafe_allow_html=True)

    var_config = {
        "TPT": {"beta":0.659, "title":"📈 Tingkat Pengangguran Terbuka (TPT) vs NEET", "sub":"Hubungan positif (elastisitas = +0.659)", "insight":"<b>Tingkat NEET cenderung meningkat</b> seiring <b>meningkatnya tingkat pengangguran terbuka</b>. Provinsi dengan <b>pengangguran yang lebih tinggi</b> umumnya memiliki <b>proporsi pemuda NEET yang lebih besar</b>."},
        "IPM": {"beta":-0.037, "title":"📉 Indeks Pembangunan Manusia (IPM) vs NEET", "sub":"Hubungan negatif (koefisien = −0.037)", "insight":"Provinsi dengan <b>IPM yang lebih tinggi</b> cenderung memiliki <b>tingkat NEET yang lebih rendah</b>. Hubungan ini merupakan yang paling kuat dibandingkan variabel lainnya, menunjukkan <b>pentingnya kualitas pembangunan manusia</b> dalam menekan NEET."},
        "SMK_Komp": {"beta":-0.005, "title":"💻 SMK dengan Fasilitas Komputer vs NEET", "sub":"Hubungan negatif (koefisien = −0.005)", "insight":"Semakin banyak <b>SMK yang memiliki fasilitas komputer</b>, <b>tingkat NEET cenderung semakin rendah</b>. Hal ini mengindikasikan <b>pentingnya akses terhadap sarana pendidikan</b> yang memadai dalam mendukung pengembangan keterampilan pemuda."},
        "Miskin": {"beta":0.5, "title":"📊 Persentase Penduduk Miskin vs NEET", "sub":"Hubungan positif", "insight":"Provinsi dengan <b>tingkat kemiskinan yang lebih tinggi</b> cenderung memiliki <b>tingkat NEET yang lebih tinggi</b>. Hal ini menunjukkan adanya keterkaitan antara <b>kerentanan ekonomi dan keterlibatan pemuda</b> dalam pendidikan maupun pekerjaan."}
    }

    col_var, col_act_pred = st.columns([1,1], gap="large")
    with col_var:
        with st.container(border=True):
                st.markdown('<div class="card-title">🔍 PENGARUH VARIABEL TERHADAP TINGKAT NEET</div>', unsafe_allow_html=True)
                var_labels = {
                    "TPT": "Tingkat Pengangguran Terbuka (TPT)",
                    "IPM": "Indeks Pembangunan Manusia (IPM)",
                    "SMK_Komp": "SMK dengan Fasilitas Komputer",
                    "Miskin": "Persentase Penduduk Miskin"
                }
                sel_var = st.selectbox("Pilih Variabel Prediktor:", list(var_config.keys()), format_func=lambda x: var_labels[x], key="scatter_var")
                vc = var_config[sel_var]
            
                # We compute trendline manually
                x_data = df[sel_var]
                x_min, x_max = x_data.min(), x_data.max()
                x_line = np.linspace(x_min, x_max, 50)
            
                # Simple linear fit for trendline to be safe for any variable
                z = np.polyfit(x_data, df["Target_NEET"], 1)
                p = np.poly1d(z)
                y_line = p(x_line)
    
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=df[sel_var], y=df["Target_NEET"],
                    mode="markers", name="Provinsi",
                    marker=dict(color=NAVY, size=9, opacity=0.85),
                    hovertemplate="<b>%{text}</b><br>"+sel_var+": %{x:.2f}<br>NEET: %{y:.2f}%<extra></extra>",
                    text=df["Provinsi"],
                ))
                fig_sc.add_trace(go.Scatter(
                    x=x_line, y=y_line, mode="lines", name="Regression Line",
                    line=dict(color="#DC2626", width=3), hoverinfo="skip",
                ))
                fig_sc.update_layout(
                    height=400, margin=dict(l=0,r=0,t=10,b=0),
                    xaxis=dict(title=sel_var, gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=13)),
                    yaxis=dict(title="NEET (%)", ticksuffix="%", gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=13)),
                    legend=dict(font=dict(size=13), orientation="h", y=-0.2),
                    paper_bgcolor="white", plot_bgcolor="white",
                )
                st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})
            
                st.markdown(f'''
                <div class="insight-box">
                    <div class="insight-title">💡 Insight: {vc["title"]}</div>
                    {vc["insight"]}
                </div>
                ''', unsafe_allow_html=True)
    with col_act_pred:
        with st.container(border=True):
                st.markdown('<div class="card-title">🎯 PERBANDINGAN NILAI AKTUAL DAN PREDIKSI NEET</div>', unsafe_allow_html=True)
            
                fig_ap = go.Figure()
                fig_ap.add_trace(go.Scatter(
                    x=df["Target_NEET"], y=df["pred_neet"],
                    mode="markers", name="Provinsi",
                    marker=dict(color=NAVY, size=9, opacity=0.85),
                    hovertemplate="<b>%{text}</b><br>Actual NEET: %{x:.2f}%<br>Predicted: %{y:.2f}%<extra></extra>",
                    text=df["Provinsi"],
                ))
            
                # Reference Line (y=x)
                min_val = min(df["Target_NEET"].min(), df["pred_neet"].min())
                max_val = max(df["Target_NEET"].max(), df["pred_neet"].max())
                fig_ap.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val], mode="lines", name="Perfect Prediction (100%)",
                    line=dict(color="#DC2626", width=2, dash="dash"), hoverinfo="skip",
                ))
            
                fig_ap.update_layout(
                    height=400, margin=dict(l=0,r=0,t=10,b=0),
                    xaxis=dict(title="Actual NEET (%)", gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=13)),
                    yaxis=dict(title="Predicted NEET (%)", gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=13)),
                    legend=dict(font=dict(size=13), orientation="h", y=-0.2),
                    paper_bgcolor="white", plot_bgcolor="white",
                )
                st.plotly_chart(fig_ap, use_container_width=True, config={"displayModeBar": False})
            
                st.markdown('''
                <div class="insight-box">
                    <div class="insight-title">💡 Insight Model Akurasi</div>
                    Sebagian besar titik berada di sekitar garis prediksi sempurna, menunjukkan bahwa <b>model mampu menangkap pola utama tingkat NEET antarprovinsi dengan cukup baik</b>. Namun, beberapa provinsi masih berada cukup jauh dari garis, yang mengindikasikan <b>adanya faktor lain di luar model yang turut memengaruhi tingkat NEET</b>.
                </div>
                ''', unsafe_allow_html=True)





elif page == "Clustering Risiko NEET":
    st.markdown("""
<div class="page-header">
    <div style="font-size: 26px; margin-bottom: 4px;">🗺️</div>
    <div class="page-title" style="font-weight: 800;">Analisis Clustering Risiko Pemuda NEET</div>
    <div class="page-desc">Hasil segmentasi <b>K-Means (k=3)</b> clustering terhadap 38 provinsi berdasarkan profil NEET</div>
</div>
""", unsafe_allow_html=True)

    c1_df = df[df["cluster"]==1]
    c2_df = df[df["cluster"]==2]
    c3_df = df[df["cluster"]==3]

    with st.container(border=True):
        st.markdown('<div class="card-title">🗺️ PETA CLUSTER TINGKAT RISIKO PEMUDA NEET</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Distribusi geografis ketiga cluster K-Means</div>', unsafe_allow_html=True)

        df_map = df.copy()
        df_map["cluster_str"] = df_map["cluster"].map({1:"Provinsi Berkinerja Baik",2:"Provinsi Transisi",3:"Provinsi Prioritas Intervensi"})
        color_discrete = {"Provinsi Berkinerja Baik":"#059669","Provinsi Transisi":"#EAB308","Provinsi Prioritas Intervensi":"#DC2626"}

        fig_cmap = go.Figure(go.Choropleth(
            geojson=geojson,
            locations=df_map["kode_provinsi"],
            z=df_map["cluster"].astype(float),
            featureidkey="properties.KODE_PROV",
            colorscale=[[0,"#059669"],[0.5,"#EAB308"],[1,"#DC2626"]],
            zmin=1, zmax=3,
            showscale=False,
            hovertemplate="<b>%{text}</b><br>Cluster: %{z:.0f}<extra></extra>",
            text=df_map["Provinsi"] + " — " + df_map["cluster_str"],
        ))
        fig_cmap.update_geos(visible=False, fitbounds="locations", bgcolor="white")
        fig_cmap.update_layout(
            height=450, margin=dict(l=0,r=0,t=0,b=0),
            geo=dict(bgcolor="white"),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig_cmap, use_container_width=True, config={"displayModeBar": False})

        st.markdown("""
        <div style="display:flex;gap:12px;margin-top:8px;flex-wrap:wrap;justify-content:center;">
            <span style="font-size:13px;display:inline-flex;align-items:center;gap:6px;">
                <span style="width:14px;height:14px;background:#059669;border-radius:3px;display:inline-block;"></span>Provinsi Berkinerja Baik</span>
            <span style="font-size:13px;display:inline-flex;align-items:center;gap:6px;">
                <span style="width:14px;height:14px;background:#EAB308;border-radius:3px;display:inline-block;"></span>Provinsi Transisi</span>
            <span style="font-size:13px;display:inline-flex;align-items:center;gap:6px;">
                <span style="width:14px;height:14px;background:#DC2626;border-radius:3px;display:inline-block;"></span>Provinsi Prioritas Intervensi</span>
        </div>
        """, unsafe_allow_html=True)


    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown(f"""
        <div class="cluster-card green">
            <div class="cc-title" style="color:#059669;">🟢 Provinsi Berkinerja Baik</div>
            <div class="cc-stat" style="color:#059669;">{len(c1_df)} Provinsi</div>
            <div class="cc-item">📍 <b>Pemuda NEET rata-rata: {c1_df["Target_NEET"].mean():.1f}%</b></div>
            <div class="cc-item">Provinsi dalam kelompok ini memiliki <b>tingkat NEET paling rendah</b>, didukung oleh IPM yang relatif tinggi, akses fasilitas komputer di SMK yang lebih baik, serta kondisi ketenagakerjaan yang lebih stabil.</div>
        </div>
        """, unsafe_allow_html=True)

    with cc2:
        st.markdown(f"""
        <div class="cluster-card yellow">
            <div class="cc-title" style="color:#EAB308;">🟡 Provinsi Transisi</div>
            <div class="cc-stat" style="color:#EAB308;">{len(c2_df)} Provinsi</div>
            <div class="cc-item">📍 <b>Pemuda NEET rata-rata: {c2_df["Target_NEET"].mean():.1f}%</b></div>
            <div class="cc-item">Provinsi pada cluster ini menunjukkan <b>kondisi menengah</b>. Tingkat NEET masih cukup tinggi dan kualitas pembangunan manusia maupun fasilitas pendidikan belum sebaik kelompok risiko rendah.</div>
        </div>
        """, unsafe_allow_html=True)

    with cc3:
        st.markdown(f"""
        <div class="cluster-card red">
            <div class="cc-title" style="color:#DC2626;">🔴 Provinsi Prioritas Intervensi</div>
            <div class="cc-stat" style="color:#DC2626;">{len(c3_df)} Provinsi</div>
            <div class="cc-item">📍 <b>Pemuda NEET rata-rata: {c3_df["Target_NEET"].mean():.1f}%</b></div>
            <div class="cc-item">Kelompok ini memiliki <b>tingkat NEET tertinggi dan IPM terendah</b>. Karakteristik tersebut menunjukkan perlunya <b>prioritas intervensi untuk meningkatkan kualitas sumber daya manusia dan akses pendidikan</b></div>
        </div>
        """, unsafe_allow_html=True)



    st.markdown("""
    <div class="insight-box" style="margin-top:10px;margin-bottom:20px;">
        <b>Sebagian besar provinsi</b> berada pada kelompok <b>Berkinerja Baik</b> dan <b>Transisi</b>, sementara kelompok <b>Priroitas Intervensi</b> hanya ditemukan di wilayah <b>Papua</b>. Hal ini menunjukkan bahwa <b>tantangan NEET paling serius terkonsentrasi pada wilayah tertentu</b> dan memerlukan <b>intervensi yang lebih spesifik</b> dibandingkan provinsi lainnya.
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="card-title">🕸️ PROFIL KARAKTERISTIK CLUSTER</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Komparasi rata-rata 5 dimensi antar cluster</div>', unsafe_allow_html=True)

        radar_labels = ["Indeks Pembangunan Manusia (IPM)", "SMK dengan Fasilitas Komputer", "Persentase Penduduk Miskin", "Tingkat Pengangguran Terbuka (TPT)", "Pemuda NEET"]
        radar_data = {
            1: [85, 72, 90, 75, 80],
            2: [60, 50, 65, 55, 50],
            3: [30, 28, 20, 40, 20],
        }
        fig_rad = go.Figure()
        for c in [1,2,3]:
            vals = radar_data[c] + [radar_data[c][0]]
            lbls = radar_labels + [radar_labels[0]]
            fig_rad.add_trace(go.Scatterpolar(
                r=vals, theta=lbls,
                fill="toself",
                name=CLUSTER_SHORT[c],
                fillcolor=hex_to_rgba(CLUSTER_COLORS[c], 0.15),
                line=dict(color=CLUSTER_COLORS[c], width=2),
                marker=dict(size=5, color=CLUSTER_COLORS[c]),
            ))
        fig_rad.update_layout(
            height=450, margin=dict(l=30,r=30,t=20,b=0),
            polar=dict(
                radialaxis=dict(visible=True, range=[0,100], showticklabels=False, showgrid=False, showline=False, ticks="", tickfont=dict(size=11)),
                angularaxis=dict(tickfont=dict(size=12, weight=600), gridcolor="#CBD5E1", gridwidth=1.5, linecolor="#CBD5E1", linewidth=1.5),
            ),
            legend=dict(font=dict(size=12), orientation="h", y=-0.15),
            paper_bgcolor="white", plot_bgcolor="white",
            showlegend=True,
        )
        st.plotly_chart(fig_rad, use_container_width=True, config={"displayModeBar": False})
        st.markdown("""
        <div class="insight-box" style="margin-top: 10px;">
            <div class="insight-title">💡 Insight Profil Radar</div>
            Provinsi-provinsi di <b>Provinsi Berkinerja Baik</b> menunjukkan <b>capaian yang paling baik</b> di hampir semua indikator pembangunan. Sebaliknya, provinsi-provinsi di <b>Cluster 3 (Provinsi Prioritas Intervensi)</b> masih dihadapkan pada kendala <b>tingginya pengangguran terbuka dan kemiskinan</b>, ditambah dengan <b>terbatasnya fasilitas atau kesempatan pelatihan</b> bagi masyarakatnya.
        </div>
        """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="card-title">📋 RINGKASAN HASIL CLUSTERING TIAP PROVINSI</div>', unsafe_allow_html=True)
        
        tc1, tc2 = st.columns([2, 1])
        with tc1:
            cluster_filter = st.selectbox("Filter:", ["Semua", "Provinsi Berkinerja Baik", "Provinsi Transisi", "Provinsi Prioritas Intervensi"], key="cluster_table_filter")
        with tc2:
            search_q = st.text_input("🔍 Cari provinsi:", key="search_prov", placeholder="Ketik nama provinsi...")

        tbl_df = df.copy()
        if cluster_filter != "Semua":
            cmap = {"Provinsi Berkinerja Baik": 1, "Provinsi Transisi": 2, "Provinsi Prioritas Intervensi": 3}
            c_num = cmap.get(cluster_filter, 1)
            tbl_df = tbl_df[tbl_df["cluster"] == c_num]
        if search_q:
            tbl_df = tbl_df[tbl_df["Provinsi"].str.contains(search_q, case=False, na=False)]

        tbl_df = tbl_df.sort_values("Target_NEET", ascending=False).reset_index(drop=True)

        rows_html = ""
        for i, row in tbl_df.iterrows():
            dot_color = CLUSTER_COLORS[row["cluster"]]
            rows_html += f"""<tr>
                <td style="color:#64748B;">{i+1}</td>
                <td><strong>{row["Provinsi"]}</strong></td>
                <td><span class="cluster-dot"><span style="width:8px;height:8px;background:{dot_color};border-radius:50%;display:inline-block;"></span>C{row["cluster"]}</span></td>
                <td style="font-weight:600;">{row["Target_NEET"]:.1f}%</td>
                <td>{row["TPT"]:.2f}%</td>
                <td>{row["IPM"]:.1f}</td>
                <td>{row["SMK_Komp"]:.1f}%</td>
                <td>{row["Miskin"]:.2f}%</td>
            </tr>"""

        st.markdown(f"""
        <div style="margin-bottom:16px;">
        <table class="prov-table">
            <thead><tr><th>#</th><th>Provinsi</th><th>Cluster</th><th style="width:14%;">Pemuda NEET</th><th style="width:14%;">TPT</th><th style="width:14%;">IPM</th><th style="width:14%;">SMK Komputer</th><th style="width:14%;">Kemiskinan</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)
        
        csv_data = tbl_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Data Tabel",
            data=csv_data,
            file_name="data_provinsi_cluster.csv",
            mime="text/csv",
        )



elif page == "Simulasi & Rekomendasi":
    st.markdown("""
<div class="page-header">
    <div style="font-size: 26px; margin-bottom: 4px;">🎯</div>
    <div class="page-title">Simulasi Kebijakan & Rekomendasi</div>
    <div class="page-desc">Proyeksi dampak intervensi kebijakan terhadap angka NEET menggunakan model regresi</div>
</div>
""", unsafe_allow_html=True)

    # Key findings
    st.markdown('', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown("""
        <div class="finding-card">
            <span class="finding-num">TEMUAN 1</span>
            <div class="finding-text"><strong>Peningkatan Akses Teknologi di SMK</strong></div>
            <div class="finding-text">Setiap kenaikan 10% SMK yang memiliki fasilitas komputer berkaitan dengan penurunan NEET sekitar 0,5 poin persentase.</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="finding-card">
            <span class="finding-num">TEMUAN 2</span>
            <div class="finding-text"><strong>IPM sebagai Faktor Utama</strong></div>
            <div class="finding-text">IPM merupakan faktor yang paling konsisten berkaitan dengan NEET. Setiap kenaikan 1 poin IPM berkaitan dengan penurunan NEET sekitar 0,037%.</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="finding-card">
            <span class="finding-num">TEMUAN 3</span>
            <div class="finding-text"><strong>Pengangguran dan NEET bergerak bersama</strong></div>
            <div class="finding-text">Provinsi dengan tingkat pengangguran terbuka yang lebih tinggi cenderung memiliki tingkat NEET yang lebih tinggi.</div>
        </div>
        """, unsafe_allow_html=True)
    with f4:
        st.markdown("""
        <div class="finding-card">
            <span class="finding-num">TEMUAN 4</span>
            <div class="finding-text"><strong>Konsentrasi Risiko di Papua</strong></div>
            <div class="finding-text">Papua Tengah dan Papua Pegunungan menjadi kelompok dengan risiko NEET tertinggi dan memerlukan perhatian kebijakan yang lebih spesifik.</div>
        </div>
        """, unsafe_allow_html=True)

    # Simulator
    st.markdown('', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="card-title">🧪 SIMULATOR DAMPAK KEBIJAKAN</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Geser slider untuk melihat proyeksi dampak kebijakan terhadap angka NEET nasional</div>', unsafe_allow_html=True)

        BASE_TPT = 5.82
        BASE_IPM = 72.29
        BASE_SMK = 60.0

        def compute_neet(tpt, ipm, smk):
            log_n = 5.218 + 0.659 * np.log(max(0.01, tpt)) - 0.037 * ipm - 0.005 * smk
            return np.exp(log_n)

        base_neet = compute_neet(BASE_TPT, BASE_IPM, BASE_SMK)

        slider_col, result_col = st.columns([1,1], gap="large")

        with slider_col:
            delta_tpt = st.slider("📉 Perubahan Tingkat Pengangguran Terbuka (TPT) (%)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1, key="tpt_slide",
                                  help="Negatif = kebijakan penyerapan tenaga kerja berhasil menurunkan TPT")
            delta_ipm = st.slider("📈 Perubahan Indeks Pembangunan Manusia (IPM)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1, key="ipm_slide",
                                  help="Simulasi dampak perubahan pembangunan manusia")
            delta_smk = st.slider("💻 Perubahan Persentase SMK dengan Fasilitas Komputer (%)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1, key="smk_slide",
                                  help="Target perubahan digitalisasi sekolah vokasi")

        new_tpt = max(0.01, BASE_TPT + delta_tpt)
        new_ipm = max(0.0, min(100.0, BASE_IPM + delta_ipm))
        smk_val = max(0.0, min(100.0, BASE_SMK + delta_smk))
        
        new_neet = compute_neet(new_tpt, new_ipm, smk_val)
        delta_neet = new_neet - base_neet
        pct_change = (delta_neet / base_neet * 100)

        with result_col:
            result_color = "#059669" if delta_neet < 0 else ("#EAB308" if abs(delta_neet) < 0.5 else "#DC2626")
            direction = "↓ Kebijakan berhasil menurunkan NEET" if delta_neet < 0 else ("← Baseline" if abs(delta_neet)<0.01 else "↑ NEET diprediksi meningkat")

            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f"""
                <div class="sim-result">
                    <div class="sim-label">Prediksi NEET Nasional</div>
                    <div class="sim-value" style="color:{result_color};">{new_neet:.2f}%</div>
                    <div class="sim-sub">Baseline: {base_neet:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with r2:
                st.markdown(f"""
                <div class="sim-result">
                    <div class="sim-label">Perubahan Absolut</div>
                    <div class="sim-value" style="color:{result_color};">{delta_neet:+.2f}%</div>
                    <div class="sim-sub">{pct_change:+.1f}% relatif dari baseline</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:{'#ECFDF5' if delta_neet<0 else '#FEF2F2'};border:1px solid {'#A7F3D0' if delta_neet<0 else '#FECACA'};
                border-radius:8px;padding:12px 16px;margin-top:12px;font-size:15px;font-weight:600;color:{result_color};text-align:center;">
                {direction}
            </div>
            """, unsafe_allow_html=True)

        # Simulation Bar Chart (replacing line chart)
        st.markdown('<div class="card-title" style="margin-top: 20px;">📊 Prediksi Angka NEET Nasional</div>', unsafe_allow_html=True)
        fig_sim = go.Figure(data=[
            go.Bar(
                name='NEET', 
                x=['Baseline', 'Skenario Baru'], 
                y=[base_neet, new_neet],
                marker_color=['#94A3B8', result_color], 
                text=[f"{base_neet:.2f}%", f"{new_neet:.2f}%"], 
                textposition='auto',
                width=[0.4, 0.4]
            )
        ])
        fig_sim.update_layout(
            height=300, margin=dict(l=0,r=0,t=10,b=0),
            yaxis=dict(title="Angka NEET (%)", gridcolor="rgba(0,0,0,0.05)", range=[0, max(base_neet, new_neet) + 5]),
            paper_bgcolor="white", plot_bgcolor="white", showlegend=False
        )
        st.plotly_chart(fig_sim, use_container_width=True, config={"displayModeBar": False})

        selisih_abs = abs(delta_neet)
        prediksi_neet = new_neet
        if delta_neet < -0.01:
            insight_simulasi = f"Skenario ini diperkirakan <b>menurunkan tingkat NEET nasional menjadi {prediksi_neet:.2f}%</b>. Penurunan ini menunjukkan bahwa kombinasi <b>peningkatan kualitas pembangunan manusia</b>, <b>akses fasilitas pendidikan</b>, dan <b>perbaikan kondisi ketenagakerjaan</b> berpotensi <b>mengurangi proporsi pemuda NEET</b>."
        elif delta_neet > 0.01:
            insight_simulasi = f"Skenario ini diperkirakan <b>meningkatkan tingkat NEET nasional menjadi {prediksi_neet:.2f}%</b>. Peningkatan ini menunjukkan bahwa memburuknya <b>indikator pendidikan atau ketenagakerjaan</b> dapat <b>meningkatkan risiko pemuda berada dalam kondisi NEET</b>."
        else:
            insight_simulasi = "Skenario yang dipilih hanya menghasilkan <b>perubahan NEET yang relatif kecil</b> dibandingkan kondisi dasar. Hal ini menunjukkan bahwa kombinasi perubahan indikator yang dimasukkan belum cukup kuat untuk mengubah kondisi NEET secara signifikan."

        st.markdown(f"""
        <div class="insight-box" style="margin-top: 10px;">
            <div class="insight-title">💡 Insight Simulasi</div>
            {insight_simulasi}
        </div>
        """, unsafe_allow_html=True)

    # Recommendations per Province
    with st.container(border=True):
        st.markdown('<div class="card-title" style="text-transform: uppercase;">🎯 REKOMENDASI SPESIFIK TINGKAT PROVINSI</div>', unsafe_allow_html=True)

        rec_data = {
            1: {
                "cls":"green", "label":"PROVINSI BERKINERJA BAIK",
                "recs":[
                    "Mempertahankan <b>kualitas pendidikan</b> dan <b>pelatihan vokasi</b> yang sudah berjalan baik.",
                    "Memperluas <b>program magang</b> dan <b>kerja sama sekolah dengan dunia usaha dan industri</b>.",
                    "Mengembangkan <b>pelatihan keterampilan digital tingkat lanjut</b> (<i>advanced digital skills</i>).",
                    "Memperkuat <b>layanan bimbingan karier</b> dan <b><i>job matching</i></b> bagi lulusan sekolah dan perguruan tinggi.",
                ]
            },
            2: {
                "cls":"yellow", "label":"PROVINSI TRANSISI",
                "recs":[
                    "Meningkatkan <b>akses pelatihan kerja</b> berbasis kebutuhan industri lokal.",
                    "Memperluas kepemilikan <b>fasilitas komputer</b> dan laboratorium praktik di <b>SMK</b>.",
                    "Memperkuat <b>program <i>reskilling</i> dan <i>upskilling</i></b> bagi pemuda yang belum bekerja.",
                    "Mendorong <b>kemitraan</b> antara <b>pemerintah daerah</b>, <b>SMK</b>, dan <b>perusahaan setempat</b>.",
                    "Memfokuskan <b>intervensi</b> pada <b>kabupaten/kota dengan tingkat NEET tertinggi</b> di dalam provinsi.",
                ]
            },
            3: {
                "cls":"red", "label":"PROVINSI PRIORITAS INTERVENSI",
                "recs":[
                    "Menetapkan wilayah sebagai <b>prioritas utama</b> program penurunan NEET nasional.",
                    "Meningkatkan <b>akses pendidikan menengah</b> dan <b>pelatihan kerja</b> bagi pemuda rentan.",
                    "Memberikan <b>bantuan pendidikan</b> dan <b>pelatihan</b> yang terintegrasi dengan <b>program perlindungan sosial</b>.",
                    "Mempercepat <b>pembangunan infrastruktur pendidikan</b> dan <b>fasilitas digital sekolah</b>.",
                    "Mengembangkan <b>program padat karya</b>, <b>kewirausahaan pemuda</b>, dan <b>pendampingan usaha mikro</b>.",
                ]
            }
        }

        prov_options = ["— Pilih Provinsi untuk Rekomendasi —"] + sorted(df["Provinsi"].tolist())
        sel_prov = st.selectbox("Pilih Provinsi:", prov_options, key="prov_select")

        if sel_prov and sel_prov != "— Pilih Provinsi untuk Rekomendasi —":
            prov_row = df[df["Provinsi"] == sel_prov].iloc[0]
            c_num = int(prov_row["cluster"])
            rec = rec_data[c_num]
            border_color = {"green":"#A7F3D0","yellow":"#FDE68A","red":"#FECACA"}[rec["cls"]]
            bg_color = {"green":"#ECFDF5","yellow":"#FFFBEB","red":"#FEF2F2"}[rec["cls"]]
            title_color = {"green":"#059669","yellow":"#EAB308","red":"#DC2626"}[rec["cls"]]

            col_radar, col_rec = st.columns([2, 3], gap="large")
            with col_radar:
                prov_radar_vals = [
                    min(100, prov_row["IPM"]),
                    min(100, prov_row["SMK_Komp"]),
                    max(0, 100 - prov_row["Miskin"]*2),
                    max(0, 100 - prov_row["TPT"]*6),
                    max(0, 100 - prov_row["Target_NEET"]*2.5),
                ]
                fig_pr = go.Figure(go.Scatterpolar(
                    r=prov_radar_vals + [prov_radar_vals[0]],
                    theta=["IPM","SMK dengan Komputer","Kemiskinan","TPT","Pemuda NEET","IPM"],
                    fill="toself",
                    fillcolor="rgba(30, 58, 138, 0.15)",
                    line=dict(color="#1E3A8A", width=2.5),
                    marker=dict(size=6, color="#1E3A8A"),
                    name=sel_prov,
                ))
                fig_pr.update_layout(
                    height=350, margin=dict(l=20,r=20,t=40,b=20),
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0,100], showticklabels=False, showgrid=False, showline=False, ticks="", tickfont=dict(size=11)),
                        angularaxis=dict(tickfont=dict(size=12, weight=600), gridcolor="#CBD5E1", gridwidth=1.5, linecolor="#CBD5E1", linewidth=1.5),
                    ),
                    paper_bgcolor="white",
                    showlegend=False,
                    title=dict(text=f"Profil {sel_prov}", font=dict(size=14, color=NAVY), x=0.5, y=0.98, xanchor="center", yanchor="top"),
                )
                st.plotly_chart(fig_pr, use_container_width=True, config={"displayModeBar": False})

            with col_rec:
                recs_html = "".join([f"<li style='margin-bottom:8px;font-size:15px;'>{r}</li>" for r in rec["recs"]])
                st.markdown(f"""
                <div style="background:{bg_color};border:1px solid {border_color};border-radius:12px;padding:18px 20px;">
                    <div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:{title_color};margin-bottom:4px;">{rec["label"]}</div>
                    <div style="font-size:16px;font-weight:700;color:#0F172A;margin-bottom:14px;">{sel_prov}  ·  Tingkat Pemuda NEET: {prov_row["Target_NEET"]:.2f}%</div>
                    <ul style="padding-left:18px;margin:0;">
                        {recs_html}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
