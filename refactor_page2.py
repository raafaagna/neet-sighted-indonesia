import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update Threshold Line on Page 1
old_vline = """            fig_bar.add_vline(
                x=nat_avg,
                line_dash="dash",
                line_color="#1E3A8A",
                line_width=2,
                annotation_text=f"Nasional: {nat_avg:.2f}%",
                annotation_position="top right",
                annotation_font_size=10,
                annotation_bgcolor="rgba(238,242,255,0.9)",
                annotation_font_color="#1E3A8A",
            )"""
new_vline = """            fig_bar.add_vline(
                x=nat_avg,
                line_dash="dash",
                line_color="#DC2626",
                line_width=2,
                annotation_text=f"Nasional: {nat_avg:.2f}%",
                annotation_position="top right",
                annotation_font_size=10,
                annotation_bgcolor="rgba(254,226,226,0.9)",
                annotation_font_color="#DC2626",
            )"""
if old_vline in content:
    content = content.replace(old_vline, new_vline)
else:
    print("Warning: old_vline not found!")


# 2. Refactor Page 2
page2_start = 'elif page == "Regresi & Korelasi":'
page3_start = 'elif page == "Clustering <b>K-Means</b>":'

# Extract everything between page2_start and page3_start
pattern = re.compile(re.escape(page2_start) + r'(.*?)' + re.escape(page3_start), re.DOTALL)
match = pattern.search(content)

if match:
    new_page2 = """elif page == "Regresi & Korelasi":
    st.markdown('''
<div class="page-header">
    <div style="font-size: 26px; margin-bottom: 4px;">📈</div>
    <div class="page-title">Analisis Regresi & Korelasi</div>
    <div class="page-desc">Hasil pengujian regresi OLS log-log terhadap variabel makro ekonomi & pendidikan</div>
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
    </div>
    ''', unsafe_allow_html=True)

    # Section 1 — Eksplorasi Awal
    st.markdown('<div class="section-title">Bagian 1 — Eksplorasi Hubungan Awal</div>', unsafe_allow_html=True)
    col_heat, col_corr = st.columns([1,1], gap="large")

    with col_heat:
        with st.container(border=True):
            st.markdown('<div class="card-title">🌡️ Heatmap Matriks Korelasi</div>', unsafe_allow_html=True)

            heat_vars = CORR_MATRIX["vars"]
            mat = np.array(CORR_MATRIX["matrix"])
            colorscale = [
                [0.0, "#1E293B"], [0.25, "#475569"], [0.5, "#F1F5F9"], [0.75, "#60A5FA"], [1.0, "#1E3A8A"]
            ]
            fig_heat = go.Figure(go.Heatmap(
                z=mat, x=heat_vars, y=heat_vars,
                colorscale=colorscale, zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in mat],
                texttemplate="%{text}", textfont=dict(size=12),
                hoverongaps=False, colorbar=dict(thickness=12, len=0.7, tickfont=dict(size=10)),
            ))
            fig_heat.update_layout(
                height=250, margin=dict(l=0,r=0,t=10,b=0),
                xaxis=dict(tickfont=dict(size=11)),
                yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
                paper_bgcolor="white", plot_bgcolor="white",
            )
            st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})
            st.markdown('''
            <div class="insight-box" style="margin-top: 10px;">
                <div class="insight-title">💡 Insight Heatmap</div>
                Variabel <b>IPM</b> memiliki <b>korelasi negatif</b> terkuat dengan NEET (r = -0.72) dan <b>Kemiskinan</b> (r = -0.69). Ini membuktikan bahwa <b>investasi pembangunan manusia</b> linear dengan penurunan kemiskinan dan pengangguran pemuda.
            </div>
            ''', unsafe_allow_html=True)

    with col_corr:
        with st.container(border=True):
            st.markdown('<div class="card-title">📏 Korelasi dengan Target NEET</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-subtitle">Kekuatan & arah hubungan tiap variabel</div>', unsafe_allow_html=True)

            vars_corr = ["<b>IPM</b>", "SMK_Int", "<b>SMK_Komp</b>", "Miskin", "<b>TPT</b>"]
            corr_vals = [-0.72, -0.52, -0.44, 0.61, 0.55]
            colors_div = ["#1E3A8A" if c < 0 else "#60A5FA" for c in corr_vals]

            fig_div = go.Figure(go.Bar(
                x=corr_vals, y=vars_corr, orientation="h",
                marker_color=colors_div, hovertemplate="%{y}: r = %{x:.2f}<extra></extra>",
                text=[f"r = {c:+.2f}" for c in corr_vals], textposition="outside", textfont=dict(size=11),
            ))
            fig_div.add_vline(x=0, line_color="#64748B", line_width=1.5)
            fig_div.update_layout(
                height=260, margin=dict(l=0,r=50,t=10,b=0),
                xaxis=dict(range=[-1,1], gridcolor="rgba(0,0,0,0.06)", tickfont=dict(size=10), zeroline=False),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11)),
                paper_bgcolor="white", plot_bgcolor="white", showlegend=False,
            )
            st.plotly_chart(fig_div, use_container_width=True, config={"displayModeBar": False})

            st.markdown('''
            <div class="insight-box">
                <div class="insight-title">📌 Interpretasi</div>
                <b>IPM</b> memiliki <b>korelasi negatif</b> terkuat (r=−0.72): semakin tinggi pembangunan manusia, semakin rendah NEET. <b>TPT</b> menunjukkan <b>korelasi positif</b> terkuat (r=+0.61), mengindikasikan pengangguran terbuka dan NEET saling memperkuat.
            </div>
            ''', unsafe_allow_html=True)

    # Section 2 — Scatter Plot Interaktif
    st.markdown('<div class="section-title">Bagian 2 — Pemodelan Inti</div>', unsafe_allow_html=True)

    var_config = {
        "TPT": {"beta":0.659, "title":"📈 <b>TPT</b> vs NEET", "sub":"Hubungan positif (elastisitas = +0.659)", "insight":"Setiap kenaikan 1% <b>TPT</b>, NEET naik ~0.659 poin. <b>Papua</b> & <b>NTT</b> secara konsisten di atas garis prediksi."},
        "IPM": {"beta":-0.037, "title":"📉 <b>IPM</b> vs NEET", "sub":"Hubungan negatif (koefisien = −0.037)", "insight":"Semakin tinggi <b>IPM</b> sebuah provinsi, angka NEET semakin rendah — membuktikan <b>investasi pembangunan manusia</b> adalah kebijakan efektif."},
        "SMK_Komp": {"beta":-0.005, "title":"💻 <b>SMK_Komp</b> vs NEET", "sub":"Hubungan negatif (koefisien = −0.005)", "insight":"Digitalisasi sekolah vokasi terbukti menurunkan NEET. Provinsi <b>Jawa</b>-<b>Bali</b> dengan <b>SMK_Komp</b> tinggi memiliki NEET lebih rendah."},
        "Miskin": {"beta":0.5, "title":"📊 Kemiskinan vs NEET", "sub":"Hubungan positif", "insight":"Tingkat kemiskinan berbanding lurus dengan NEET, menegaskan ketimpangan struktural."}
    }

    with st.container(border=True):
        st.markdown('<div class="card-title">🔍 Scatter Plot + Regression Line</div>', unsafe_allow_html=True)
        sel_var = st.radio("Pilih Variabel Prediktor:", list(var_config.keys()), horizontal=True, key="scatter_var")
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
        for c in [1,2,3]:
            sub = df[df["cluster"]==c]
            fig_sc.add_trace(go.Scatter(
                x=sub[sel_var], y=sub["Target_NEET"],
                mode="markers", name=CLUSTER_SHORT[c],
                marker=dict(color=CLUSTER_COLORS[c], size=9, opacity=0.85),
                hovertemplate="<b>%{text}</b><br>"+sel_var+": %{x:.2f}<br>NEET: %{y:.2f}%<extra></extra>",
                text=sub["Provinsi"].str.title(),
            ))
        fig_sc.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines", name="Regression Line",
            line=dict(color=NAVY, width=3), hoverinfo="skip",
        ))
        fig_sc.update_layout(
            height=400, margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(title=sel_var, gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=11)),
            yaxis=dict(title="NEET (%)", ticksuffix="%", gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=11)),
            legend=dict(font=dict(size=11), orientation="h", y=-0.2),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})
        
        st.markdown(f'''
        <div class="insight-box">
            <div class="insight-title">💡 Insight: {vc["title"]}</div>
            {vc["insight"]}
        </div>
        ''', unsafe_allow_html=True)


"""
    content = content[:match.start()] + new_page2 + "\n\n" + page3_start + content[match.end():]
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Page 2 successfully refactored.")
else:
    print("Error: Could not locate Page 2 boundaries.")
