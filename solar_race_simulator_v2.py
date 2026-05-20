"""
태양광 레이스 시뮬레이터 v3
논문 1 (서울대) · 논문 2 (BWSC) · 논문 3 (GHI 신뢰구간) · 논문 4 (Sun Chaser II)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import minimize_scalar
import folium
from streamlit_folium import st_folium
import urllib.request
import os

# ── 폰트 설정 (서버/로컬 모두 대응) ─────────────────────────────
def set_font():
    # Streamlit Cloud: packages.txt로 fonts-nanum 설치됨
    # 설치된 폰트 캐시 새로고침
    fm._load_fontmanager(try_read_cache=False)
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ['NanumGothic', 'Nanum Gothic', 'NanumBarunGothic', 'AppleGothic', 'Malgun Gothic']:
        if candidate in available:
            matplotlib.rcParams['font.family'] = candidate
            break
    else:
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['axes.unicode_minus'] = False

set_font()

# ════════════════════════════════════════════════════════════════
# 1. 상수 및 기본 파라미터
# ════════════════════════════════════════════════════════════════
DEFAULT_PARAMS = dict(
    mass_kg=180.0, panel_area_m2=4.0, panel_eff=0.24,
    battery_kwh=5.0, Cd=0.13, Cr=0.003, frontal_area=0.9,
    eta=0.90, aux_power_w=50.0, air_density=1.18, g=9.81,
)

CHECKPOINTS = [
    (0,    "Darwin",        -12.46, 130.84),
    (332,  "Katherine",     -14.47, 132.26),
    (588,  "Daly Waters",   -16.26, 133.38),
    (987,  "Tennant Creek", -19.65, 134.19),
    (1210, "Barrow Creek",  -21.54, 133.88),
    (1493, "Alice Springs", -23.70, 133.88),
    (1766, "Kulgera",       -25.84, 133.29),
    (2179, "Coober Pedy",   -29.01, 134.75),
    (2432, "Port Augusta",  -32.49, 137.77),
    (2720, "Gawler",        -34.60, 138.74),
    (3022, "Adelaide",      -34.93, 138.60),
]
TOTAL_KM  = 3022
DRIVE_HRS = 9.0

GHI_MEAN = np.array([5.5, 6.0, 5.8, 6.0, 5.7, 5.0, 5.5, 5.5])
GHI_STD  = GHI_MEAN * 0.15
CI_Z     = 1.0
WIND_EFF = np.array([1.01, 1.02, 0.99, 1.005, 1.03, 1.01, 1.02, 1.00])

np.random.seed(42)
_x = np.linspace(0, TOTAL_KM, TOTAL_KM)
_base = 200*np.sin(_x/600) + 100*np.sin(_x/200)
_noise = np.cumsum(np.random.randn(TOTAL_KM)*0.8)
ELEVATION = np.clip(_base + (_noise - _noise.mean())*0.4, -50, 600)

# ════════════════════════════════════════════════════════════════
# 2. 물리 모델 (논문 4)
# ════════════════════════════════════════════════════════════════
def power_w(v_kmh, slope_deg, p, wind_f=1.0):
    if v_kmh < 0.1:
        return 0.0
    v = v_kmh / 3.6
    theta = np.radians(slope_deg)
    F1 = 0.5 * p['air_density'] * p['frontal_area'] * p['Cd'] * v**2
    F2 = p['Cr'] * p['mass_kg'] * p['g'] * np.cos(theta)
    F3 = p['mass_kg'] * p['g'] * np.sin(theta)
    return max(0.0, (v/p['eta'])*(F1+F2+F3)*wind_f + p['aux_power_w'])

def solar_w(ghi, area, eff, hour):
    if hour <= 6.0 or hour >= 18.0:
        return 0.0
    angle = np.pi * (hour - 6.0) / 12.0
    peak  = ghi * 1000 / (np.pi/2) * (np.pi/12.0)
    return max(0.0, area * eff * peak * np.sin(angle))

# ════════════════════════════════════════════════════════════════
# 3. 빔서치 (논문 1)
# ════════════════════════════════════════════════════════════════
def beam_search(params, ghi_vals):
    speeds = []
    for day in range(8):
        ghi = max(0.5, float(ghi_vals[day]))
        wf  = float(WIND_EFF[day])
        total_solar = sum(
            solar_w(ghi, params['panel_area_m2'], params['panel_eff'], 8.5+h) / 1000.0
            for h in range(int(DRIVE_HRS))
        )
        def balance(v):
            return abs(power_w(v, 0.0, params, wf)/1000.0 * DRIVE_HRS - total_solar)
        res = minimize_scalar(balance, bounds=(40, 120), method='bounded')
        speeds.append(round(float(np.clip(res.x, 40, 120)), 1))
    return speeds

# ════════════════════════════════════════════════════════════════
# 4. 시뮬레이션 엔진
# ════════════════════════════════════════════════════════════════
def simulate(speed_per_day, params, ghi_vals):
    records = []
    battery   = float(params['battery_kwh'])
    pos_km    = 0.0
    finished  = False
    finish_info = None
    total_h   = 0.0

    for day in range(8):
        if finished:
            break
        v_target  = float(speed_per_day[day]) if day < len(speed_per_day) else float(speed_per_day[-1])
        ghi_today = max(0.5, float(ghi_vals[day]) if day < len(ghi_vals) else float(ghi_vals[-1]))
        wf        = float(WIND_EFF[day]) if day < len(WIND_EFF) else 1.0

        for h_idx in range(int(DRIVE_HRS)):
            hour = 8.0 + h_idx
            Ps   = solar_w(ghi_today, params['panel_area_m2'], params['panel_eff'], hour+0.5)
            si   = int(np.clip(pos_km, 0, TOTAL_KM-2))
            ei   = int(np.clip(pos_km + v_target, 0, TOTAL_KM-1))
            dh   = float(ELEVATION[ei] - ELEVATION[si]) if ei > si else 0.0
            slp  = float(np.degrees(np.arctan(dh / (v_target*1000 + 1e-6))))
            Pu   = power_w(v_target, slp, params, wf)
            net  = (Ps - Pu) / 1000.0

            if battery + net < 0:
                battery  = float(np.clip(battery + Ps/1000.0, 0, params['battery_kwh']))
                v_actual = 0.0
                dist     = 0.0
                Pu_actual = 0.0
            else:
                battery   = float(np.clip(battery + net, 0, params['battery_kwh']))
                v_actual  = v_target
                dist      = v_target
                Pu_actual = Pu

            new_pos = min(pos_km + dist, float(TOTAL_KM))
            pos_km  = new_pos
            total_h += 1.0

            records.append({
                'day':         int(day+1),
                'hour':        float(hour),
                'elapsed_h':   float(total_h),
                'pos_km':      float(pos_km),
                'battery_pct': float(battery/params['battery_kwh']*100),
                'v_kmh':       float(v_actual),
                'P_solar_w':   float(Ps),
                'P_use_w':     float(Pu_actual),
                'net_kwh':     float(net if v_actual>0 else Ps/1000.0),
                'slope_deg':   float(slp),
                'ghi':         float(ghi_today),
                'stopped':     bool(v_actual < 0.1),
            })

            if pos_km >= TOTAL_KM:
                finished    = True
                finish_info = {'day': int(day+1), 'hour': float(hour), 'elapsed_h': float(total_h)}
                break

    df = pd.DataFrame(records)
    # 타입 명시
    for col in ['day','elapsed_h','pos_km','battery_pct','v_kmh','P_solar_w','P_use_w','net_kwh','slope_deg','ghi']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['stopped'] = df['stopped'].astype(bool)
    return df, finished, finish_info

# ════════════════════════════════════════════════════════════════
# 5. UI
# ════════════════════════════════════════════════════════════════
st.set_page_config(page_title="태양광 레이스 시뮬레이터", page_icon="☀️", layout="wide")
st.title("☀️ 태양광 레이스 시뮬레이터")
st.caption("논문 1 (서울대) · 논문 2 (BWSC) · 논문 3 (GHI 신뢰구간) · 논문 4 (Sun Chaser II) 통합 재현 v3")

with st.sidebar:
    st.header("🚗 차량 파라미터 (논문 4)")
    params = {}
    params['mass_kg']       = st.slider("차량 질량 (kg)",    100, 400, 180)
    params['panel_area_m2'] = st.slider("패널 면적 (m²)",    1.0, 4.0, 4.0, 0.1)
    params['panel_eff']     = st.slider("패널 효율",         0.15, 0.30, 0.24, 0.01)
    params['battery_kwh']   = st.slider("배터리 용량 (kWh)", 2.0, 10.0, 5.0, 0.5)
    params['Cd']            = st.slider("공기저항 Cd",        0.08, 0.35, 0.13, 0.01)
    params['Cr']            = st.slider("구름저항 Cr",        0.001, 0.010, 0.003, 0.001, format="%.3f")
    params['eta']           = st.slider("전기-기계 효율 η",   0.80, 0.98, 0.90, 0.01)
    params['frontal_area']  = 0.9
    params['aux_power_w']   = 50.0
    params['air_density']   = 1.18
    params['g']             = 9.81

    st.divider()
    st.header("🌤 날씨 시나리오 (논문 3)")
    scenario = st.radio("GHI 시나리오", ["mean","optimistic","pessimistic"],
        format_func=lambda x: {"mean":"평균","optimistic":"낙관 (+1σ)","pessimistic":"비관 (-1σ)"}[x])

ghi_vals = GHI_MEAN if scenario=="mean" else (
    GHI_MEAN + CI_Z*GHI_STD if scenario=="optimistic" else GHI_MEAN - CI_Z*GHI_STD)

beam_spd = beam_search(params, ghi_vals)

tab1, tab2, tab3, tab4 = st.tabs(["📊 전략 비교", "⏱ 시간별 플래너", "🗺 경로 분석", "⚡ 에너지 상세"])

# ── 탭 1 ────────────────────────────────────────────────────────
with tab1:
    st.subheader("5가지 속도 전략 비교")
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**일별 목표 속도 (내 설정)**")
        custom = []
        for d in range(8):
            v = st.number_input(f"Day {d+1}", 40, 120, int(beam_spd[d]), 1, key=f"spd_{d}")
            custom.append(float(v))
        st.caption(f"💡 빔서치 추천: {[int(v) for v in beam_spd]} km/h")

    strategies = {
        "최소속도 (60)":  [60.0]*8,
        "최대속도 (100)": [100.0]*8,
        "평균속도 (80)":  [80.0]*8,
        "빔서치 최적":    beam_spd,
        "내 설정":        custom,
    }
    colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"]

    results = {}
    for name, spd in strategies.items():
        df_s, fin, finfo = simulate(spd, params, ghi_vals)
        results[name] = (df_s, fin, finfo)

    with col_right:
        fig, axes = plt.subplots(2, 1, figsize=(9, 7))
        ax1 = axes[0]
        for (name, (df_s, _, _)), color in zip(results.items(), colors):
            ax1.plot(df_s['elapsed_h'], df_s['pos_km'], label=name, color=color, lw=1.8)
        ax1.axhline(TOTAL_KM, color='gray', ls='--', lw=0.8, alpha=0.6)
        ax1.set_ylabel("누적 주행거리 (km)")
        ax1.set_title("누적 주행거리")
        ax1.legend(fontsize=8, loc='lower right')
        for d in range(1,8):
            ax1.axvline(d*DRIVE_HRS, color='lightgray', lw=0.6)

        ax2 = axes[1]
        for (name, (df_s, _, _)), color in zip(results.items(), colors):
            ax2.plot(df_s['elapsed_h'], df_s['battery_pct'], label=name, color=color, lw=1.8)
        ax2.axhline(20, color='red',   ls=':', lw=0.8, alpha=0.5)
        ax2.axhline(80, color='green', ls=':', lw=0.8, alpha=0.5)
        ax2.set_ylabel("배터리 SoC (%)")
        ax2.set_title("배터리 충전 상태")
        ax2.set_xlabel("총 경과 시간 (h)")
        ax2.set_ylim(0, 105)
        for d in range(1,8):
            ax2.axvline(d*DRIVE_HRS, color='lightgray', lw=0.6)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("전략별 결과 요약")
    rows = []
    for name, (df_s, fin, finfo) in results.items():
        pos = float(df_s['pos_km'].iloc[-1])
        bat = float(df_s['battery_pct'].iloc[-1])
        rows.append({
            "전략": str(name),
            "최종 위치 (km)": f"{pos:.0f}",
            "완주": "✅" if fin else "❌",
            "완주 시간": f"{finfo['day']}일차 {finfo['hour']:.0f}시 ({finfo['elapsed_h']:.0f}h)" if fin else f"{pos:.0f}km 중단",
            "최종 배터리": f"{bat:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── 탭 2 ────────────────────────────────────────────────────────
with tab2:
    st.subheader("시간별 주행 플래너 (논문 1 Hourly Planner)")
    df_plan, fin_plan, finfo_plan = simulate(custom, params, ghi_vals)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("최종 위치", f"{df_plan['pos_km'].iloc[-1]:.0f} km")
    c2.metric("완주 여부", "✅ 완주" if fin_plan else "❌ 미완주")
    if fin_plan:
        c3.metric("완주 시간", f"{finfo_plan['day']}일차 {finfo_plan['hour']:.0f}시")
        c4.metric("총 소요", f"{finfo_plan['elapsed_h']:.0f}시간")
    else:
        c3.metric("최종 배터리", f"{df_plan['battery_pct'].iloc[-1]:.1f}%")
        c4.metric("부족 거리", f"{TOTAL_KM - df_plan['pos_km'].iloc[-1]:.0f} km")

    sel_day = st.selectbox("확인할 날짜", list(range(1,9)), format_func=lambda x: f"Day {x}")
    df_day  = df_plan[df_plan['day'] == sel_day]

    if len(df_day) > 0:
        fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        hours_day = df_day['hour']

        ax = axes2[0]
        ax.bar(hours_day, df_day['P_solar_w']/1000, color='#F4B942', alpha=0.8, width=0.8, label='태양광 발전 (kW)')
        ax.bar(hours_day, -df_day['P_use_w']/1000,  color='#4C72B0', alpha=0.7, width=0.8, label='소비 전력 (kW)')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel("전력 (kW)")
        ax.set_title(f"Day {sel_day} 시간별 전력 수지")
        ax.legend(fontsize=9)

        ax2b = axes2[1]
        ax2b.fill_between(hours_day, df_day['battery_pct'], alpha=0.35, color='#55A868')
        ax2b.plot(hours_day, df_day['battery_pct'], color='#2E7D32', lw=1.8)
        ax2b.axhline(20, color='red',   ls=':', alpha=0.5)
        ax2b.axhline(80, color='green', ls=':', alpha=0.5)
        ax2b.set_ylabel("배터리 SoC (%)")
        ax2b.set_ylim(0, 105)

        ax3b = axes2[2]
        stopped_mask = df_day['stopped']
        ax3b.bar(hours_day[~stopped_mask], df_day['v_kmh'][~stopped_mask], color='#C44E52', alpha=0.8, width=0.8, label='주행')
        ax3b.bar(hours_day[stopped_mask],  df_day['v_kmh'][stopped_mask],  color='#AAAAAA', alpha=0.5, width=0.8, label='정차')
        ax3b.set_ylabel("속도 (km/h)")
        ax3b.set_xlabel("시각 (시)")
        ax3b.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.subheader(f"Day {sel_day} 시간별 상세")
        tbl = df_day[['hour','pos_km','v_kmh','P_solar_w','P_use_w','battery_pct']].copy()
        tbl = tbl.round(1).reset_index(drop=True)
        tbl.columns = ['시각','위치(km)','속도(km/h)','발전(W)','소비(W)','배터리(%)']
        tbl['시각'] = tbl['시각'].apply(lambda h: f"{int(h):02d}:00")
        tbl['상태'] = ['⏸ 정차' if s else '▶ 주행' for s in df_day['stopped'].values]
        st.dataframe(tbl, use_container_width=True, hide_index=True)

# ── 탭 3 ────────────────────────────────────────────────────────
with tab3:
    st.subheader("BWSC 경로 지도")
    df_plan2, _, _ = simulate(custom, params, ghi_vals)
    final_km = float(df_plan2['pos_km'].iloc[-1])

    m = folium.Map(location=[-23.0, 134.0], zoom_start=5, tiles='CartoDB positron')
    cp_coords = [(lat, lon) for _, _, lat, lon in CHECKPOINTS]
    folium.PolyLine(cp_coords, color='#DD8452', weight=3, opacity=0.8).add_to(m)

    ratio = final_km / TOTAL_KM
    lat_cur = CHECKPOINTS[0][2] + ratio*(CHECKPOINTS[-1][2]-CHECKPOINTS[0][2])
    lon_cur = CHECKPOINTS[0][3] + ratio*(CHECKPOINTS[-1][3]-CHECKPOINTS[0][3])

    for km, name, lat, lon in CHECKPOINTS:
        color = 'red' if km == 0 or km == TOTAL_KM else 'blue'
        folium.Marker([lat, lon], popup=f"{name} ({km}km)", tooltip=name,
            icon=folium.Icon(color=color, icon='flag' if km in [0,TOTAL_KM] else 'info-sign', prefix='glyphicon')
        ).add_to(m)
    folium.Marker([lat_cur, lon_cur], popup=f"내 설정 최종: {final_km:.0f}km", tooltip="내 설정 최종 위치",
        icon=folium.Icon(color='green', icon='star', prefix='glyphicon')).add_to(m)
    st_folium(m, width=860, height=420)

    st.subheader("고도 프로필")
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    x_km = np.arange(TOTAL_KM)
    ax3.fill_between(x_km, ELEVATION, alpha=0.3, color='#8B7355')
    ax3.plot(x_km, ELEVATION, color='#5C4A32', lw=0.8)
    for km, name, _, _ in CHECKPOINTS:
        if 0 < km < TOTAL_KM:
            idx = min(int(km), TOTAL_KM-1)
            ax3.axvline(km, color='red', lw=0.8, alpha=0.5)
            ax3.text(km+10, float(ELEVATION[idx])+20, name.split()[0], fontsize=7, rotation=45, color='darkred')
    ax3.axvline(final_km, color='green', lw=1.5, ls='--', label=f'최종: {final_km:.0f}km')
    ax3.set_xlabel("누적 거리 (km)")
    ax3.set_ylabel("고도 (m)")
    ax3.set_title("Darwin to Adelaide 경로 고도")
    ax3.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.subheader("일별 GHI 예측 및 신뢰구간 (논문 3, +-1sigma = 68%)")
    days_x = np.arange(1,9)
    fig4, ax4 = plt.subplots(figsize=(10, 3.5))
    ax4.fill_between(days_x, GHI_MEAN-CI_Z*GHI_STD, GHI_MEAN+CI_Z*GHI_STD,
                     alpha=0.25, color='#4C72B0', label='68% 신뢰구간 (논문 3)')
    ax4.plot(days_x, GHI_MEAN, 'o-', color='#4C72B0', lw=2, ms=6, label='GHI 평균')
    ax4.plot(days_x, GHI_MEAN+CI_Z*GHI_STD, '--', color='#55A868', lw=1, alpha=0.7, label='낙관 (+1sigma)')
    ax4.plot(days_x, GHI_MEAN-CI_Z*GHI_STD, '--', color='#DD8452', lw=1, alpha=0.7, label='비관 (-1sigma)')
    ax4.plot(days_x, ghi_vals, 'o', color='purple', ms=5, label=f'현재: {scenario}')
    ax4.set_xlabel("레이스 일차")
    ax4.set_ylabel("GHI (kWh/m2/day)")
    ax4.set_title("일별 GHI 예측 및 68% 신뢰구간 (논문 3)")
    ax4.legend(fontsize=8)
    ax4.set_xticks(days_x)
    ax4.set_xticklabels([f"Day {d}" for d in days_x])
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

# ── 탭 4 ────────────────────────────────────────────────────────
with tab4:
    st.subheader("에너지 수지 상세 — 내 설정 전략")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("최종 위치", f"{df_plan['pos_km'].iloc[-1]:.0f} km")
    c2.metric("최종 배터리", f"{df_plan['battery_pct'].iloc[-1]:.1f}%")
    c3.metric("총 태양광 발전", f"{df_plan['P_solar_w'].sum()/1000:.1f} kWh")
    c4.metric("총 소비 전력", f"{df_plan['P_use_w'].sum()/1000:.1f} kWh")

    fig5, axes5 = plt.subplots(3,1, figsize=(10,9), sharex=True)
    hrs = df_plan['elapsed_h']

    axes5[0].bar(hrs, df_plan['P_solar_w']/1000, color='#F4B942', alpha=0.8, width=0.9, label='태양광 발전 (kW)')
    axes5[0].bar(hrs, -df_plan['P_use_w']/1000,  color='#4C72B0', alpha=0.7, width=0.9, label='소비 전력 (kW)')
    axes5[0].axhline(0, color='black', lw=0.5)
    axes5[0].set_ylabel("전력 (kW)")
    axes5[0].set_title("시간별 전력 수지")
    axes5[0].legend(fontsize=9)

    axes5[1].fill_between(hrs, df_plan['battery_pct'], alpha=0.35, color='#55A868')
    axes5[1].plot(hrs, df_plan['battery_pct'], color='#2E7D32', lw=1.5)
    axes5[1].axhline(20, color='red',   ls=':', alpha=0.5)
    axes5[1].axhline(80, color='green', ls=':', alpha=0.5)
    axes5[1].set_ylabel("배터리 SoC (%)")
    axes5[1].set_title("배터리 충전 상태")
    axes5[1].set_ylim(0,105)

    axes5[2].plot(hrs, df_plan['v_kmh'], color='#C44E52', lw=1.5)
    axes5[2].set_ylabel("속도 (km/h)")
    axes5[2].set_xlabel("총 경과 시간 (h)")
    axes5[2].set_title("시간별 속도")

    for ax_i in axes5:
        for d in range(1,8):
            ax_i.axvline(d*DRIVE_HRS, color='lightgray', lw=0.6)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

    st.subheader("일별 주행 요약")
    daily_rows = []
    for d in range(1,9):
        dd = df_plan[df_plan['day']==d]
        if len(dd)==0:
            daily_rows.append({'일차':str(d),'주행거리(km)':'0','평균속도(km/h)':'0',
                               '발전(kWh)':'0','소비(kWh)':'0','배터리(%)':'-','GHI':'-','상태':'미진입'})
        else:
            moving = dd[dd['v_kmh']>0]
            state  = '완주' if float(dd['pos_km'].iloc[-1])>=TOTAL_KM else ('정차 포함' if dd['stopped'].any() else '주행')
            daily_rows.append({
                '일차':        str(d),
                '주행거리(km)': f"{float(dd['v_kmh'].sum()):.0f}",
                '평균속도(km/h)': f"{float(moving['v_kmh'].mean()):.1f}" if len(moving)>0 else '0',
                '발전(kWh)':   f"{float(dd['P_solar_w'].sum()/1000):.2f}",
                '소비(kWh)':   f"{float(dd['P_use_w'].sum()/1000):.2f}",
                '배터리(%)':   f"{float(dd['battery_pct'].iloc[-1]):.1f}",
                'GHI':         f"{float(dd['ghi'].iloc[0]):.1f}",
                '상태':        state,
            })
    st.dataframe(pd.DataFrame(daily_rows), use_container_width=True, hide_index=True)

st.divider()
st.caption("논문 1: Kim et al., Seoul National Univ. | 논문 2: TopDutch ESGI170 | 논문 3: Oosthuizen et al., TUT | 논문 4: Sun Chaser II, TUT")
