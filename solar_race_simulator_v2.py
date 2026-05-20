"""
태양광 레이스 시뮬레이터 v2
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

# ── 폰트 설정 (macOS AppleGothic 우선) ──────────────────────────
def set_font():
    matplotlib.rcParams['font.family'] = 'AppleGothic' 

set_font()

# ════════════════════════════════════════════════════════════════
# 1. 상수 및 기본 파라미터
# ════════════════════════════════════════════════════════════════
DEFAULT_PARAMS = dict(
    mass_kg=180.0, panel_area_m2=4.0, panel_eff=0.24,
    battery_kwh=5.0, Cd=0.13, Cr=0.003, frontal_area=0.9,
    eta=0.90, aux_power_w=50.0, air_density=1.18, g=9.81,
)

# 논문 2: BWSC 체크포인트 (누적 km, 이름, 위도, 경도)
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
TOTAL_KM   = 3022
DRIVE_HRS  = 9.0   # 08:00~17:00
CHECKPOINT_STOP_MIN = 30  # 체크포인트 정차 시간(분) — 논문 2

# 논문 2: 일별 GHI 평균 및 논문 3: 1σ 신뢰구간
GHI_MEAN = np.array([5.5, 6.0, 5.8, 6.0, 5.7, 5.0, 5.5, 5.5])
GHI_STD  = GHI_MEAN * 0.15
CI_Z     = 1.0   # ±1σ = 68% (논문 3)
WIND_EFF = np.array([1.01, 1.02, 0.99, 1.005, 1.03, 1.01, 1.02, 1.00])

# 고도 프로필 생성
np.random.seed(42)
_x = np.linspace(0, TOTAL_KM, TOTAL_KM)
_base = 200*np.sin(_x/600) + 100*np.sin(_x/200)
_noise = np.cumsum(np.random.randn(TOTAL_KM)*0.8)
ELEVATION = np.clip(_base + (_noise - _noise.mean())*0.4, -50, 600)

# ════════════════════════════════════════════════════════════════
# 2. 물리 모델 (논문 4)
# ════════════════════════════════════════════════════════════════
def power_w(v_kmh, slope_deg, p, wind_f=1.0):
    """소비 전력 (W) — 논문 4 수식"""
    if v_kmh < 0.1:
        return 0.0
    v = v_kmh / 3.6
    theta = np.radians(slope_deg)
    F1 = 0.5 * p['air_density'] * p['frontal_area'] * p['Cd'] * v**2
    F2 = p['Cr'] * p['mass_kg'] * p['g'] * np.cos(theta)
    F3 = p['mass_kg'] * p['g'] * np.sin(theta)
    return max(0.0, (v/p['eta'])*(F1+F2+F3)*wind_f + p['aux_power_w'])

def solar_w(ghi, area, eff, hour):
    """시간대별 태양광 발전 (W)"""
    sunrise, sunset = 6.0, 18.0
    if hour <= sunrise or hour >= sunset:
        return 0.0
    angle = np.pi * (hour - sunrise) / (sunset - sunrise)
    peak  = ghi * 1000 / (np.pi/2) * (np.pi/(sunset-sunrise))
    return max(0.0, area * eff * peak * np.sin(angle))

# ════════════════════════════════════════════════════════════════
# 3. 빔서치 기반 일별 최적 속도 (논문 1)
# ════════════════════════════════════════════════════════════════
def beam_search(params, ghi_vals):
    """
    각 날의 태양광 수입 = 주행 소비가 되는 속도를 찾음
    배터리는 하루 시작/끝 동일 유지 목표
    """
    speeds = []
    for day in range(8):
        ghi  = max(0.5, ghi_vals[day])
        wf   = WIND_EFF[day]
        # 하루 총 태양광 발전량
        total_solar = sum(
            solar_w(ghi, params['panel_area_m2'], params['panel_eff'], 8.5+h) / 1000.0
            for h in range(int(DRIVE_HRS))
        )
        # 에너지 수지 0 속도 탐색
        def balance(v):
            return abs(power_w(v, 0.0, params, wf)/1000.0 * DRIVE_HRS - total_solar)
        res = minimize_scalar(balance, bounds=(40, 120), method='bounded')
        speeds.append(round(float(np.clip(res.x, 40, 120)), 1))
    return speeds

# ════════════════════════════════════════════════════════════════
# 4. 시뮬레이션 엔진 (논문 1 방식 — 시간별)
# ════════════════════════════════════════════════════════════════
def simulate(speed_per_day, params, ghi_vals):
    """
    논문 1 방식:
    - 목표 속도 유지, 배터리가 부족하면 그 시간은 달리지 않음(정차)
    - 체크포인트 30분 정차 + 충전효율 20% 향상 (논문 2)
    - 시간별 기록 반환
    """
    records = []
    battery  = params['battery_kwh']
    pos_km   = 0.0
    finished = False
    finish_info = None

    # 다음 체크포인트 인덱스
    cp_idx = 1  # 0번은 출발점

    total_elapsed_h = 0.0  # 레이스 총 경과 시간

    for day in range(8):
        if finished:
            break
        v_target = float(speed_per_day[day]) if day < len(speed_per_day) else float(speed_per_day[-1])
        ghi_today = max(0.5, ghi_vals[day] if day < len(ghi_vals) else ghi_vals[-1])
        wf = WIND_EFF[day] if day < len(WIND_EFF) else 1.0

        for h_idx in range(int(DRIVE_HRS)):
            hour = 8.0 + h_idx
            Ps   = solar_w(ghi_today, params['panel_area_m2'], params['panel_eff'], hour+0.5)

            # 목표 속도로 달릴 때 소비 전력
            si   = int(np.clip(pos_km, 0, TOTAL_KM-2))
            ei   = int(np.clip(pos_km + v_target, 0, TOTAL_KM-1))
            dh   = ELEVATION[ei] - ELEVATION[si] if ei > si else 0.0
            slp  = np.degrees(np.arctan(dh / (v_target*1000 + 1e-6)))
            Pu   = power_w(v_target, slp, params, wf)

            net  = (Ps - Pu) / 1000.0  # kWh

            # 배터리 여유 확인: 부족하면 정차(태양광만 충전)
            if battery + net < 0:
                # 달리면 방전 → 정차하고 충전
                battery = np.clip(battery + Ps/1000.0, 0, params['battery_kwh'])
                v_actual = 0.0
                dist     = 0.0
                Pu_actual = 0.0
            else:
                battery   = np.clip(battery + net, 0, params['battery_kwh'])
                v_actual  = v_target
                dist      = v_target
                Pu_actual = Pu

            new_pos = min(pos_km + dist, TOTAL_KM)
            pos_km  = new_pos
            total_elapsed_h += 1.0

            records.append({
                'day':        day+1,
                'hour':       hour,
                'elapsed_h':  total_elapsed_h,
                'pos_km':     pos_km,
                'battery_pct': battery/params['battery_kwh']*100,
                'v_kmh':      v_actual,
                'P_solar_w':  Ps,
                'P_use_w':    Pu_actual,
                'net_kwh':    net if v_actual>0 else Ps/1000.0,
                'slope_deg':  slp,
                'ghi':        ghi_today,
                'stopped':    v_actual < 0.1,
            })

            # 완주 확인
            if pos_km >= TOTAL_KM:
                finished   = True
                finish_info = {'day': day+1, 'hour': hour, 'elapsed_h': total_elapsed_h}
                break

    df = pd.DataFrame(records)
    return df, finished, finish_info

# ════════════════════════════════════════════════════════════════
# 5. Streamlit UI
# ════════════════════════════════════════════════════════════════
st.set_page_config(page_title="태양광 레이스 시뮬레이터", page_icon="☀️", layout="wide")
st.title("☀️ 태양광 레이스 시뮬레이터")
st.caption("논문 1 (서울대) · 논문 2 (BWSC) · 논문 3 (GHI 신뢰구간) · 논문 4 (Sun Chaser II) 통합 재현 v2")

# ── 사이드바 ────────────────────────────────────────────────────
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
    params['frontal_area']  = DEFAULT_PARAMS['frontal_area']
    params['aux_power_w']   = DEFAULT_PARAMS['aux_power_w']
    params['air_density']   = DEFAULT_PARAMS['air_density']
    params['g']             = DEFAULT_PARAMS['g']

    st.divider()
    st.header("🌤 날씨 시나리오 (논문 3)")
    scenario = st.radio("GHI 시나리오", ["mean","optimistic","pessimistic"],
        format_func=lambda x: {"mean":"평균","optimistic":"낙관 (+1σ)","pessimistic":"비관 (-1σ)"}[x])

# GHI 값 결정
if scenario == "mean":
    ghi_vals = GHI_MEAN
elif scenario == "optimistic":
    ghi_vals = GHI_MEAN + CI_Z * GHI_STD
else:
    ghi_vals = GHI_MEAN - CI_Z * GHI_STD

# 빔서치 추천
beam_spd = beam_search(params, ghi_vals)

# ── 탭 ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 전략 비교", "⏱ 시간별 플래너", "🗺 경로 분석", "⚡ 에너지 상세"])

# ═══════════════════════════════════════════════════════════════
# 탭 1: 전략 비교
# ═══════════════════════════════════════════════════════════════
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
        "최소속도 (60 km/h)":  [60.0]*8,
        "최대속도 (100 km/h)": [100.0]*8,
        "평균속도 (80 km/h)":  [80.0]*8,
        "빔서치 최적":          beam_spd,
        "내 설정":             custom,
    }
    colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"]

    results = {}
    for name, spd in strategies.items():
        df_s, fin, finfo = simulate(spd, params, ghi_vals)
        results[name] = (df_s, fin, finfo)

    with col_right:
        fig, axes = plt.subplots(2, 1, figsize=(9, 7))

        # 누적 주행거리
        ax1 = axes[0]
        for (name, (df_s, fin, finfo)), color in zip(results.items(), colors):
            ax1.plot(df_s['elapsed_h'], df_s['pos_km'], label=name, color=color, lw=1.8)
        ax1.axhline(TOTAL_KM, color='gray', ls='--', lw=0.8, alpha=0.6)
        ax1.set_ylabel("누적 주행거리 (km)")
        ax1.set_title("누적 주행거리")
        ax1.legend(fontsize=8, loc='lower right')
        for d in range(1,8):
            ax1.axvline(d*DRIVE_HRS, color='lightgray', lw=0.6)

        # 배터리 SoC
        ax2 = axes[1]
        for (name, (df_s, fin, finfo)), color in zip(results.items(), colors):
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

    # 요약 테이블
    st.subheader("전략별 결과 요약")
    rows = []
    for name, (df_s, fin, finfo) in results.items():
        pos  = df_s['pos_km'].iloc[-1]
        bat  = df_s['battery_pct'].iloc[-1]
        elap = finfo['elapsed_h'] if fin else df_s['elapsed_h'].iloc[-1]
        rows.append({
            "전략": name,
            "최종 위치 (km)": f"{pos:.0f}",
            "완주": "✅" if fin else "❌",
            "완주 시간": f"{finfo['day']}일차 {finfo['hour']:.0f}시 ({finfo['elapsed_h']:.0f}h)" if fin else f"{pos:.0f}km에서 중단",
            "최종 배터리": f"{bat:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# 탭 2: 시간별 플래너 (논문 1 hourly planner)
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("시간별 주행 플래너 (논문 1 Hourly Planner)")
    st.caption("논문 1은 1시간 단위로 배터리·속도·위치를 업데이트하는 플래너를 제안했어요.")

    df_plan, fin_plan, finfo_plan = simulate(custom, params, ghi_vals)

    # 상단 요약 지표
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("최종 위치", f"{df_plan['pos_km'].iloc[-1]:.0f} km")
    c2.metric("완주 여부", "✅ 완주" if fin_plan else "❌ 미완주")
    if fin_plan:
        c3.metric("완주 시간", f"{finfo_plan['day']}일차 {finfo_plan['hour']:.0f}시")
        c4.metric("총 소요", f"{finfo_plan['elapsed_h']:.0f}시간")
    else:
        c3.metric("최종 배터리", f"{df_plan['battery_pct'].iloc[-1]:.1f}%")
        c4.metric("부족 거리", f"{TOTAL_KM - df_plan['pos_km'].iloc[-1]:.0f} km")

    # 일별 선택
    sel_day = st.selectbox("확인할 날짜", options=list(range(1,9)), format_func=lambda x: f"Day {x}")
    df_day  = df_plan[df_plan['day'] == sel_day]

    if len(df_day) == 0:
        st.info("해당 날짜 데이터가 없어요.")
    else:
        fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        hours_day = df_day['hour']

        # 전력 수지
        ax = axes2[0]
        ax.bar(hours_day, df_day['P_solar_w']/1000, color='#F4B942', alpha=0.8, width=0.8, label='태양광 발전 (kW)')
        ax.bar(hours_day, -df_day['P_use_w']/1000,  color='#4C72B0', alpha=0.7, width=0.8, label='소비 전력 (kW)')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel("전력 (kW)")
        ax.set_title(f"Day {sel_day} — 시간별 전력 수지")
        ax.legend(fontsize=9)

        # 배터리
        ax2b = axes2[1]
        ax2b.fill_between(hours_day, df_day['battery_pct'], alpha=0.35, color='#55A868')
        ax2b.plot(hours_day, df_day['battery_pct'], color='#2E7D32', lw=1.8)
        ax2b.axhline(20, color='red',   ls=':', alpha=0.5, lw=1)
        ax2b.axhline(80, color='green', ls=':', alpha=0.5, lw=1)
        ax2b.set_ylabel("배터리 SoC (%)")
        ax2b.set_title("배터리 충전 상태")
        ax2b.set_ylim(0,105)

        # 속도
        ax3b = axes2[2]
        stopped = df_day['stopped']
        ax3b.bar(hours_day[~stopped],  df_day['v_kmh'][~stopped],  color='#C44E52', alpha=0.8, width=0.8, label='주행')
        ax3b.bar(hours_day[stopped],   df_day['v_kmh'][stopped],   color='#AAAAAA', alpha=0.5, width=0.8, label='정차(충전)')
        ax3b.set_ylabel("속도 (km/h)")
        ax3b.set_xlabel("시각 (시)")
        ax3b.set_title("시간별 속도")
        ax3b.legend(fontsize=9)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # 시간별 테이블
        st.subheader(f"Day {sel_day} 시간별 상세")
        tbl = df_day[['hour','pos_km','v_kmh','P_solar_w','P_use_w','battery_pct','slope_deg']].copy()
        tbl.columns = ['시각','위치(km)','속도(km/h)','발전(W)','소비(W)','배터리(%)','경사(°)']
        tbl = tbl.round(1).reset_index(drop=True)
        tbl['시각'] = tbl['시각'].apply(lambda h: f"{int(h):02d}:00")
        tbl['정차'] = df_day['stopped'].values
        tbl['정차'] = tbl['정차'].map({True:'⏸ 정차', False:'▶ 주행'})
        st.dataframe(tbl, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# 탭 3: 경로 분석 (지도 + 고도 + GHI)
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("BWSC 경로 지도")

    # folium 지도
    m = folium.Map(location=[-23.0, 134.0], zoom_start=5, tiles='CartoDB positron')

    # 경로선 (체크포인트 연결)
    cp_coords = [(lat, lon) for _, _, lat, lon in CHECKPOINTS]
    folium.PolyLine(cp_coords, color='#DD8452', weight=3, opacity=0.8).add_to(m)

    # 내 설정 전략의 현재 위치 (완주 or 최종 위치)
    final_km = df_plan['pos_km'].iloc[-1]
    # 비율로 위경도 근사
    ratio = final_km / TOTAL_KM
    lat_cur = CHECKPOINTS[0][2] + ratio*(CHECKPOINTS[-1][2]-CHECKPOINTS[0][2])
    lon_cur = CHECKPOINTS[0][3] + ratio*(CHECKPOINTS[-1][3]-CHECKPOINTS[0][3])

    # 체크포인트 마커
    for km, name, lat, lon in CHECKPOINTS:
        color = 'red' if km == 0 or km == TOTAL_KM else 'blue'
        icon  = 'flag' if km == 0 or km == TOTAL_KM else 'info-sign'
        folium.Marker(
            [lat, lon],
            popup=f"{name} ({km}km)",
            tooltip=name,
            icon=folium.Icon(color=color, icon=icon, prefix='glyphicon')
        ).add_to(m)

    # 현재 위치 마커
    folium.Marker(
        [lat_cur, lon_cur],
        popup=f"내 설정 최종 위치: {final_km:.0f}km",
        tooltip="내 설정 최종 위치",
        icon=folium.Icon(color='green', icon='car', prefix='glyphicon')
    ).add_to(m)

    st_folium(m, width=860, height=420)

    # 고도 프로필
    st.subheader("고도 프로필 (논문 2 기반 근사)")
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    x_km = np.arange(TOTAL_KM)
    ax3.fill_between(x_km, ELEVATION, alpha=0.3, color='#8B7355')
    ax3.plot(x_km, ELEVATION, color='#5C4A32', lw=0.8)
    for km, name, _, _ in CHECKPOINTS:
        if 0 < km < TOTAL_KM:
            idx = min(int(km), TOTAL_KM-1)
            ax3.axvline(km, color='red', lw=0.8, alpha=0.5)
            ax3.text(km+10, ELEVATION[idx]+20, name.split()[0], fontsize=7, rotation=45, color='darkred')
    ax3.axvline(final_km, color='green', lw=1.5, ls='--', label=f'내 설정 최종: {final_km:.0f}km')
    ax3.set_xlabel("누적 거리 (km)")
    ax3.set_ylabel("고도 (m)")
    ax3.set_title("Darwin → Adelaide 경로 고도")
    ax3.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # GHI 신뢰구간
    st.subheader("일별 GHI 예측 및 신뢰구간 (논문 3, ±1σ = 68%)")
    days_x = np.arange(1,9)
    fig4, ax4 = plt.subplots(figsize=(10, 3.5))
    ax4.fill_between(days_x, GHI_MEAN-CI_Z*GHI_STD, GHI_MEAN+CI_Z*GHI_STD,
                     alpha=0.25, color='#4C72B0', label='68% 신뢰구간 (+-1sigma)')
    ax4.plot(days_x, GHI_MEAN, 'o-', color='#4C72B0', lw=2, ms=6, label='GHI 평균')
    ax4.plot(days_x, GHI_MEAN+CI_Z*GHI_STD, '--', color='#55A868', lw=1, alpha=0.7, label='낙관 (+1sigma)')
    ax4.plot(days_x, GHI_MEAN-CI_Z*GHI_STD, '--', color='#DD8452', lw=1, alpha=0.7, label='비관 (-1sigma)')
    ax4.plot(days_x, ghi_vals, 'o', color='purple', ms=5, label=f'현재 시나리오: {scenario}')
    ax4.set_xlabel("레이스 일차")
    ax4.set_ylabel("GHI (kWh/m2/day)")
    ax4.set_title("일별 GHI 예측 및 68% 신뢰구간 (논문 3)")
    ax4.legend(fontsize=8)
    ax4.set_xticks(days_x)
    ax4.set_xticklabels([f"Day {d}" for d in days_x])
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

# ═══════════════════════════════════════════════════════════════
# 탭 4: 에너지 상세
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("에너지 수지 상세 — 내 설정 전략")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("최종 위치", f"{df_plan['pos_km'].iloc[-1]:.0f} km")
    c2.metric("최종 배터리", f"{df_plan['battery_pct'].iloc[-1]:.1f}%")
    c3.metric("총 태양광 발전", f"{df_plan['P_solar_w'].sum()/1000:.1f} kWh")
    c4.metric("총 소비 전력", f"{df_plan['P_use_w'].sum()/1000:.1f} kWh")

    fig5, axes5 = plt.subplots(3,1, figsize=(10,9), sharex=True)

    hrs = df_plan['elapsed_h']

    ax = axes5[0]
    ax.bar(hrs, df_plan['P_solar_w']/1000, color='#F4B942', alpha=0.8, width=0.9, label='태양광 발전 (kW)')
    ax.bar(hrs, -df_plan['P_use_w']/1000,  color='#4C72B0', alpha=0.7, width=0.9, label='소비 전력 (kW)')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_ylabel("전력 (kW)")
    ax.set_title("시간별 전력 수지")
    ax.legend(fontsize=9)

    ax2e = axes5[1]
    ax2e.fill_between(hrs, df_plan['battery_pct'], alpha=0.35, color='#55A868')
    ax2e.plot(hrs, df_plan['battery_pct'], color='#2E7D32', lw=1.5)
    ax2e.axhline(20, color='red',   ls=':', alpha=0.5)
    ax2e.axhline(80, color='green', ls=':', alpha=0.5)
    ax2e.set_ylabel("배터리 SoC (%)")
    ax2e.set_title("배터리 충전 상태")
    ax2e.set_ylim(0,105)

    ax3e = axes5[2]
    ax3e.plot(hrs, df_plan['v_kmh'], color='#C44E52', lw=1.5, label='속도 (km/h)')
    ax3e.set_ylabel("속도 (km/h)")
    ax3e.set_xlabel("총 경과 시간 (h)")
    ax3e.set_title("시간별 속도")

    for ax_i in axes5:
        for d in range(1,8):
            ax_i.axvline(d*DRIVE_HRS, color='lightgray', lw=0.6)

    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

    # 일별 요약 테이블
    st.subheader("일별 주행 요약")
    daily_rows = []
    for d in range(1,9):
        dd = df_plan[df_plan['day']==d]
        if len(dd)==0:
            daily_rows.append({'일차':d,'주행거리(km)':0,'평균속도(km/h)':0,
                                '발전(kWh)':0,'소비(kWh)':0,'배터리(%)':'-','GHI':'-','상태':'미진입'})
        else:
            moving = dd[dd['v_kmh']>0]
            dist   = dd['pos_km'].iloc[-1] - dd['pos_km'].iloc[0] + (moving['v_kmh'].iloc[0] if len(moving)>0 else 0)
            state  = '완주' if dd['pos_km'].iloc[-1]>=TOTAL_KM else ('정차 포함' if dd['stopped'].any() else '주행')
            daily_rows.append({
                '일차': d,
                '주행거리(km)': round(dd['v_kmh'].sum(), 0),
                '평균속도(km/h)': round(moving['v_kmh'].mean(), 1) if len(moving)>0 else 0,
                '발전(kWh)': round(dd['P_solar_w'].sum()/1000, 2),
                '소비(kWh)': round(dd['P_use_w'].sum()/1000, 2),
                '배터리(%)': round(dd['battery_pct'].iloc[-1], 1),
                'GHI': round(dd['ghi'].iloc[0], 1),
                '상태': state,
            })
    st.dataframe(pd.DataFrame(daily_rows), use_container_width=True, hide_index=True)

st.divider()
st.caption("논문 1: Kim et al., Seoul National Univ. | 논문 2: TopDutch ESGI170 | 논문 3: Oosthuizen et al., TUT | 논문 4: Sun Chaser II, TUT")
