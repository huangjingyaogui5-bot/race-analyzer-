"""
analysis_engine.py  ─  全解析機能 Web対応エンジン
==================================================
matplotlib を Agg バックエンドで使用し、
PNG 画像を base64 で返す。
GUI依存（TkAgg, plt.show, messagebox）は完全除去。

公開関数:
  compute_chart(chart_id, df, lap_data, lap_times, best_lap, selected_laps) -> dict
  compute_ai(ai_id, df, lap_data, lap_times, best_lap) -> dict
  generate_pdf_bytes(df, lap_data, lap_times, best_lap, selected_laps) -> bytes
"""

import io, base64, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # ← GUI不要のバックエンドに固定
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import pickle, os
warnings.filterwarnings("ignore")

# ============================================================
# カラー定数（元コードと同一）
# ============================================================
COLORS_LAP = [
    '#FF4E4E','#4E9FFF','#4EFF9F','#FFD24E','#FF4EE0',
    '#4EFFF0','#FF884E','#A44EFF','#B8FF4E','#FF4E88',
    '#4EC8FF','#FFEE4E','#FF6B4E','#4EFFB8','#884EFF',
    '#FF4EC0','#4EFFD4','#FFB84E','#4E7FFF','#D4FF4E',
]
COLOR_BEST    = '#00E5FF'
COLOR_WARN    = '#FF6B35'
COLOR_GOOD    = '#39FF8A'
COLOR_NEUTRAL = '#8B9EC0'
BRAKE_THR     = -0.2
THROTTLE_THR  = 0.05
MIN_CORNER_PTS = 8

PLT_STYLE = {
    "figure.facecolor": "#0A0E1A", "axes.facecolor": "#0F1421",
    "axes.edgecolor": "#1E2840",   "axes.labelcolor": "#8B9EC0",
    "axes.titlecolor": "#E8EEF8",  "xtick.color": "#5A6A8A",
    "ytick.color": "#5A6A8A",      "grid.color": "#1A2035",
    "legend.facecolor": "#0F1421", "legend.edgecolor": "#1E2840",
    "legend.labelcolor": "#8B9EC0","text.color": "#E8EEF8",
    "savefig.facecolor": "#0A0E1A","font.family": "monospace",
}
plt.rcParams.update(PLT_STYLE)

# ============================================================
# ユーティリティ
# ============================================================

def fmt_time(sec: float) -> str:
    if sec is None: return "—"
    m = int(sec // 60); s = sec - m * 60
    return f"{m}:{s:06.3f}"

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, dpi=150, bbox_inches="tight", facecolor="#0A0E1A", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def _lap_dist(ld: pd.DataFrame) -> pd.DataFrame:
    ld = ld.copy()
    ld["ld"] = ld["dist_m"] - ld["dist_m"].min()
    return ld.sort_values("ld")

def _resample(ld: pd.DataFrame, pts: int = 1500) -> pd.DataFrame:
    ld = _lap_dist(ld)
    if len(ld) < 4: return ld
    nd = np.linspace(ld["ld"].min(), ld["ld"].max(), pts)
    out = {"dist": nd}
    for col in ["speed_kmh","long_g","lat_g","throttle","brake_pct","time_sec"]:
        if col in ld.columns:
            out[col] = np.interp(nd, ld["ld"].values, ld[col].fillna(0).values)
    t0 = ld["time_sec"].min()
    out["time"] = np.interp(nd, ld["ld"].values, (ld["time_sec"] - t0).values)
    return pd.DataFrame(out)

def _detect_corners(ld: pd.DataFrame):
    if "lat_g" not in ld.columns: return []
    ld = _lap_dist(ld)
    latg = ld["lat_g"].abs().rolling(5, min_periods=1).mean()
    corners = []; in_c = False; cs = 0
    for i in range(len(latg)):
        if latg.iloc[i] > 0.25 and not in_c: cs = i; in_c = True
        elif latg.iloc[i] < 0.20 and in_c:
            in_c = False
            sec = ld.iloc[cs:i]
            if len(sec) > MIN_CORNER_PTS:
                corners.append((float(ld["ld"].iloc[cs]), float(ld["ld"].iloc[min(i, len(ld)-1)])))
    return corners

def _theoretical_best(lap_data: dict, lap_times: dict, n: int = 10) -> float:
    total = 0.0
    for i in range(n):
        times = []
        for lap in lap_times:
            ld = _lap_dist(lap_data[lap])
            length = ld["ld"].max()
            if length == 0: continue
            sl = length / n
            sec = ld[(ld["ld"] >= i*sl) & (ld["ld"] < (i+1)*sl)]
            if len(sec) > 3:
                t = sec["time_sec"].max() - sec["time_sec"].min()
                if t > 0: times.append(t)
        if times: total += min(times)
    return total

def _color(lap, best_lap, idx):
    return COLOR_BEST if lap == best_lap else COLORS_LAP[idx % len(COLORS_LAP)]

def _track_bg(ax, df, lat_col="lat", lon_col="lon", alpha=0.35, lw=10):
    ax.plot(df[lon_col], df[lat_col], color="#0D1520", lw=lw, alpha=alpha, zorder=1)

# ============================================================
# チャート生成関数
# ============================================================

def _chart_speed_trace(df, lap_data, lap_times, best_lap, selected):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9),
                             gridspec_kw={"height_ratios":[3,1,1],"hspace":0.05})
    ax_spd, ax_lng, ax_lat = axes
    for idx, lap in enumerate(sorted(selected, key=lambda l: lap_times[l])):
        ld = _lap_dist(lap_data[lap])
        is_best = (lap == best_lap)
        c = _color(lap, best_lap, idx); lw = 2.2 if is_best else 0.9; al = 1.0 if is_best else 0.5
        label = f"Lap {lap}  {fmt_time(lap_times[lap])}" + ("  ★BEST" if is_best else "")
        ax_spd.plot(ld["ld"], ld["speed_kmh"], color=c, lw=lw, alpha=al, label=label, zorder=10 if is_best else 3)
        if "long_g" in ld: ax_lng.plot(ld["ld"], ld["long_g"], color=c, lw=lw*0.8, alpha=al)
        if "lat_g"  in ld: ax_lat.plot(ld["ld"], ld["lat_g"],  color=c, lw=lw*0.8, alpha=al)
    for ax in axes:
        ax.grid(True, color="#1A2035", lw=0.5, alpha=0.8); ax.spines[:].set_color("#1E2840")
        ax.tick_params(colors="#5A6A8A", labelsize=8)
    ax_spd.set_ylabel("Speed (km/h)", color="#8B9EC0", fontsize=9)
    ax_spd.set_title("SPEED TRACE", color="#E8EEF8", fontsize=11, fontweight="bold", pad=6)
    ax_spd.tick_params(labelbottom=False)
    ax_spd.legend(fontsize=7, loc="upper right", ncol=min(4, len(selected)), framealpha=0.7)
    ax_lng.axhline(0, color="#2A3A5A", lw=0.8); ax_lng.set_ylabel("Long G", color="#8B9EC0", fontsize=8); ax_lng.tick_params(labelbottom=False)
    ax_lat.axhline(0, color="#2A3A5A", lw=0.8); ax_lat.set_ylabel("Lat G",  color="#8B9EC0", fontsize=8)
    ax_lat.set_xlabel("Distance (m)", color="#8B9EC0", fontsize=9)
    return _fig_to_b64(fig)

def _chart_delta_time(df, lap_data, lap_times, best_lap, selected):
    if len(selected) < 2:
        fig, ax = plt.subplots(figsize=(13, 4)); ax.text(0.5, 0.5, "2 laps minimum required", transform=ax.transAxes, color="#8B9EC0", ha="center", va="center")
        return _fig_to_b64(fig)
    base = _resample(lap_data[best_lap])
    fig, ax = plt.subplots(figsize=(13, 5))
    for idx, lap in enumerate(sorted([l for l in selected if l != best_lap], key=lambda l: lap_times[l])):
        res = _resample(lap_data[lap])
        ml = min(len(base), len(res))
        delta = res["time"].values[:ml] - base["time"].values[:ml]
        dist_arr = res["dist"].values[:ml]
        diff = lap_times[lap] - lap_times[best_lap]
        c = COLORS_LAP[idx % len(COLORS_LAP)]
        ax.plot(dist_arr, delta, color=c, lw=1.5, label=f"Lap {lap}  ({'+' if diff>=0 else ''}{diff:.3f}s)", alpha=0.85)
        ax.fill_between(dist_arr, delta, 0, where=(delta>0), alpha=0.07, color="#FF4E4E")
        ax.fill_between(dist_arr, delta, 0, where=(delta<0), alpha=0.07, color="#4EFF9F")
    ax.axhline(0, color=COLOR_BEST, lw=1.5, ls="--", zorder=5)
    ax.set_title(f"DELTA TIME  (Base: Lap {best_lap} ★BEST  {fmt_time(lap_times[best_lap])})", color="#E8EEF8", fontsize=11, fontweight="bold")
    ax.set_xlabel("Distance (m)", color="#8B9EC0", fontsize=9)
    ax.set_ylabel("Time Delta (s)  (+ = slower)", color="#8B9EC0", fontsize=9)
    ax.grid(True, color="#1A2035", lw=0.5); ax.spines[:].set_color("#1E2840"); ax.tick_params(colors="#5A6A8A", labelsize=8)
    ax.legend(fontsize=8, framealpha=0.7)
    return _fig_to_b64(fig)

def _chart_gg(df, lap_data, lap_times, best_lap, selected):
    fig, ax = plt.subplots(figsize=(7, 7))
    for idx, lap in enumerate(sorted(selected, key=lambda l: lap_times[l])):
        ld = lap_data[lap]; is_best = (lap == best_lap)
        if "lat_g" not in ld.columns or "long_g" not in ld.columns: continue
        c = _color(lap, best_lap, idx)
        ax.scatter(ld["lat_g"], ld["long_g"], s=12 if is_best else 4, color=c,
                   alpha=0.9 if is_best else 0.25, label=f"Lap {lap}{'  ★BEST' if is_best else ''}", zorder=10 if is_best else 3, linewidths=0)
    for r in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        th = np.linspace(0, 2*np.pi, 300)
        ax.plot(r*np.cos(th), r*np.sin(th), color="#1E2840", lw=0.8, ls="--", alpha=0.8)
        ax.text(r*0.707+0.02, r*0.707+0.02, f"{r}g", color="#3A4A6A", fontsize=7)
    ax.axhline(0, color="#2A3A5A", lw=0.8); ax.axvline(0, color="#2A3A5A", lw=0.8)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3, 2)
    ax.set_xlabel("Lateral G  (←Left  Right→)", color="#8B9EC0", fontsize=9)
    ax.set_ylabel("Longitudinal G  (↓Brake  Accel↑)", color="#8B9EC0", fontsize=9)
    ax.set_title("G-G DIAGRAM", color="#E8EEF8", fontsize=11, fontweight="bold")
    ax.spines[:].set_color("#1E2840"); ax.tick_params(colors="#5A6A8A", labelsize=8)
    ax.legend(fontsize=8, framealpha=0.7)
    return _fig_to_b64(fig)

def _chart_brake_map(df, lap_data, lap_times, best_lap, selected):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax1, ax2 = axes
    for ax in axes: ax.set_facecolor("#080C14"); ax.axis("off")
    if "lat" in df.columns and "lon" in df.columns:
        _track_bg(ax1, df); _track_bg(ax2, df)
    for idx, lap in enumerate(selected):
        ld = lap_data[lap]
        if "brake" not in ld.columns or "lat" not in ld.columns: continue
        is_best = (lap == best_lap); c = _color(lap, best_lap, idx)
        brk = ld[ld["brake"] == True]
        ax1.scatter(brk["lon"], brk["lat"], s=20 if is_best else 10, color=c,
                    alpha=1.0 if is_best else 0.6, label=f"Lap {lap}{'  ★' if is_best else ''}", zorder=5+is_best, linewidths=0)
    ax1.set_aspect("equal"); ax1.set_title("BRAKE POINTS", color="#E8EEF8", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=7, facecolor="#0F1421", edgecolor="#1E2840")
    ld_b = lap_data[best_lap]
    if "long_g" in ld_b.columns and "lat" in ld_b.columns:
        bd = ld_b[ld_b["long_g"] < -0.05]
        if len(bd) > 0:
            sc = ax2.scatter(bd["lon"], bd["lat"], c=-bd["long_g"], cmap="RdYlBu_r", s=12,
                             alpha=0.9, zorder=5, vmin=0, vmax=2.5, linewidths=0)
            cb = plt.colorbar(sc, ax=ax2, pad=0.02, fraction=0.03)
            cb.set_label("Brake Intensity (G)", color="#8B9EC0", fontsize=8)
    ax2.set_aspect("equal"); ax2.set_title(f"BRAKE INTENSITY HEATMAP  Lap {best_lap} ★BEST", color="#E8EEF8", fontsize=10, fontweight="bold")
    plt.tight_layout()
    return _fig_to_b64(fig)

def _chart_corner_speed(df, lap_data, lap_times, best_lap, selected):
    ld_b = _lap_dist(lap_data[best_lap])
    corners = _detect_corners(ld_b)
    if not corners:
        fig, ax = plt.subplots(figsize=(12, 4)); ax.text(0.5, 0.5, "No corners detected (lat_g required)", transform=ax.transAxes, color="#8B9EC0", ha="center", va="center"); return _fig_to_b64(fig)
    data = []
    for c_num, (ds, de) in enumerate(corners, 1):
        sec = ld_b[(ld_b["ld"] >= ds) & (ld_b["ld"] <= de)]
        if len(sec) < 3: continue
        entry = float(sec["speed_kmh"].iloc[0]); minv = float(sec["speed_kmh"].min()); exitv = float(sec["speed_kmh"].iloc[-1])
        all_min = []
        for lap in lap_times:
            ldd = _lap_dist(lap_data[lap]); cd = ldd[(ldd["ld"] >= ds) & (ldd["ld"] <= de)]
            if len(cd) > 2: all_min.append(float(cd["speed_kmh"].min()))
        bm = max(all_min) if all_min else minv
        data.append({"num": c_num, "entry": entry, "min": minv, "exit": exitv, "best_min": bm, "loss": bm - minv})
    if not data:
        fig, ax = plt.subplots(figsize=(12, 4)); ax.text(0.5, 0.5, "No corner data", transform=ax.transAxes, color="#8B9EC0", ha="center", va="center"); return _fig_to_b64(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios":[2,1]})
    ax_bar, ax_rank = axes
    x = np.arange(len(data)); w = 0.26
    ax_bar.bar(x-w, [d["entry"] for d in data], w, label="Entry",   color="#4E9FFF", alpha=0.85, edgecolor="none")
    ax_bar.bar(x,   [d["min"]   for d in data], w, label="Minimum", color="#FF4E4E", alpha=0.85, edgecolor="none")
    ax_bar.bar(x+w, [d["exit"]  for d in data], w, label="Exit",    color="#4EFF9F", alpha=0.85, edgecolor="none")
    for i, d in enumerate(data):
        ax_bar.plot(i, d["best_min"], "_", color="#FFD700", ms=12, lw=2.5, zorder=10, label="Best Min" if i==0 else "")
        if d["loss"] > 1.0: ax_bar.annotate(f"-{d['loss']:.1f}", (i, d["min"]), textcoords="offset points", xytext=(0,-14), ha="center", fontsize=7, color=COLOR_WARN)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels([f"C{d['num']}" for d in data], color="#8B9EC0", fontsize=9)
    ax_bar.set_ylabel("Speed (km/h)", color="#8B9EC0", fontsize=9); ax_bar.set_title(f"CORNER SPEED ANALYSIS  Lap {best_lap} ★BEST", color="#E8EEF8", fontsize=10, fontweight="bold")
    ax_bar.grid(True, color="#1A2035", axis="y", lw=0.5); ax_bar.spines[:].set_color("#1E2840"); ax_bar.tick_params(colors="#5A6A8A", labelsize=8)
    ax_bar.legend(fontsize=8, facecolor="#0F1421", edgecolor="#1E2840")
    sorted_d = sorted(data, key=lambda d: -d["loss"])[:5]
    colors_r = [COLOR_WARN if d["loss"]>3 else (COLOR_NEUTRAL if d["loss"]>1 else COLOR_GOOD) for d in sorted_d]
    ax_rank.barh([f"C{d['num']}" for d in sorted_d], [d["loss"] for d in sorted_d], color=colors_r, alpha=0.85, edgecolor="none")
    ax_rank.set_xlabel("Speed Loss (km/h)", color="#8B9EC0", fontsize=8); ax_rank.set_title("MINIMUM SPEED LOSS\nRanking", color="#E8EEF8", fontsize=9, fontweight="bold")
    ax_rank.tick_params(colors="#5A6A8A", labelsize=8); ax_rank.grid(True, color="#1A2035", axis="x", lw=0.5); ax_rank.spines[:].set_color("#1E2840")
    plt.tight_layout()
    return _fig_to_b64(fig), [{"corner": d["num"], "entry": d["entry"], "min": d["min"], "exit": d["exit"], "best_min": d["best_min"], "loss": round(d["loss"],2)} for d in data]

def _chart_sector(df, lap_data, lap_times, best_lap, selected):
    n_sec = 6; sec_data = {i: {} for i in range(n_sec)}
    for si in range(n_sec):
        for lap in selected:
            ld = _lap_dist(lap_data[lap]); length = ld["ld"].max()
            if length == 0: continue
            sl = length / n_sec; sec = ld[(ld["ld"] >= si*sl) & (ld["ld"] < (si+1)*sl)]
            if len(sec) > 5: sec_data[si][lap] = sec["time_sec"].max() - sec["time_sec"].min()
    fig, ax = plt.subplots(figsize=(13, 5))
    laps_s = sorted(selected, key=lambda l: lap_times[l]); x = np.arange(n_sec)
    for li, lap in enumerate(laps_s):
        is_best = (lap == best_lap); c = _color(lap, best_lap, li)
        for si in range(n_sec):
            if lap not in sec_data[si]: continue
            t = sec_data[si][lap]; bt = min(sec_data[si].values()) if sec_data[si] else t
            offset = (li - len(laps_s)/2) * 0.055
            ax.bar(x[si]+offset, t, width=0.05, color=c if (t-bt)<0.1 else COLOR_WARN, alpha=0.85, edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels([f"S{i+1}" for i in range(n_sec)], color="#8B9EC0", fontsize=9)
    ax.set_ylabel("Sector Time (s)", color="#8B9EC0", fontsize=9); ax.set_title("SECTOR TIME COMPARISON", color="#E8EEF8", fontsize=11, fontweight="bold")
    ax.grid(True, color="#1A2035", axis="y", lw=0.5); ax.spines[:].set_color("#1E2840"); ax.tick_params(colors="#5A6A8A", labelsize=8)
    handles = [mpatches.Patch(color=_color(l, best_lap, i), label=f"Lap {l}{'  ★' if l==best_lap else ''}") for i, l in enumerate(laps_s)]
    ax.legend(handles=handles, fontsize=8, ncol=min(5, len(laps_s)), facecolor="#0F1421", edgecolor="#1E2840")
    plt.tight_layout()
    return _fig_to_b64(fig)

def _chart_lap_consistency(df, lap_data, lap_times, best_lap, selected):
    laps_list = sorted(selected); times_list = [lap_times[l] for l in laps_list]; best_t = min(times_list)
    colors_bar = [COLOR_BEST if t==best_t else (COLOR_WARN if t>best_t+2.0 else COLOR_NEUTRAL) for t in times_list]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_bar, ax_trend = axes
    ax_bar.bar(range(len(laps_list)), times_list, color=colors_bar, alpha=0.85, edgecolor="none", width=0.6)
    ax_bar.axhline(best_t, color="#FFD700", lw=1.5, ls="--", alpha=0.8, label=f"Best {fmt_time(best_t)}")
    for i, (lap, t) in enumerate(zip(laps_list, times_list)):
        diff = t - best_t
        ax_bar.text(i, t+0.1, f"+{diff:.2f}" if diff>0 else "BEST", ha="center", va="bottom", fontsize=7, color="#FFD700" if diff==0 else "#8B9EC0")
    ax_bar.set_xticks(range(len(laps_list))); ax_bar.set_xticklabels([f"L{l}" for l in laps_list], color="#8B9EC0", fontsize=8)
    ax_bar.set_ylabel("Lap Time (s)", color="#8B9EC0", fontsize=9); ax_bar.set_title("LAP TIME BREAKDOWN", color="#E8EEF8", fontsize=10, fontweight="bold")
    ax_bar.grid(True, color="#1A2035", axis="y", lw=0.5); ax_bar.spines[:].set_color("#1E2840"); ax_bar.tick_params(colors="#5A6A8A", labelsize=8); ax_bar.legend(fontsize=8)
    ax_trend.plot(range(len(laps_list)), times_list, "o-", color=COLOR_NEUTRAL, lw=1.5, ms=6, alpha=0.8)
    best_i = times_list.index(min(times_list)); ax_trend.scatter([best_i], [min(times_list)], s=80, color=COLOR_BEST, zorder=10)
    if len(times_list) >= 3:
        roll = pd.Series(times_list).rolling(3, center=True).mean()
        ax_trend.plot(range(len(laps_list)), roll, color="#FFD700", lw=1.5, ls="--", alpha=0.7, label="Rolling avg (3)")
    ax_trend.set_xticks(range(len(laps_list))); ax_trend.set_xticklabels([f"L{l}" for l in laps_list], color="#8B9EC0", fontsize=8)
    ax_trend.set_ylabel("Lap Time (s)", color="#8B9EC0", fontsize=9); ax_trend.set_title("LAP TIME TREND", color="#E8EEF8", fontsize=10, fontweight="bold")
    ax_trend.grid(True, color="#1A2035", lw=0.5); ax_trend.spines[:].set_color("#1E2840"); ax_trend.tick_params(colors="#5A6A8A", labelsize=8); ax_trend.legend(fontsize=8)
    plt.tight_layout()
    return _fig_to_b64(fig)

def _chart_racing_line(df, lap_data, lap_times, best_lap, selected):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax1, ax2 = axes
    for ax in axes: ax.set_facecolor("#080C14"); ax.axis("off")
    if "lat" in df.columns and "lon" in df.columns:
        _track_bg(ax1, df, lw=12); _track_bg(ax2, df, lw=12)
    for idx, lap in enumerate(sorted(selected, key=lambda l: lap_times[l])):
        ld = lap_data[lap]; is_best = (lap == best_lap); c = _color(lap, best_lap, idx)
        ax1.plot(ld["lon"], ld["lat"], color=c, lw=2.2 if is_best else 0.9,
                 alpha=1.0 if is_best else 0.4, label=f"Lap {lap}{'  ★' if is_best else ''}  {fmt_time(lap_times[lap])}", zorder=10 if is_best else 3)
    ax1.set_aspect("equal"); ax1.set_title("RACING LINE OVERLAY", color="#E8EEF8", fontsize=10, fontweight="bold", pad=6)
    ax1.legend(fontsize=7, loc="best", ncol=2, framealpha=0.7, facecolor="#0F1421", edgecolor="#1E2840")
    ld_b = lap_data[best_lap]
    if "speed_kmh" in ld_b.columns:
        sc = ax2.scatter(ld_b["lon"], ld_b["lat"], c=ld_b["speed_kmh"], cmap="turbo", s=10, alpha=0.95, zorder=5, linewidths=0)
        cb = plt.colorbar(sc, ax=ax2, pad=0.02, fraction=0.03); cb.set_label("Speed (km/h)", color="#8B9EC0", fontsize=8)
    ax2.set_aspect("equal"); ax2.set_title(f"SPEED HEATMAP  Lap {best_lap} ★BEST", color="#E8EEF8", fontsize=10, fontweight="bold", pad=6)
    plt.tight_layout()
    return _fig_to_b64(fig)

def _chart_theoretical_best(df, lap_data, lap_times, best_lap, selected):
    tb = _theoretical_best(lap_data, lap_times)
    best_t = lap_times[best_lap]; gain = best_t - tb
    laps_list = sorted(lap_times.keys()); times_list = [lap_times[l] for l in laps_list]
    fig, ax = plt.subplots(figsize=(12, 5))
    colors_bar = [COLOR_BEST if l==best_lap else COLOR_NEUTRAL for l in laps_list]
    ax.bar(range(len(laps_list)), times_list, color=colors_bar, alpha=0.85, edgecolor="none", width=0.6)
    ax.axhline(tb, color="#FFD700", lw=2, ls="--", label=f"Theoretical Best  {fmt_time(tb)}")
    ax.axhline(best_t, color=COLOR_BEST, lw=1.5, ls=":", label=f"Best Lap  {fmt_time(best_t)}")
    ax.set_xticks(range(len(laps_list))); ax.set_xticklabels([f"L{l}" for l in laps_list], color="#8B9EC0", fontsize=8)
    ax.set_ylabel("Lap Time (s)", color="#8B9EC0", fontsize=9)
    ax.set_title(f"THEORETICAL BEST  (Gain: {gain:.3f}s)", color="#E8EEF8", fontsize=11, fontweight="bold")
    ax.grid(True, color="#1A2035", axis="y", lw=0.5); ax.spines[:].set_color("#1E2840"); ax.tick_params(colors="#5A6A8A", labelsize=8); ax.legend(fontsize=9)
    plt.tight_layout()
    return _fig_to_b64(fig), {"theoretical_best": tb, "best_lap_time": best_t, "gain": gain}

def _chart_brake_distance(df, lap_data, lap_times, best_lap, selected):
    ld_b = _lap_dist(lap_data[best_lap]); corners = _detect_corners(ld_b)
    if not corners:
        fig, ax = plt.subplots(figsize=(12,4)); ax.text(0.5,0.5,"No corners detected",transform=ax.transAxes,color="#8B9EC0",ha="center",va="center"); return _fig_to_b64(fig)
    records = []
    for lap in selected:
        ld = _lap_dist(lap_data[lap])
        if "long_g" not in ld.columns: continue
        for c_num, (ds, de) in enumerate(corners, 1):
            pre = ld[(ld["ld"] >= max(0, ds-150)) & (ld["ld"] < ds)]
            if len(pre) < 2: continue
            mask = pre["long_g"] < BRAKE_THR
            if not mask.any(): records.append({"lap":lap,"corner":c_num,"brake_dist":0.0,"brake_spd":float(pre["speed_kmh"].iloc[-1]) if "speed_kmh" in pre.columns else 0,"no_brake":True}); continue
            bidx = int(mask.idxmax()); brow = ld.loc[bidx]
            brake_dist = max(0.0, float(ds - brow["ld"]))
            brake_spd  = float(brow["speed_kmh"]) if "speed_kmh" in brow.index else 0
            records.append({"lap":lap,"corner":c_num,"brake_dist":brake_dist,"brake_spd":brake_spd,"no_brake":False})
    if not records:
        fig, ax = plt.subplots(figsize=(12,4)); ax.text(0.5,0.5,"No brake data",transform=ax.transAxes,color="#8B9EC0",ha="center",va="center"); return _fig_to_b64(fig)
    rdf = pd.DataFrame(records); n_corners = rdf["corner"].max()
    fig, ax = plt.subplots(figsize=(13, 5))
    for li, lap in enumerate(sorted(selected, key=lambda l: lap_times[l])):
        sub = rdf[rdf["lap"]==lap].sort_values("corner")
        if sub.empty: continue
        c = _color(lap, best_lap, li); is_best = (lap==best_lap)
        ax.plot(sub["corner"], sub["brake_dist"], "o-", color=c, lw=2 if is_best else 1, ms=7 if is_best else 4, alpha=1 if is_best else 0.65, label=f"Lap {lap}{'  ★' if is_best else ''}")
    ax.set_xlabel("Corner #", color="#8B9EC0", fontsize=9); ax.set_ylabel("Brake Distance (m before corner)", color="#8B9EC0", fontsize=9)
    ax.set_title("BRAKE DISTANCE ANALYSIS", color="#E8EEF8", fontsize=11, fontweight="bold")
    ax.grid(True, color="#1A2035", lw=0.5); ax.spines[:].set_color("#1E2840"); ax.tick_params(colors="#5A6A8A", labelsize=8); ax.legend(fontsize=8)
    plt.tight_layout()
    return _fig_to_b64(fig)

def _chart_throttle_on(df, lap_data, lap_times, best_lap, selected):
    ld_b = _lap_dist(lap_data[best_lap]); corners = _detect_corners(ld_b)
    if not corners:
        fig, ax = plt.subplots(figsize=(12,4)); ax.text(0.5,0.5,"No corners detected",transform=ax.transAxes,color="#8B9EC0",ha="center",va="center"); return _fig_to_b64(fig)
    records = []
    for lap in selected:
        ld = _lap_dist(lap_data[lap])
        if "long_g" not in ld.columns or "speed_kmh" not in ld.columns: continue
        for c_num, (ds, de) in enumerate(corners, 1):
            margin = (de - ds) * 0.5
            sec = ld[(ld["ld"] >= ds-20) & (ld["ld"] <= de+margin)]
            if len(sec) < MIN_CORNER_PTS: continue
            spd = sec["speed_kmh"].values; lg = sec["long_g"].values; dist = sec["ld"].values
            min_idx = int(np.argmin(spd)); min_spd = float(spd[min_idx]); min_dist = float(dist[min_idx])
            thr_mask = lg[min_idx:] > THROTTLE_THR
            thr_dist = float(dist[min_idx + int(np.argmax(thr_mask))]) if thr_mask.any() else min_dist
            coast = max(0.0, thr_dist - min_dist)
            records.append({"lap":lap,"corner":c_num,"min_spd":min_spd,"min_dist":min_dist,"thr_dist":thr_dist,"coast":coast})
    if not records:
        fig, ax = plt.subplots(figsize=(12,4)); ax.text(0.5,0.5,"No data",transform=ax.transAxes,color="#8B9EC0",ha="center",va="center"); return _fig_to_b64(fig)
    rdf = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    ax1, ax2 = axes
    for li, lap in enumerate(sorted(selected, key=lambda l: lap_times[l])):
        sub = rdf[rdf["lap"]==lap].sort_values("corner")
        if sub.empty: continue
        c = _color(lap, best_lap, li); is_best = (lap==best_lap)
        lw = 2 if is_best else 1; al = 1 if is_best else 0.65; ms = 7 if is_best else 4
        ax1.plot(sub["corner"], sub["min_spd"], "o-", color=c, lw=lw, ms=ms, alpha=al, label=f"Lap {lap}{'  ★' if is_best else ''}")
        ax2.plot(sub["corner"], sub["coast"],   "o-", color=c, lw=lw, ms=ms, alpha=al)
    ax1.set_xlabel("Corner #", color="#8B9EC0", fontsize=9); ax1.set_ylabel("Minimum Speed (km/h)", color="#8B9EC0", fontsize=9); ax1.set_title("MINIMUM SPEED (Clip Speed)", color="#E8EEF8", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Corner #", color="#8B9EC0", fontsize=9); ax2.set_ylabel("Coast Zone (m)", color="#8B9EC0", fontsize=9); ax2.set_title("THROTTLE-ON POINT (Coast Length)", color="#E8EEF8", fontsize=10, fontweight="bold")
    for ax in axes: ax.grid(True, color="#1A2035", lw=0.5); ax.spines[:].set_color("#1E2840"); ax.tick_params(colors="#5A6A8A", labelsize=8)
    ax1.legend(fontsize=8)
    plt.tight_layout()
    return _fig_to_b64(fig)

def _chart_os_us(df, lap_data, lap_times, best_lap, selected):
    ld_b = _lap_dist(lap_data[best_lap]); corners = _detect_corners(ld_b)
    if not corners:
        fig, ax = plt.subplots(figsize=(12,4)); ax.text(0.5,0.5,"No corners detected",transform=ax.transAxes,color="#8B9EC0",ha="center",va="center"); return _fig_to_b64(fig)
    records = []
    for lap in selected:
        ld = _lap_dist(lap_data[lap])
        if "lat_g" not in ld.columns or "long_g" not in ld.columns: continue
        for c_num, (ds, de) in enumerate(corners, 1):
            sec = ld[(ld["ld"] >= ds-10) & (ld["ld"] <= de+20)]
            if len(sec) < MIN_CORNER_PTS: continue
            latg = sec["lat_g"].values; lg = sec["long_g"].values; spd = sec["speed_kmh"].values if "speed_kmh" in sec.columns else np.ones(len(sec))
            # 簡易OS/US判定（横G急落 + 縦G復帰パターン）
            latg_abs = np.abs(latg); latg_smooth = pd.Series(latg_abs).rolling(3,min_periods=1).mean().values
            max_latg = float(latg_smooth.max()); min_spd = float(spd.min())
            # 後半の横G変化
            half = len(sec)//2
            latg_first = float(latg_smooth[:half].mean()) if half>0 else 0
            latg_last  = float(latg_smooth[half:].mean()) if half>0 else 0
            drop = latg_first - latg_last  # 正 = コーナー後半で横G低下 = US傾向
            if   drop >  0.15 and max_latg > 0.3: behavior = "US"
            elif drop < -0.15 and max_latg > 0.3: behavior = "OS"
            else:                                  behavior = "NEU"
            records.append({"lap":lap,"corner":c_num,"behavior":behavior,"max_latg":max_latg,"min_spd":min_spd})
    if not records:
        fig, ax = plt.subplots(figsize=(12,4)); ax.text(0.5,0.5,"No data",transform=ax.transAxes,color="#8B9EC0",ha="center",va="center"); return _fig_to_b64(fig)
    rdf = pd.DataFrame(records); color_map = {"OS":"#FF4E4E","US":"#4E9FFF","NEU":"#4EFF9F"}
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    ax1, ax2 = axes
    for lap in sorted(selected):
        sub = rdf[rdf["lap"]==lap]
        for _, row in sub.iterrows():
            ax1.scatter(row["corner"], lap, c=color_map[row["behavior"]], s=120, marker="s", zorder=5, linewidths=0)
    ax1.set_xlabel("Corner #", color="#8B9EC0", fontsize=9); ax1.set_ylabel("Lap", color="#8B9EC0", fontsize=9)
    ax1.set_title("OS/US DETECTION  (Red=OS  Blue=US  Green=NEU)", color="#E8EEF8", fontsize=10, fontweight="bold")
    ax1.grid(True, color="#1A2035", lw=0.5); ax1.spines[:].set_color("#1E2840"); ax1.tick_params(colors="#5A6A8A", labelsize=8)
    summary = rdf.groupby(["corner","behavior"]).size().unstack(fill_value=0)
    for beh, c in color_map.items():
        if beh in summary.columns:
            ax2.bar(summary.index, summary[beh], label=beh, color=c, alpha=0.8, edgecolor="none")
    ax2.set_xlabel("Corner #", color="#8B9EC0", fontsize=9); ax2.set_ylabel("Count", color="#8B9EC0", fontsize=9)
    ax2.set_title("BEHAVIOR FREQUENCY BY CORNER", color="#E8EEF8", fontsize=10, fontweight="bold")
    ax2.grid(True, color="#1A2035", axis="y", lw=0.5); ax2.spines[:].set_color("#1E2840"); ax2.tick_params(colors="#5A6A8A", labelsize=8); ax2.legend(fontsize=8)
    plt.tight_layout()
    return _fig_to_b64(fig)

def _chart_lap_list(df, lap_data, lap_times, best_lap, selected):
    """ラップタイム一覧テーブル（chart画像 + JSONデータ両方返す）"""
    laps_list = sorted(lap_times.keys()); times = [lap_times[l] for l in laps_list]
    best_t = min(times); std = float(np.std(times)); mean_t = float(np.mean(times))
    tb = _theoretical_best(lap_data, lap_times)
    rows = [{"lap":l,"time_sec":t,"time_fmt":fmt_time(t),"delta":round(t-best_t,3),"is_best":t==best_t} for l,t in zip(laps_list,times)]
    return {"rows": rows, "best_time": fmt_time(best_t), "mean_time": fmt_time(mean_t), "std": round(std,3), "theoretical_best": fmt_time(tb), "gain": round(best_t-tb,3)}

# ============================================================
# AI 解析関数
# ============================================================

def _ai_next_lap(df, lap_data, lap_times, best_lap):
    laps = sorted(lap_times.keys()); times = np.array([lap_times[l] for l in laps]); n = len(laps)
    if n < 4: return {"error": "4ラップ以上必要です", "predicted": None}
    X = np.array([[i, lap_times[l], np.std(times[:max(1,i)]), max(times[:max(1,i)]) - min(times[:max(1,i)])] for i, l in enumerate(laps)])
    y = times
    try:
        loo = LeaveOneOut(); preds, trues = [], []
        for tr, te in loo.split(X):
            m = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
            m.fit(X[tr], y[tr]); preds.append(float(m.predict(X[te])[0])); trues.append(float(y[te]))
        mae = mean_absolute_error(trues, preds)
        m_final = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
        m_final.fit(X, y)
        next_feat = np.array([[n, float(np.mean(times[-3:])), float(np.std(times)), float(max(times)-min(times))]])
        pred = float(m_final.predict(next_feat)[0])
        return {"predicted": round(pred, 3), "predicted_fmt": fmt_time(pred), "mae": round(mae, 3), "confidence": round(max(0, 1 - mae/max(times,1)), 3), "trend": "improving" if times[-1] < np.mean(times[:-1]) else "degrading"}
    except Exception as e:
        return {"error": str(e), "predicted": None}

def _ai_tire_degradation(df, lap_data, lap_times, best_lap):
    laps = sorted(lap_times.keys()); times = np.array([lap_times[l] for l in laps]); n = len(laps)
    if n < 3: return {"error": "3ラップ以上必要です"}
    thresh = np.percentile(times, 85); mask = times < thresh
    x_clean = np.arange(n)[mask if mask.sum()>=3 else np.ones(n,bool)]
    t_clean = times[mask if mask.sum()>=3 else np.ones(n,bool)]
    coef = np.polyfit(x_clean, t_clean, deg=2)
    base_time = float(np.polyval(coef, 0)); deg_per_lap = float(coef[1])
    optimal_stint = n
    for i in range(1, 200):
        if np.polyval(coef, i) - base_time >= 1.5: optimal_stint = i; break
    x_curve = np.arange(min(n+10, 60)); y_curve = np.polyval(coef, x_curve).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    ax1, ax2 = axes
    ax1.scatter(range(n), times, color=COLOR_NEUTRAL, s=40, zorder=5, label="Actual")
    ax1.plot(x_curve[:n+5], y_curve[:n+5], color=COLOR_WARN, lw=2, label="Degradation fit")
    ax1.axvline(optimal_stint, color=COLOR_GOOD, lw=1.5, ls="--", label=f"Optimal stint: {optimal_stint} laps")
    ax1.axhline(base_time+1.5, color=COLOR_WARN, lw=1, ls=":", alpha=0.7, label="Threshold +1.5s")
    ax1.set_xlabel("Lap", color="#8B9EC0", fontsize=9); ax1.set_ylabel("Lap Time (s)", color="#8B9EC0", fontsize=9)
    ax1.set_title("TIRE DEGRADATION MODEL", color="#E8EEF8", fontsize=11, fontweight="bold")
    ax1.grid(True, color="#1A2035", lw=0.5); ax1.spines[:].set_color("#1E2840"); ax1.tick_params(colors="#5A6A8A", labelsize=8); ax1.legend(fontsize=8)
    ax2.bar(range(n), times-base_time, color=[COLOR_WARN if t-base_time>1.5 else COLOR_NEUTRAL for t in times], alpha=0.85, edgecolor="none", width=0.6)
    ax2.axhline(1.5, color=COLOR_WARN, lw=1.5, ls="--", label="Threshold")
    ax2.set_xlabel("Lap", color="#8B9EC0", fontsize=9); ax2.set_ylabel("Degradation (s from base)", color="#8B9EC0", fontsize=9)
    ax2.set_title("LAP-BY-LAP DEGRADATION", color="#E8EEF8", fontsize=10, fontweight="bold")
    ax2.grid(True, color="#1A2035", axis="y", lw=0.5); ax2.spines[:].set_color("#1E2840"); ax2.tick_params(colors="#5A6A8A", labelsize=8); ax2.legend(fontsize=8)
    plt.tight_layout()
    img = _fig_to_b64(fig)
    return {"image": img, "base_time": round(base_time,3), "deg_per_lap": round(deg_per_lap,4), "optimal_stint": optimal_stint, "curve_x": x_curve.tolist(), "curve_y": [round(v,3) for v in y_curve]}

def _ai_corner_priority(df, lap_data, lap_times, best_lap):
    ld_b = _lap_dist(lap_data[best_lap]); corners = _detect_corners(ld_b)
    if not corners: return {"error": "コーナー検出失敗", "priority": []}
    priority = []
    for c_num, (ds, de) in enumerate(corners, 1):
        all_min = []; all_exit = []; all_entry = []
        for lap in lap_times:
            ld = _lap_dist(lap_data[lap]); sec = ld[(ld["ld"]>=ds)&(ld["ld"]<=de)]
            if len(sec) > 2:
                all_min.append(float(sec["speed_kmh"].min()))
                all_exit.append(float(sec["speed_kmh"].iloc[-1]))
                all_entry.append(float(sec["speed_kmh"].iloc[0]))
        if not all_min: continue
        best_min = max(all_min); cur_min = all_min[0]; speed_loss = best_min - cur_min
        entry_var = np.std(all_entry)/max(np.mean(all_entry),1) if all_entry else 0
        exit_var  = np.std(all_exit)/max(np.mean(all_exit),1) if all_exit else 0
        score = speed_loss * (1 + entry_var + exit_var)
        if   speed_loss > 5: reason = f"ミニマム速度ロス {speed_loss:.1f}km/h → ブレーキ解除を早める"
        elif entry_var > 0.05: reason = f"進入速度ばらつき → 制動点の一貫性向上"
        elif exit_var  > 0.05: reason = f"脱出速度ばらつき → スロットルONタイミングを統一"
        else: reason = "ライン精度向上で改善可能"
        priority.append({"corner":c_num,"score":round(score,4),"speed_loss":round(speed_loss,2),"best_min":round(best_min,1),"cur_min":round(cur_min,1),"reason":reason})
    priority.sort(key=lambda x: -x["score"])
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    ax1, ax2 = axes
    if priority:
        top = priority[:8]; colors_p = [COLOR_WARN if p["speed_loss"]>5 else (COLOR_NEUTRAL if p["speed_loss"]>2 else COLOR_GOOD) for p in top]
        ax1.barh([f"C{p['corner']}" for p in top], [p["score"] for p in top], color=colors_p, alpha=0.85, edgecolor="none")
        ax1.set_xlabel("Priority Score", color="#8B9EC0", fontsize=9); ax1.set_title("CORNER PRIORITY (AI)", color="#E8EEF8", fontsize=10, fontweight="bold")
        ax1.grid(True, color="#1A2035", axis="x", lw=0.5); ax1.spines[:].set_color("#1E2840"); ax1.tick_params(colors="#5A6A8A", labelsize=8)
        ax2.barh([f"C{p['corner']}" for p in top], [p["speed_loss"] for p in top], color=colors_p, alpha=0.85, edgecolor="none")
        ax2.set_xlabel("Speed Loss (km/h)", color="#8B9EC0", fontsize=9); ax2.set_title("MINIMUM SPEED LOSS", color="#E8EEF8", fontsize=10, fontweight="bold")
        ax2.grid(True, color="#1A2035", axis="x", lw=0.5); ax2.spines[:].set_color("#1E2840"); ax2.tick_params(colors="#5A6A8A", labelsize=8)
    plt.tight_layout()
    img = _fig_to_b64(fig)
    return {"image": img, "priority": priority}

def _ai_advice(df, lap_data, lap_times, best_lap):
    laps = sorted(lap_times.keys()); times = [lap_times[l] for l in laps]
    best_t = lap_times[best_lap]; tb = _theoretical_best(lap_data, lap_times)
    gain = best_t - tb; std = float(np.std(times)); total = len(laps)
    ld = lap_data[best_lap]
    max_spd = float(ld["speed_kmh"].max()) if "speed_kmh" in ld.columns else 0
    min_spd = float(ld["speed_kmh"].min()) if "speed_kmh" in ld.columns else 0
    avg_spd = float(ld["speed_kmh"].mean()) if "speed_kmh" in ld.columns else 0
    max_latg = float(ld["lat_g"].abs().max()) if "lat_g" in ld.columns else 0
    max_brakeg = float(ld["long_g"].min()) if "long_g" in ld.columns else 0
    max_accg  = float(ld["long_g"].max()) if "long_g" in ld.columns else 0
    cons = "高い" if std < 1.0 else ("普通" if std < 2.0 else "低い")
    advices = []
    if gain > 3: advices.append({"icon":"🏆","title":"タイムポテンシャル","body":f"理論ベスト {fmt_time(tb)} まで {gain:.2f}秒。各セクターでベストを同一ラップで再現することが目標です。"})
    elif gain > 1: advices.append({"icon":"🏆","title":"タイムポテンシャル","body":f"理論ベスト {fmt_time(tb)} まであと {gain:.2f}秒。細部の詰めが重要です。"})
    else: advices.append({"icon":"🏆","title":"タイムポテンシャル","body":f"理論ベストまで {gain:.2f}秒。素晴らしい走りです！"})
    if std > 2.0: advices.append({"icon":"📊","title":"ラップ安定性","body":f"STD {std:.2f}秒は大きいです。制動点を毎ラップ同じ目印で行い、進入速度を一定にしてください。"})
    elif std < 0.8: advices.append({"icon":"📊","title":"ラップ安定性","body":f"STD {std:.2f}秒。非常に安定した走りです。"})
    if max_latg < 1.2: advices.append({"icon":"↩️","title":"コーナリング","body":f"横G最大 {max_latg:.2f}g。グリップ限界に余裕があります。コーナリング速度を3〜5km/h上げてください。"})
    if abs(max_brakeg) < 0.8: advices.append({"icon":"🛑","title":"ブレーキング","body":f"制動G最大 {abs(max_brakeg):.2f}g。制動点を5〜10m遅らせ、より強い初期制動を試してください。"})
    if min_spd < 50: advices.append({"icon":"⬇️","title":"コーナー最低速度","body":f"最低速度 {min_spd:.1f}km/h。トレイルブレーキングでクリップ速度を上げてください。"})
    summary = {"best_lap":best_lap,"best_time":fmt_time(best_t),"theoretical_best":fmt_time(tb),"gain":round(gain,3),"std":round(std,3),"total_laps":total,"max_speed":round(max_spd,1),"min_speed":round(min_spd,1),"avg_speed":round(avg_spd,1),"max_latg":round(max_latg,2),"max_brakeg":round(abs(max_brakeg),2),"max_accg":round(max_accg,2),"consistency":cons}
    return {"advices": advices, "summary": summary}

# ============================================================
# PDF生成
# ============================================================

def _generate_pdf_bytes(df, lap_data, lap_times, best_lap, selected):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable
        from reportlab.lib.styles import ParagraphStyle
        import datetime, tempfile, os as _os
    except ImportError:
        return None, "reportlab not installed"

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15*mm, rightMargin=15*mm, topMargin=15*mm, bottomMargin=15*mm)
    C_BG = colors.HexColor('#0A0E1A'); C_ACCENT = colors.HexColor('#00E5FF')
    C_GOLD = colors.HexColor('#FFD700'); C_TEXT = colors.HexColor('#E8EEF8'); C_SUB = colors.HexColor('#8B9EC0')
    W = 174*mm
    def S(name,**kw): return ParagraphStyle(name,**{"fontName":"Helvetica",**kw})
    def mkp(txt,c=None): return Paragraph(str(txt),S(f'p{id(txt)}',fontSize=9,textColor=c or C_SUB))

    story = []
    # タイトル
    title_tbl = Table([[Paragraph("🏁  RACE TELEMETRY ANALYSIS REPORT",S('T',fontSize=16,textColor=C_ACCENT,fontName='Helvetica-Bold'))]], colWidths=[W])
    title_tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#0F1421')),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10),('LEFTPADDING',(0,0),(-1,-1),12)]))
    story.append(title_tbl); story.append(Spacer(1,8*mm))

    # サマリー
    advice = _ai_advice(df, lap_data, lap_times, best_lap)["summary"]
    info_rows = [["Best Lap", advice["best_time"]], ["Theoretical Best", advice["theoretical_best"]], ["Potential Gain", f"{advice['gain']:.3f}s"], ["Total Laps", str(advice["total_laps"])], ["Lap STD", f"{advice['std']:.3f}s"], ["Consistency", advice["consistency"]], ["Max Speed", f"{advice['max_speed']} km/h"], ["Max Lat-G", f"{advice['max_latg']}g"]]
    tbl = Table([[mkp(r[0],C_ACCENT), mkp(r[1],C_TEXT)] for r in info_rows], colWidths=[W/2,W/2])
    tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#0F1421')),('LINEBELOW',(0,0),(-1,-1),0.5,colors.HexColor('#1E2840')),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),8)]))
    story.append(tbl); story.append(Spacer(1,6*mm))

    # ラップタイム一覧
    lap_info = _chart_lap_list(df, lap_data, lap_times, best_lap, selected)
    lt_rows = [[mkp("Lap",C_ACCENT), mkp("Time",C_ACCENT), mkp("Δ Best",C_ACCENT)]]
    for row in lap_info["rows"]:
        c = C_GOLD if row["is_best"] else C_SUB
        lt_rows.append([mkp(str(row["lap"]),c), mkp(row["time_fmt"],c), mkp("BEST" if row["is_best"] else f"+{row['delta']:.3f}s",c)])
    lt_tbl = Table(lt_rows, colWidths=[W/3,W/3,W/3])
    lt_tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#141929')),('BACKGROUND',(0,1),(-1,-1),colors.HexColor('#0F1421')),('LINEBELOW',(0,0),(-1,-1),0.5,colors.HexColor('#1E2840')),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),8)]))
    story.append(lt_tbl); story.append(Spacer(1,6*mm))

    # チャート画像を埋め込む
    def embed_chart(title, img_b64):
        story.append(Paragraph(f"◆  {title}", S('h',fontSize=10,textColor=C_TEXT,fontName='Helvetica-Bold')))
        story.append(Spacer(1,2*mm))
        img_data = base64.b64decode(img_b64.split(",")[1])
        img_buf = io.BytesIO(img_data)
        story.append(Image(img_buf, width=W, height=W*0.5))
        story.append(Spacer(1,5*mm))

    try: embed_chart("SPEED TRACE", _chart_speed_trace(df, lap_data, lap_times, best_lap, selected))
    except: pass
    try: embed_chart("DELTA TIME",  _chart_delta_time(df, lap_data, lap_times, best_lap, selected))
    except: pass
    try: embed_chart("G-G DIAGRAM", _chart_gg(df, lap_data, lap_times, best_lap, selected))
    except: pass
    try:
        result = _chart_corner_speed(df, lap_data, lap_times, best_lap, selected)
        embed_chart("CORNER SPEED ANALYSIS", result[0] if isinstance(result, tuple) else result)
    except: pass

    # AIアドバイス
    story.append(Spacer(1,4*mm)); story.append(Paragraph("◆  AI DRIVING ADVICE", S('h',fontSize=10,textColor=C_TEXT,fontName='Helvetica-Bold'))); story.append(Spacer(1,2*mm))
    for adv in _ai_advice(df, lap_data, lap_times, best_lap)["advices"]:
        story.append(mkp(f"{adv['icon']}  {adv['title']}: {adv['body']}", C_SUB)); story.append(Spacer(1,2*mm))

    # フッター
    story.append(Spacer(1,8*mm)); story.append(HRFlowable(width=W, color=colors.HexColor('#1E2840')))
    story.append(mkp(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Race Telemetry Analyzer Web", C_SUB))

    doc.build(story)
    buf.seek(0)
    return buf.read(), None

# ============================================================
# 公開エントリーポイント
# ============================================================

CHART_FUNCS = {
    "speed_trace":       _chart_speed_trace,
    "delta_time":        _chart_delta_time,
    "gg_diagram":        _chart_gg,
    "brake_map":         _chart_brake_map,
    "corner_speed":      _chart_corner_speed,
    "sector_comparison": _chart_sector,
    "lap_consistency":   _chart_lap_consistency,
    "racing_line":       _chart_racing_line,
    "theoretical_best":  _chart_theoretical_best,
    "brake_distance":    _chart_brake_distance,
    "throttle_on":       _chart_throttle_on,
    "os_us":             _chart_os_us,
}

AI_FUNCS = {
    "next_lap_prediction": _ai_next_lap,
    "tire_degradation":    _ai_tire_degradation,
    "corner_priority":     _ai_corner_priority,
    "ai_advice":           _ai_advice,
    "lap_list":            lambda df, lap_data, lap_times, best_lap: _chart_lap_list(df, lap_data, lap_times, best_lap, list(lap_times.keys())),
}


def compute_chart(chart_id: str, df, lap_data, lap_times, best_lap, selected_laps):
    if chart_id not in CHART_FUNCS:
        raise ValueError(f"Unknown chart: {chart_id}")
    result = CHART_FUNCS[chart_id](df, lap_data, lap_times, best_lap, selected_laps)
    if isinstance(result, tuple):
        return {"image": result[0], "data": result[1]}
    if isinstance(result, str):
        return {"image": result}
    return {"data": result}


def compute_ai(ai_id: str, df, lap_data, lap_times, best_lap):
    if ai_id not in AI_FUNCS:
        raise ValueError(f"Unknown AI function: {ai_id}")
    return AI_FUNCS[ai_id](df, lap_data, lap_times, best_lap)


def generate_pdf_bytes(df, lap_data, lap_times, best_lap, selected_laps):
    return _generate_pdf_bytes(df, lap_data, lap_times, best_lap, selected_laps)
