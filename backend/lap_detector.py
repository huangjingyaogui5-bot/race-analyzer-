"""
lap_detector.py  ─  Web対応 ラップ検出エンジン
================================================
元の lap_ai.py から TkAgg / matplotlib GUI を完全除去。
フロントエンドの Web Canvas クリックで S/F 座標を受け取り、
同じ距離ベース検出 + RandomForest 学習を実行する。

公開API:
  detect_laps(df, sf_lat, sf_lon, col_map, session_model) -> LapResult
  build_features(df, lat_col, lon_col, speed_col, time_col) -> DataFrame
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from math import radians, cos, sin, sqrt, atan2
from dataclasses import dataclass, field
from typing import Optional
import pickle, io, base64


# ============================================================
# データクラス
# ============================================================

@dataclass
class LapInfo:
    lap:        int
    time_sec:   float          # ラップタイム（秒）
    start_idx:  int
    end_idx:    int
    start_time: float
    end_time:   float

@dataclass
class LapResult:
    laps:          list[LapInfo]
    lap_col:       list[int]    # df の各行に対応するラップ番号配列
    crossings:     list[int]    # S/F通過インデックス
    sf_point:      tuple[float, float]   # (lat, lon)
    ai_score:      Optional[float]       # F1スコア（学習した場合）
    method:        str          # "ai_model" | "distance" | "manual_only"
    model_bytes:   Optional[bytes]       # pkl バイト列（Supabase保存用）
    gps_track:     list[dict]   # [{lat, lon, lap}] 間引きしたトラック点


# ============================================================
# ユーティリティ
# ============================================================

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def build_features(df: pd.DataFrame,
                   lat_col:   str,
                   lon_col:   str,
                   speed_col: str,
                   time_col:  str) -> pd.DataFrame:
    """各データ点の特徴量を生成（lap_ai.py と同一ロジック）"""
    feat = pd.DataFrame(index=df.index)

    spd = df[speed_col].fillna(0)
    feat["speed"]        = spd
    feat["speed_diff"]   = spd.diff().fillna(0)
    feat["speed_roll5"]  = spd.rolling(5,  min_periods=1).mean()
    feat["speed_roll20"] = spd.rolling(20, min_periods=1).mean()
    feat["speed_std10"]  = spd.rolling(10, min_periods=1).std().fillna(0)

    lat = df[lat_col].ffill()
    lon = df[lon_col].ffill()
    feat["lat"]  = lat
    feat["lon"]  = lon
    feat["dlat"] = lat.diff().fillna(0)
    feat["dlon"] = lon.diff().fillna(0)

    d = np.sqrt(feat["dlat"]**2 + feat["dlon"]**2) * 111000
    feat["step_dist"] = d
    feat["cum_dist"]  = d.cumsum()

    heading = np.arctan2(feat["dlon"], feat["dlat"])
    feat["heading"]      = heading
    feat["heading_diff"] = pd.Series(
        np.diff(np.unwrap(heading.values), prepend=0), index=df.index)

    dt = df[time_col].diff().fillna(0.05).clip(lower=0.001)
    feat["dt"]    = dt
    feat["time"]  = df[time_col]
    feat["accel"] = spd.diff() / dt / 3.6

    return feat.fillna(0)


def label_crossings(n: int, crossings: list[int], window: int = 3) -> np.ndarray:
    labels = np.zeros(n, dtype=int)
    for idx in crossings:
        for w in range(-window, window + 1):
            i = idx + w
            if 0 <= i < n:
                labels[i] = 1
    return labels


# ============================================================
# S/Fライン通過検出（距離ベース）
# ============================================================

def _detect_crossings_by_distance(df: pd.DataFrame,
                                   lat_col:   str,
                                   lon_col:   str,
                                   speed_col: str,
                                   sf_lat:    float,
                                   sf_lon:    float,
                                   threshold_m:     float = 25.0,
                                   min_lap_points:  int   = 80) -> list[int]:
    """
    S/F点から threshold_m メートル以内を通過ゾーンとみなし、
    最接近点をラップ境界インデックスとして返す。
    """
    lat_m = (df[lat_col].values - sf_lat)  * 111000
    lon_m = (df[lon_col].values - sf_lon)  * 111000 * cos(radians(sf_lat))
    dist  = np.sqrt(lat_m**2 + lon_m**2)

    crossings  = []
    in_zone    = False
    last_idx   = -min_lap_points
    zone_start = 0

    for i, d in enumerate(dist):
        if d < threshold_m and not in_zone:
            in_zone    = True
            zone_start = i
        elif d >= threshold_m and in_zone:
            in_zone = False
            zone_slice = dist[zone_start:i]
            if len(zone_slice) == 0:
                continue
            best_i = int(zone_start + np.argmin(zone_slice))
            if best_i - last_idx >= min_lap_points:
                spd_val = float(df[speed_col].iloc[best_i]) if speed_col in df.columns else 99.0
                if spd_val > 5.0:
                    crossings.append(best_i)
                    last_idx = best_i

    return crossings


def _find_crossings(df, lat_col, lon_col, speed_col,
                    sf_lat, sf_lon) -> list[int]:
    """段階的にしきい値を広げてS/F通過点を確実に検出する"""
    for threshold in [25, 50, 100]:
        crossings = _detect_crossings_by_distance(
            df, lat_col, lon_col, speed_col,
            sf_lat, sf_lon, threshold_m=threshold)
        if len(crossings) >= 2:
            return crossings
    return crossings


# ============================================================
# ラップ番号割り当て
# ============================================================

def _assign_laps(df: pd.DataFrame,
                 crossings: list[int],
                 time_col:  str) -> tuple[list[int], list[LapInfo]]:
    n       = len(df)
    lap_col = [0] * n
    laps    = []

    for i, start in enumerate(crossings):
        end = crossings[i + 1] if i + 1 < len(crossings) else n
        lap_num = i + 1
        for j in range(start, end):
            lap_col[j] = lap_num

        t_vals = df[time_col].iloc[start:end]
        t_start = float(t_vals.iloc[0])
        t_end   = float(t_vals.iloc[-1])
        laps.append(LapInfo(
            lap       = lap_num,
            time_sec  = round(t_end - t_start, 3),
            start_idx = start,
            end_idx   = end - 1,
            start_time= t_start,
            end_time  = t_end,
        ))

    # 短すぎるラップを除外（中央値の 30% 未満）
    if laps:
        median_t = float(np.median([l.time_sec for l in laps]))
        laps = [l for l in laps if l.time_sec > median_t * 0.3]

    return lap_col, laps


# ============================================================
# AI 学習
# ============================================================

def _train_model(df: pd.DataFrame,
                 feat: pd.DataFrame,
                 crossings: list[int]) -> tuple[RandomForestClassifier, StandardScaler, float]:
    labels     = label_crossings(len(df), crossings, window=3)
    pos_weight = max(1, int((labels == 0).sum() / max((labels == 1).sum(), 1)))

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight={0: 1, 1: pos_weight},
        random_state=42,
        n_jobs=-1,
    )
    scaler = StandardScaler()
    X  = feat.values
    Xs = scaler.fit_transform(X)
    clf.fit(Xs, labels)

    n_cv = min(5, max(2, int(labels.sum() // 2)))
    try:
        scores = cross_val_score(clf, Xs, labels, cv=n_cv, scoring="f1")
        score  = float(scores.mean())
    except Exception:
        score = 0.0

    return clf, scaler, score


def _predict_crossings(feat: pd.DataFrame,
                       clf: RandomForestClassifier,
                       scaler: StandardScaler,
                       min_lap_points: int = 80) -> list[int]:
    Xs   = scaler.transform(feat.values)
    prob = clf.predict_proba(Xs)[:, 1]

    candidates = np.where(prob > 0.4)[0]
    crossings  = []
    last       = -min_lap_points
    i = 0
    while i < len(candidates):
        j = i
        while j + 1 < len(candidates) and candidates[j+1] - candidates[j] < 10:
            j += 1
        best = candidates[i:j+1][np.argmax(prob[candidates[i:j+1]])]
        if int(best) - last >= min_lap_points:
            crossings.append(int(best))
            last = int(best)
        i = j + 1

    return crossings


# ============================================================
# GPS トラック間引き（フロントエンドCanvas描画用）
# ============================================================

def _decimate_track(df: pd.DataFrame,
                    lat_col: str,
                    lon_col: str,
                    lap_col: list[int],
                    max_points: int = 2000) -> list[dict]:
    """全体を max_points 点に間引いてフロントに返す"""
    step = max(1, len(df) // max_points)
    rows = []
    for i in range(0, len(df), step):
        rows.append({
            "lat": round(float(df[lat_col].iloc[i]), 7),
            "lon": round(float(df[lon_col].iloc[i]), 7),
            "lap": int(lap_col[i]),
            "spd": round(float(df["speed_kmh"].iloc[i]), 1) if "speed_kmh" in df.columns else 0,
        })
    return rows


# ============================================================
# メインエントリーポイント
# ============================================================

def detect_laps(df:            pd.DataFrame,
                sf_lat:        float,
                sf_lon:        float,
                lat_col:       str   = "lat",
                lon_col:       str   = "lon",
                speed_col:     str   = "speed_kmh",
                time_col:      str   = "time_sec",
                session_model: Optional[bytes] = None) -> LapResult:
    """
    S/F 座標 (sf_lat, sf_lon) を受け取り、ラップを検出して返す。

    session_model: 前回学習した pkl バイト列があれば渡す（再学習スキップ）
    """
    # ── GPS列チェック ─────────────────────────────────────────
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError("GPS列（lat/lon）が見つかりません")
    if speed_col not in df.columns:
        raise ValueError("Speed列が見つかりません")
    if time_col not in df.columns:
        raise ValueError("Time列が見つかりません")

    feat = build_features(df, lat_col, lon_col, speed_col, time_col)

    # ── 既存モデルで推論を試みる ──────────────────────────────
    clf, scaler, ai_score = None, None, None
    method = "distance"

    if session_model:
        try:
            saved    = pickle.loads(session_model)
            clf      = saved["model"]
            scaler   = saved["scaler"]
            ai_score = saved.get("score")
            crossings = _predict_crossings(feat, clf, scaler)
            if len(crossings) >= 2:
                method = "ai_model"
        except Exception:
            crossings = []

    # ── 距離ベース検出（モデルなし or 検出失敗時）───────────
    if method == "distance" or len(crossings) < 2:
        crossings = _find_crossings(df, lat_col, lon_col, speed_col, sf_lat, sf_lon)

    if len(crossings) < 2:
        # S/F 1点しか通過しない → アウトラップのみ
        lap_col_arr = [1] * len(df)
        laps = [LapInfo(
            lap=1, time_sec=float(df[time_col].max() - df[time_col].min()),
            start_idx=0, end_idx=len(df)-1,
            start_time=float(df[time_col].min()),
            end_time=float(df[time_col].max()),
        )]
        return LapResult(
            laps=laps, lap_col=lap_col_arr, crossings=[0],
            sf_point=(sf_lat, sf_lon), ai_score=None,
            method="manual_only", model_bytes=None,
            gps_track=_decimate_track(df, lat_col, lon_col, lap_col_arr),
        )

    # ── AI 学習（距離ベースの結果で教師データ生成）───────────
    if method == "distance":
        clf, scaler, ai_score = _train_model(df, feat, crossings)
        # 再推論
        ai_crossings = _predict_crossings(feat, clf, scaler)
        if len(ai_crossings) >= 2:
            crossings = ai_crossings
            method    = "ai_trained"

    # ── ラップ番号割り当て ────────────────────────────────────
    lap_col_arr, laps = _assign_laps(df, crossings, time_col)

    # ── モデルをバイト列に ────────────────────────────────────
    model_bytes = None
    if clf and scaler:
        buf = io.BytesIO()
        pickle.dump({"model": clf, "scaler": scaler,
                     "sf_point": (sf_lat, sf_lon), "score": ai_score}, buf)
        model_bytes = buf.getvalue()

    gps_track = _decimate_track(df, lat_col, lon_col, lap_col_arr)

    return LapResult(
        laps        = laps,
        lap_col     = lap_col_arr,
        crossings   = crossings,
        sf_point    = (sf_lat, sf_lon),
        ai_score    = ai_score,
        method      = method,
        model_bytes = model_bytes,
        gps_track   = gps_track,
    )
