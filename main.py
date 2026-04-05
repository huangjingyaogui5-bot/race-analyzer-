"""
main.py  ─  耐久レース解析 Web API  (本番対応版)
=================================================
Railway 環境変数:
  ALLOWED_ORIGINS  : カンマ区切りの許可オリジン
                     例) https://your-app.vercel.app,https://your-app2.vercel.app
                     未設定時は開発用に * を使用
  SESSION_TTL_MIN  : セッション有効期限（分）デフォルト120
  PORT             : Railway が自動設定
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import io, uuid, json, os, time, threading
from typing import Optional
from datetime import datetime

# ============================================================
# アプリ初期化
# ============================================================

app = FastAPI(
    title="Race Analyzer API",
    version="4.0.0",
    description="耐久レーステレメトリー解析 Web API",
)

# ── CORS ────────────────────────────────────────────────────
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
if _raw_origins.strip():
    ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]
else:
    # 未設定 = 開発モード（全許可）
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",  # Vercel preview URLも許可
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=False,
)

# ── セッションストア（メモリ + TTL）────────────────────────
SESSION_TTL = int(os.environ.get("SESSION_TTL_MIN", "120")) * 60  # 秒
SESSIONS: dict[str, dict] = {}
_sessions_lock = threading.Lock()

def _cleanup_sessions():
    """期限切れセッションを定期削除（メモリリーク防止）"""
    while True:
        time.sleep(300)  # 5分ごとにチェック
        now = time.time()
        with _sessions_lock:
            expired = [k for k, v in SESSIONS.items()
                       if now - v.get("created_ts", now) > SESSION_TTL]
            for k in expired:
                del SESSIONS[k]
        if expired:
            print(f"[Session] {len(expired)} expired sessions cleaned up")

threading.Thread(target=_cleanup_sessions, daemon=True).start()

def get_session(session_id: str) -> dict:
    with _sessions_lock:
        s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="セッションが見つかりません（期限切れの可能性があります）")
    return s

def save_session(session_id: str, data: dict):
    with _sessions_lock:
        SESSIONS[session_id] = data

# ============================================================
# 定数・マッピング
# ============================================================

BRAKE_THR = -0.2
MIN_SPEED = 5

JP_COL_MAP = {
    "秒": "time_sec", "時間": "time_str", "緯度": "lat", "経度": "lon",
    "距離(km)": "distance_km", "標高(m)": "altitude_m", "速度(km/h)": "speed_kmh",
    "旋回半径(m)": "turn_radius_m", "コーナリングG": "cornering_g",
    "加減速G": "accel_g_raw", "合算G": "total_g",
}

COLUMN_HINTS = {
    "time":     ["time", "sec", "秒", "時間", "elapsed", "t(s)", "time_sec"],
    "lap":      ["lap", "ラップ", "laps"],
    "speed":    ["speed", "速度", "spd", "velocity", "kmh", "km/h", "mph"],
    "throttle": ["throttle", "アクセル", "スロットル", "accel_pct", "tps", "acc(%)"],
    "brake":    ["brake", "ブレーキ", "brk", "brake_pct", "brk(%)"],
    "lat":      ["lat", "緯度", "latitude"],
    "lon":      ["lon", "lng", "経度", "longitude"],
}

# ============================================================
# ユーティリティ
# ============================================================

def detect_encoding_and_load(raw: bytes):
    for enc in ["shift_jis", "cp932", "utf-8", "utf-8-sig", "latin1"]:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            return df, enc
        except Exception:
            continue
    raise ValueError("CSVの文字コードを判定できませんでした")

def auto_rename(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in JP_COL_MAP.items() if k in df.columns})

def guess_column_mapping(columns: list) -> dict:
    mapping = {}
    for role, hints in COLUMN_HINTS.items():
        found = None
        for col in columns:
            if any(h.lower() in col.lower() for h in hints):
                found = col
                break
        mapping[role] = found
    return mapping

def preprocess(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    rename_map = {}
    for role, col in col_map.items():
        if col and col in df.columns:
            internal = {
                "time": "time_sec", "lap": "lap_raw", "speed": "speed_kmh",
                "throttle": "throttle", "brake": "brake_pct", "lat": "lat", "lon": "lon",
            }.get(role)
            if internal and internal != col:
                rename_map[col] = internal
    df = df.rename(columns=rename_map)

    num_cols = ["time_sec", "lat", "lon", "speed_kmh", "throttle", "brake_pct",
                "distance_km", "altitude_m", "cornering_g", "accel_g_raw"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "time_sec" in df.columns:
        df = df.sort_values("time_sec").reset_index(drop=True)

    if "lat" in df.columns and "lon" in df.columns:
        df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
        for col in ["lat", "lon"]:
            df[col] = df[col].rolling(5, min_periods=1).mean()
        if len(df) > 25:
            from scipy.signal import savgol_filter
            w = max(5, min(11, len(df) - 2))
            w = w if w % 2 == 1 else w - 1
            df["lat"] = savgol_filter(df["lat"].fillna(0), w, 3)
            df["lon"] = savgol_filter(df["lon"].fillna(0), w, 3)

    if "speed_kmh" in df.columns:
        jump = df["speed_kmh"].diff().abs()
        df.loc[jump > 80, "speed_kmh"] = np.nan
        df["speed_kmh"] = df["speed_kmh"].interpolate()

    if "time_sec" in df.columns:
        dt = df["time_sec"].diff().fillna(0.05).clip(lower=0.001)
        df["dt"] = dt
        if "speed_kmh" in df.columns:
            spd_ms = df["speed_kmh"] / 3.6
            df["accel"]  = spd_ms.diff() / dt
            df["long_g"] = (df["accel"] / 9.81).clip(-5, 5)

    if "lat" in df.columns and "lon" in df.columns and "speed_kmh" in df.columns and "dt" in df.columns:
        lat_diff = df["lat"].diff().fillna(0)
        lon_diff = df["lon"].diff().fillna(0)
        heading  = np.arctan2(lon_diff, lat_diff)
        h_diff   = np.diff(np.unwrap(heading.values), prepend=0)
        spd_ms2  = df["speed_kmh"].fillna(0) / 3.6
        df["lat_g"] = ((spd_ms2 * h_diff / df["dt"]) / 9.81).clip(-5, 5)
        df["lat_g"] = df["lat_g"].rolling(3).mean().fillna(0)

    if "brake_pct" in df.columns:
        df["brake"] = df["brake_pct"] > 5
    elif "long_g" in df.columns:
        df["brake"] = df["long_g"] < BRAKE_THR

    if "lat" in df.columns and "lon" in df.columns:
        lat1, lat2 = df["lat"].values[:-1], df["lat"].values[1:]
        lon1, lon2 = df["lon"].values[:-1], df["lon"].values[1:]
        d = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111000
        df["dist_m"] = np.insert(np.cumsum(d), 0, 0)

    return df

def df_to_json_safe(df: pd.DataFrame) -> list:
    return json.loads(df.to_json(orient="records", force_ascii=False))

def _fmt_time(sec: float) -> str:
    m = int(sec // 60); s = sec - m * 60
    return f"{m}:{s:06.3f}"

def _get_lap_data(session_id: str):
    session = get_session(session_id)
    df = session.get("proc_df")
    if df is None:
        raise HTTPException(status_code=400, detail="前処理未完了")
    if "lap" not in df.columns:
        raise HTTPException(status_code=400, detail="ラップ検出が未完了です")
    lap_nums = sorted([l for l in df["lap"].unique() if l > 0])
    lap_data  = {int(l): df[df["lap"] == l].reset_index(drop=True) for l in lap_nums}
    lap_times = {}
    for l, ld in lap_data.items():
        if "time_sec" in ld.columns and len(ld) > 1:
            lap_times[l] = round(float(ld["time_sec"].max() - ld["time_sec"].min()), 3)
    if not lap_times:
        raise HTTPException(status_code=400, detail="有効なラップデータがありません")
    return df, lap_data, lap_times

# ============================================================
# エンドポイント
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Race Analyzer API",
        "version": "4.0.0",
        "sessions": len(SESSIONS),
        "cors_origins": ALLOWED_ORIGINS,
    }

@app.get("/health")
def health():
    """Railway ヘルスチェック用"""
    return {"status": "healthy", "sessions": len(SESSIONS)}

# ── Upload ──────────────────────────────────────────────────
class SessionInfo(BaseModel):
    session_id: str
    row_count:  int
    columns:    list
    guess:      dict
    preview:    list

@app.post("/api/upload", response_model=SessionInfo)
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSVファイルのみ対応しています")
    raw = await file.read()
    if len(raw) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="50MB以下のファイルを使用してください")
    try:
        df, encoding = detect_encoding_and_load(raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    df = auto_rename(df)
    columns = df.columns.tolist()
    session_id = str(uuid.uuid4())
    save_session(session_id, {
        "raw_df":     df,
        "proc_df":    None,
        "col_map":    None,
        "filename":   file.filename,
        "encoding":   encoding,
        "created_ts": time.time(),
    })
    return SessionInfo(
        session_id=session_id,
        row_count=len(df),
        columns=columns,
        guess=guess_column_mapping(columns),
        preview=df_to_json_safe(df.head(10)),
    )

# ── Column Map ──────────────────────────────────────────────
class ColumnMapRequest(BaseModel):
    session_id: str
    col_map:    dict

@app.post("/api/column-map")
def apply_column_map(req: ColumnMapRequest):
    session = get_session(req.session_id)
    if not req.col_map.get("time"):
        raise HTTPException(status_code=422, detail="Time列の指定は必須です")
    try:
        df_proc = preprocess(session["raw_df"].copy(), req.col_map)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"前処理エラー: {e}")
    session["col_map"] = req.col_map
    session["proc_df"] = df_proc
    save_session(req.session_id, session)

    summary = {
        "row_count":     len(df_proc),
        "duration_sec":  None,
        "speed_max":     None,
        "speed_mean":    None,
        "gps_available": "lat" in df_proc.columns and "lon" in df_proc.columns,
    }
    if "time_sec" in df_proc.columns:
        t = df_proc["time_sec"].dropna()
        summary["duration_sec"] = round(float(t.max() - t.min()), 2)
    if "speed_kmh" in df_proc.columns:
        spd = df_proc["speed_kmh"].dropna()
        summary["speed_max"]  = round(float(spd.max()), 1)
        summary["speed_mean"] = round(float(spd.mean()), 1)
    return {"session_id": req.session_id, "status": "ok", "summary": summary,
            "preview": df_to_json_safe(df_proc.head(10))}

# ── GPS Track ───────────────────────────────────────────────
@app.get("/api/session/{session_id}/track")
def get_gps_track(session_id: str):
    session = get_session(session_id)
    df = session.get("proc_df")
    if df is None:
        raise HTTPException(status_code=400, detail="前処理未完了")
    if "lat" not in df.columns or "lon" not in df.columns:
        raise HTTPException(status_code=422, detail="GPS列がありません")
    step = max(1, len(df) // 3000)
    track = []
    for i in range(0, len(df), step):
        row = df.iloc[i]
        track.append({
            "lat": round(float(row["lat"]), 7),
            "lon": round(float(row["lon"]), 7),
            "spd": round(float(row["speed_kmh"]), 1) if "speed_kmh" in df.columns else 0,
            "idx": i,
        })
    return {
        "session_id": session_id,
        "track":      track,
        "center":     {"lat": float(df["lat"].mean()), "lon": float(df["lon"].mean())},
        "total_rows": len(df),
    }

# ── Detect Laps ─────────────────────────────────────────────
from lap_detector import detect_laps, LapResult

class DetectLapsRequest(BaseModel):
    session_id:    str
    sf_lat:        float
    sf_lon:        float
    force_retrain: bool = False

@app.post("/api/detect-laps")
def api_detect_laps(req: DetectLapsRequest):
    session = get_session(req.session_id)
    df = session.get("proc_df")
    if df is None:
        raise HTTPException(status_code=400, detail="前処理未完了")
    if "lat" not in df.columns or "lon" not in df.columns:
        raise HTTPException(status_code=422, detail="GPS列がありません")
    saved_model = None if req.force_retrain else session.get("lap_model_bytes")
    try:
        result: LapResult = detect_laps(
            df=df, sf_lat=req.sf_lat, sf_lon=req.sf_lon,
            session_model=saved_model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ラップ検出エラー: {e}")

    if result.model_bytes:
        session["lap_model_bytes"] = result.model_bytes
    df["lap"] = result.lap_col
    session["proc_df"] = df
    save_session(req.session_id, session)

    best_lap = min(result.laps, key=lambda l: l.time_sec, default=None)
    return {
        "session_id": req.session_id,
        "method":     result.method,
        "ai_score":   round(result.ai_score, 4) if result.ai_score else None,
        "sf_point":   {"lat": result.sf_point[0], "lon": result.sf_point[1]},
        "lap_count":  len(result.laps),
        "best_lap":   best_lap.lap if best_lap else None,
        "best_time":  _fmt_time(best_lap.time_sec) if best_lap else None,
        "laps":       [{"lap": l.lap, "time_sec": l.time_sec,
                        "time_fmt": _fmt_time(l.time_sec),
                        "start_idx": l.start_idx, "end_idx": l.end_idx,
                        "start_time": l.start_time, "end_time": l.end_time}
                       for l in result.laps],
        "gps_track":  result.gps_track,
        "crossings":  result.crossings,
    }

# ── Lap Summary ─────────────────────────────────────────────
@app.get("/api/session/{session_id}/laps")
def get_lap_summary(session_id: str):
    df, lap_data, lap_times = _get_lap_data(session_id)
    best_lap = min(lap_times, key=lambda l: lap_times[l])
    best_t   = lap_times[best_lap]
    return {
        "session_id": session_id,
        "laps":       [{"lap": l, "time_sec": t, "time_fmt": _fmt_time(t),
                        "delta": round(t - best_t, 3), "is_best": l == best_lap}
                       for l, t in sorted(lap_times.items())],
        "best_lap":   best_lap,
        "best_time":  _fmt_time(best_t),
        "lap_count":  len(lap_times),
    }

# ── Analysis Chart ──────────────────────────────────────────
from analysis_engine import compute_chart, compute_ai, generate_pdf_bytes, CHART_FUNCS, AI_FUNCS

class ChartRequest(BaseModel):
    session_id:    str
    chart_id:      str
    selected_laps: list = []
    best_lap:      Optional[int] = None

@app.post("/api/analysis/chart")
def api_chart(req: ChartRequest):
    df, lap_data, lap_times = _get_lap_data(req.session_id)
    selected = req.selected_laps if req.selected_laps else list(lap_times.keys())
    selected = [l for l in selected if l in lap_times]
    if not selected:
        raise HTTPException(status_code=422, detail="有効なラップが選択されていません")
    best_lap = req.best_lap or min(lap_times, key=lambda l: lap_times[l])
    try:
        result = compute_chart(req.chart_id, df, lap_data, lap_times, best_lap, selected)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"チャート生成エラー: {e}")
    return {"session_id": req.session_id, "chart_id": req.chart_id, "best_lap": best_lap, **result}

# ── Analysis AI ─────────────────────────────────────────────
class AIRequest(BaseModel):
    session_id: str
    ai_id:      str
    best_lap:   Optional[int] = None

@app.post("/api/analysis/ai")
def api_ai(req: AIRequest):
    df, lap_data, lap_times = _get_lap_data(req.session_id)
    best_lap = req.best_lap or min(lap_times, key=lambda l: lap_times[l])
    try:
        result = compute_ai(req.ai_id, df, lap_data, lap_times, best_lap)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI解析エラー: {e}")
    return {"session_id": req.session_id, "ai_id": req.ai_id, "best_lap": best_lap, **result}

# ── PDF ─────────────────────────────────────────────────────
class PDFRequest(BaseModel):
    session_id:    str
    selected_laps: list = []
    best_lap:      Optional[int] = None

@app.post("/api/analysis/pdf")
def api_pdf(req: PDFRequest):
    df, lap_data, lap_times = _get_lap_data(req.session_id)
    selected = req.selected_laps if req.selected_laps else list(lap_times.keys())
    selected = [l for l in selected if l in lap_times]
    best_lap = req.best_lap or min(lap_times, key=lambda l: lap_times[l])
    pdf_bytes, err = generate_pdf_bytes(df, lap_data, lap_times, best_lap, selected)
    if err:
        raise HTTPException(status_code=500, detail=err)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=race_report.pdf"},
    )

# ── Session Cleanup ─────────────────────────────────────────
@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    with _sessions_lock:
        if session_id in SESSIONS:
            del SESSIONS[session_id]
    return {"status": "deleted"}
