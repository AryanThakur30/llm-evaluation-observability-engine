"""FastAPI dashboard + JSON API over the DuckDB store.

Run:   uvicorn api.app:app --port 8000     (from the project root)
Open:  http://localhost:8000
"""
from __future__ import annotations

import datetime
import json
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from engine.storage import Storage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("EVAL_DB", os.path.join(ROOT, "data", "demo_eval.duckdb"))

app = FastAPI(title="LLM Eval & Observability Engine")


def storage() -> Storage:
    return Storage(DB_PATH)


def _json_default(o):
    if isinstance(o, (datetime.datetime, datetime.date)):
        return str(o)
    return str(o)


@app.get("/")
def index():
    path = os.path.join(ROOT, "dashboard", "index.html")
    if not os.path.exists(path):
        return HTMLResponse("<h1>dashboard/index.html not found</h1>", status_code=404)
    with open(path) as f:
        return HTMLResponse(f.read())


@app.get("/api/summary")
def summary():
    s = storage()
    row = s.con.execute(
        """SELECT count(*), avg(latency_ms), sum(cost_usd), avg(top_retrieval_score)
            FROM traffic"""
    ).fetchone()
    n_runs = s.con.execute("SELECT count(*) FROM eval_runs").fetchone()[0]
    pass_rate = s.con.execute("SELECT avg(CASE WHEN passed THEN 1.0 ELSE 0 END) FROM eval_runs").fetchone()[0]
    return {
        "requests": row[0] or 0,
        "avg_latency_ms": round(row[1] or 0, 1),
        "total_cost_usd": round(row[2] or 0, 4),
        "avg_retrieval_score": round(row[3] or 0, 4),
        "eval_runs": n_runs,
        "suite_pass_rate": round(pass_rate or 0, 3),
    }


@app.get("/api/traffic")
def traffic():
    return storage().daily_series()


@app.get("/api/evals")
def evals():
    return storage().eval_history()


@app.get("/api/drift")
def drift():
    return storage().latest_report("drift") or {"drift_detected": False, "alerts": []}


@app.get("/api/canary")
def canary():
    return storage().latest_report("canary") or {}


@app.get("/api/traces")
def traces():
    return storage().recent_traces(25)
