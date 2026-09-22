"""DuckDB persistence: spans, traffic log, eval runs and results.

DuckDB is single-file, serverless, and SQL -- the whole analytics story of
this project is a handful of queries over these tables.
"""
from __future__ import annotations

import json
import os

import duckdb

SCHEMA = [
    "CREATE SEQUENCE IF NOT EXISTS traffic_id_seq",
    "CREATE SEQUENCE IF NOT EXISTS eval_run_id_seq",
    """CREATE TABLE IF NOT EXISTS spans (
        trace_id VARCHAR, span_id VARCHAR, parent_id VARCHAR,
        name VARCHAR, start_ns BIGINT, end_ns BIGINT, attributes JSON
    )""",
    """CREATE TABLE IF NOT EXISTS traffic (
        id INTEGER DEFAULT nextval('traffic_id_seq'), trace_id VARCHAR, ts TIMESTAMP, day INTEGER,
        query VARCHAR, response VARCHAR, model VARCHAR,
        prompt_tokens INTEGER, completion_tokens INTEGER,
        cost_usd DOUBLE, latency_ms DOUBLE,
        n_docs INTEGER, top_retrieval_score DOUBLE
    )""",
    """CREATE TABLE IF NOT EXISTS eval_runs (
        id INTEGER DEFAULT nextval('eval_run_id_seq'), suite VARCHAR, suite_version INTEGER, model VARCHAR,
        started_at TIMESTAMP, passed BOOLEAN, summary JSON
    )""",
    """CREATE TABLE IF NOT EXISTS eval_results (
        run_id INTEGER, case_id VARCHAR, judge VARCHAR,
        score DOUBLE, passed BOOLEAN, detail JSON, regression BOOLEAN
    )""",
    """CREATE TABLE IF NOT EXISTS reports (
        kind VARCHAR, created_at TIMESTAMP, payload JSON
    )""",
]


def _rows_to_dicts(cur, columns):
    return [dict(zip(columns, r)) for r in cur.fetchall()]


class Storage:
    def __init__(self, path: str = "data/eval.duckdb"):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.con = duckdb.connect(path)
        for ddl in SCHEMA:
            self.con.execute(ddl)

    # -- writers -----------------------------------------------------------

    def insert_span(self, span):
        self.con.execute(
            "INSERT INTO spans VALUES (?,?,?,?,?,?,?)",
            (
                span.trace_id,
                span.span_id,
                span.parent_id,
                span.name,
                span.start_ns,
                span.end_ns,
                json.dumps(span.attributes, default=str),
            ),
        )

    def insert_traffic(self, row: dict):
        self.con.execute(
            """INSERT INTO traffic
            (trace_id, ts, day, query, response, model, prompt_tokens,
             completion_tokens, cost_usd, latency_ms, n_docs, top_retrieval_score)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row["trace_id"],
                row["ts"],
                row.get("day"),
                row["query"],
                row["response"],
                row["model"],
                row["prompt_tokens"],
                row["completion_tokens"],
                row["cost_usd"],
                row["latency_ms"],
                row.get("n_docs"),
                row.get("top_retrieval_score"),
            ),
        )

    def insert_eval_run(self, suite, version, model, passed, summary: dict) -> int:
        cur = self.con.execute(
            "INSERT INTO eval_runs VALUES (DEFAULT,?,?,?,?,?,?) RETURNING id",
            (suite, version, model, summary.get("started_at"), passed, json.dumps(summary, default=str)),
        )
        return cur.fetchone()[0]

    def insert_eval_result(self, run_id, case_id, judge, score, passed, detail: dict, regression: bool):
        self.con.execute(
            "INSERT INTO eval_results VALUES (?,?,?,?,?,?,?)",
            (run_id, case_id, judge, score, passed, json.dumps(detail, default=str), regression),
        )

    def insert_report(self, kind: str, payload: dict):
        self.con.execute(
            "INSERT INTO reports VALUES (?,?,?)",
            (kind, payload.get("created_at"), json.dumps(payload, default=str)),
        )

    # -- readers ------------------------------------------------------------

    def latest_passing_run(self, suite: str, version: int):
        r = self.con.execute(
            """SELECT id, model, started_at FROM eval_runs
               WHERE suite=? AND suite_version=? AND passed=TRUE
               ORDER BY id DESC LIMIT 1""",
            [suite, version],
        ).fetchone()
        return {"id": r[0], "model": r[1], "started_at": str(r[2])} if r else None

    def run_results(self, run_id: int):
        cur = self.con.execute(
            "SELECT case_id, judge, score, passed, detail, regression FROM eval_results WHERE run_id=?",
            [run_id],
        )
        return [
            {"case_id": r[0], "judge": r[1], "score": r[2], "passed": r[3], "detail": json.loads(r[4]), "regression": r[5]}
            for r in cur.fetchall()
        ]

    def eval_history(self):
        cur = self.con.execute(
            "SELECT id, suite, suite_version, model, started_at, passed, summary FROM eval_runs ORDER BY id"
        )
        return [
            {"id": r[0], "suite": r[1], "version": r[2], "model": r[3], "started_at": str(r[4]),
             "passed": r[5], "summary": json.loads(r[6])}
            for r in cur.fetchall()
        ]

    def traffic_metrics_between_days(self, lo: int, hi: int):
        cur = self.con.execute(
            """SELECT latency_ms, completion_tokens, cost_usd, length(response), response
                FROM traffic WHERE day BETWEEN ? AND ?""",
            [lo, hi],
        )
        rows = cur.fetchall()
        return {
            "latency_ms": [r[0] for r in rows],
            "completion_tokens": [r[1] for r in rows],
            "cost_usd": [r[2] for r in rows],
            "response_words": [r[3] for r in rows],
            "responses": [r[4] for r in rows],
        }

    def daily_series(self):
        cur = self.con.execute(
            """SELECT day, count(*), avg(latency_ms), sum(cost_usd), avg(top_retrieval_score)
                FROM traffic GROUP BY day ORDER BY day"""
        )
        cols = ["day", "requests", "avg_latency_ms", "total_cost_usd", "avg_retrieval_score"]
        return _rows_to_dicts(cur, cols)

    def recent_traces(self, limit: int = 50):
        cur = self.con.execute(
            """SELECT trace_id, day, query, model, latency_ms, cost_usd,
                      completion_tokens, top_retrieval_score
                FROM traffic ORDER BY id DESC LIMIT ?""",
            [limit],
        )
        cols = ["trace_id", "day", "query", "model", "latency_ms", "cost_usd",
                "completion_tokens", "top_retrieval_score"]
        return _rows_to_dicts(cur, cols)

    def latest_report(self, kind: str):
        r = self.con.execute(
            "SELECT payload FROM reports WHERE kind=? ORDER BY created_at DESC LIMIT 1", [kind]
        ).fetchone()
        return json.loads(r[0]) if r else None
