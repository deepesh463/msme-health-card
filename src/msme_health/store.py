"""Tiny SQLite store for ingested data, consents and generated cards.

In production this would be Postgres (records) plus a feature store; SQLite keeps the demo
self-contained.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT

DB_PATH = ROOT / "data" / "app.db"


class Store:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS msme_data (msme_id TEXT PRIMARY KEY, data TEXT, updated_at TEXT);
                CREATE TABLE IF NOT EXISTS health_cards (msme_id TEXT PRIMARY KEY, card TEXT, scored_at TEXT);
                CREATE TABLE IF NOT EXISTS consents (consent_id TEXT PRIMARY KEY, msme_id TEXT, artefact TEXT, created_at TEXT);
                CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT, msme_id TEXT, event TEXT, detail TEXT, at TEXT);
                """
            )

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _one(self, sql, args):
        with self.lock:
            r = self.conn.execute(sql, args).fetchone()
        return r

    def _exec(self, sql, args):
        with self.lock:
            self.conn.execute(sql, args)
            self.conn.commit()

    def get_data(self, msme_id):
        r = self._one("SELECT data FROM msme_data WHERE msme_id=?", (msme_id,))
        return json.loads(r[0]) if r else None

    def put_data(self, msme_id, data: dict):
        self._exec("INSERT OR REPLACE INTO msme_data VALUES (?,?,?)", (msme_id, json.dumps(data), self.now()))

    def get_card(self, msme_id):
        r = self._one("SELECT card, scored_at FROM health_cards WHERE msme_id=?", (msme_id,))
        return (json.loads(r[0]), r[1]) if r else (None, None)

    def put_card(self, msme_id, card: dict) -> str:
        ts = self.now()
        self._exec("INSERT OR REPLACE INTO health_cards VALUES (?,?,?)", (msme_id, json.dumps(card, default=str), ts))
        return ts

    def put_consent(self, consent_id, msme_id, artefact: dict):
        self._exec("INSERT OR REPLACE INTO consents VALUES (?,?,?,?)", (consent_id, msme_id, json.dumps(artefact), self.now()))

    def get_consent(self, consent_id):
        r = self._one("SELECT artefact FROM consents WHERE consent_id=?", (consent_id,))
        return json.loads(r[0]) if r else None

    def audit(self, msme_id, event, detail: dict | None = None):
        self._exec("INSERT INTO audit_log (msme_id, event, detail, at) VALUES (?,?,?,?)",
                   (msme_id, event, json.dumps(detail or {}, default=str), self.now()))
