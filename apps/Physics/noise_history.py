# -*- coding: utf-8 -*-
"""SQLite-backed long-term store for the Noise app."""
from __future__ import annotations

import csv
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import numpy as np


class NoiseHistory:
    """Tiny SQLite-backed long-term store: one aggregate row per second."""

    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
                ts REAL PRIMARY KEY,
                z_db REAL NOT NULL,
                a_db REAL NOT NULL,
                peak_db REAL NOT NULL,
                alpha REAL
            )
            """
        )
        # Migrate databases created before the 1/f slope column existed.
        columns = [row[1] for row in self.conn.execute("PRAGMA table_info(samples)")]
        if "alpha" not in columns:
            self.conn.execute("ALTER TABLE samples ADD COLUMN alpha REAL")
        self.conn.commit()

    def add(self, ts, z_db, a_db, peak_db, alpha=None):
        self.conn.execute(
            "INSERT OR REPLACE INTO samples(ts, z_db, a_db, peak_db, alpha) VALUES(?,?,?,?,?)",
            (float(ts), float(z_db), float(a_db), float(peak_db),
             None if alpha is None or not np.isfinite(alpha) else float(alpha)),
        )
        self.conn.commit()

    def summary(self, days=7):
        since = time.time() - days * 86400
        row = self.conn.execute(
            "SELECT COUNT(*), AVG(a_db), MAX(a_db), AVG(alpha) FROM samples WHERE ts >= ?",
            (since,),
        ).fetchone()
        count, avg, peak, alpha = row
        return {
            "count": int(count or 0),
            "avg": float(avg) if avg is not None else float("nan"),
            "peak": float(peak) if peak is not None else float("nan"),
            "alpha": float(alpha) if alpha is not None else float("nan"),
        }

    def hourly_profile(self, days=7):
        since = time.time() - days * 86400
        rows = self.conn.execute(
            """
            SELECT CAST(strftime('%H', ts, 'unixepoch', 'localtime') AS INTEGER) AS hour,
                   AVG(a_db)
            FROM samples
            WHERE ts >= ?
            GROUP BY hour
            ORDER BY hour
            """,
            (since,),
        ).fetchall()
        values = np.full(24, np.nan, dtype=np.float64)
        for hour, avg in rows:
            if 0 <= int(hour) < 24 and avg is not None:
                values[int(hour)] = float(avg)
        return values

    def export_csv(self, path: Path):
        path = Path(path)
        rows = self.conn.execute(
            "SELECT ts, z_db, a_db, peak_db, alpha FROM samples ORDER BY ts"
        ).fetchall()
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time", "z_db", "a_db", "peak_db", "alpha"])
            for ts, z_db, a_db, peak_db, alpha in rows:
                writer.writerow([
                    datetime.fromtimestamp(ts).isoformat(sep=" ", timespec="seconds"),
                    f"{z_db:.2f}", f"{a_db:.2f}", f"{peak_db:.2f}",
                    "" if alpha is None else f"{alpha:.3f}",
                ])
        return len(rows)

    def close(self):
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass
