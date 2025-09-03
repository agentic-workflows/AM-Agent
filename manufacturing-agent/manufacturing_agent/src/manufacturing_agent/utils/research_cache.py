"""
Research Cache Utility

SQLite-backed cache for literature review results keyed by a stable prompt hash.
"""

import sqlite3
import json
import time
import os
from pathlib import Path
from typing import Optional, Dict, Any


class ResearchCache:
    """Simple SQLite cache for research results keyed by prompt hash."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_directory_and_schema()

    def _ensure_directory_and_schema(self) -> None:
        """Ensure database directory exists and create schema if needed."""
        try:
            # Create directory if it doesn't exist
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            # Create schema
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS research_cache (
                      prompt_hash TEXT PRIMARY KEY,
                      prompt_json TEXT NOT NULL,
                      model TEXT,
                      result_json TEXT NOT NULL,
                      citations_json TEXT NOT NULL,
                      created_at REAL NOT NULL
                    )
                    """
                )
                conn.commit()
                
        except Exception as e:
            print(f"⚠️  Warning: Failed to initialize research cache database at {self.db_path}: {e}")
            print("   Research caching will be disabled, but the system will continue to work.")
            # Set a flag to disable caching operations
            self._cache_disabled = True
            return
            
        self._cache_disabled = False

    def get_by_prompt(self, prompt_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached research result by prompt hash."""
        if getattr(self, '_cache_disabled', False):
            return None
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM research_cache WHERE prompt_hash = ?",
                    (prompt_hash,),
                ).fetchone()
                if not row:
                    return None
                return {
                    "prompt_hash": row["prompt_hash"],
                    "prompt_json": row["prompt_json"],
                    "model": row["model"],
                    "result_json": json.loads(row["result_json"]),
                    "citations_json": json.loads(row["citations_json"]),
                    "created_at": row["created_at"],
                }
        except Exception as e:
            print(f"Warning: Failed to read from research cache: {e}")
            return None

    def put_by_prompt(
        self,
        *,
        prompt_hash: str,
        prompt_json: str,
        result_json: Dict[str, Any],
        citations_json: list,
        model: str,
    ) -> None:
        """Store research result in cache."""
        if getattr(self, '_cache_disabled', False):
            return
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO research_cache
                      (prompt_hash, prompt_json, model, result_json, citations_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        prompt_hash,
                        prompt_json,
                        model,
                        json.dumps(result_json, ensure_ascii=False),
                        json.dumps(citations_json, ensure_ascii=False),
                        time.time(),
                    ),
                )
                conn.commit()
        except Exception as e:
            print(f"Warning: Failed to write to research cache: {e}")


