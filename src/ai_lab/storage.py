"""Versioned, persistent passage catalog using the Python standard library."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from .rag import Passage, chunk_text


class DocumentStore:
    def __init__(self, database: str | Path):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS documents (
                source TEXT PRIMARY KEY, sha256 TEXT NOT NULL
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS passages (
                source TEXT NOT NULL, chunk INTEGER NOT NULL, text TEXT NOT NULL,
                PRIMARY KEY (source, chunk),
                FOREIGN KEY (source) REFERENCES documents(source) ON DELETE CASCADE
            )""")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def sync(self, directory: str | Path) -> dict[str, int]:
        """Synchronize .md/.txt files; unchanged files retain their rows."""
        root = Path(directory)
        if not root.is_dir():
            raise ValueError(f"Document directory does not exist: {root}")
        files = sorted(path for path in root.rglob("*") if path.is_file()
                       and path.suffix.lower() in {".md", ".txt"})
        updated = deleted = 0
        with self._connect() as connection:
            existing = dict(connection.execute("SELECT source, sha256 FROM documents"))
            seen = set()
            for path in files:
                source = path.relative_to(root).as_posix()
                seen.add(source)
                content = path.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                if existing.get(source) == digest:
                    continue
                chunks = chunk_text(content.decode("utf-8"))
                connection.execute("DELETE FROM documents WHERE source=?", (source,))
                if chunks:
                    connection.execute("INSERT INTO documents VALUES (?, ?)", (source, digest))
                    connection.executemany("INSERT INTO passages VALUES (?, ?, ?)",
                                           ((source, i, text) for i, text in enumerate(chunks, 1)))
                updated += 1
            for source in existing.keys() - seen:
                connection.execute("DELETE FROM documents WHERE source=?", (source,))
                deleted += 1
            count = connection.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        return {"updated": updated, "deleted": deleted, "passages": count}

    def passages(self) -> list[Passage]:
        with self._connect() as connection:
            rows = connection.execute("SELECT source, chunk, text FROM passages ORDER BY source, chunk")
            return [Passage(*row) for row in rows]
