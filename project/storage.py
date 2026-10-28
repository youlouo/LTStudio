# project/storage.py
from __future__ import annotations
import json
import shutil
import sqlite3
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class ProjectMetadata:
    project_id: str
    title: str
    speaker: str = ""
    description: str = ""
    folder: str = "Общие"
    tags: Optional[List[str]] = None
    is_favorite: bool = False
    sample_rate: int = 48000
    duration_sec: float = 0.0
    notes: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        if self.tags is None:
            self.tags = []
        now = time.time()
        if self.created_at == 0.0:
            self.created_at = now
        if self.updated_at == 0.0:
            self.updated_at = now


class ProjectRepository:
    """SQLite менеджер проектов LTStudio с поддержкой создания, загрузки, автосохранения и удаления."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    speaker TEXT DEFAULT '',
                    description TEXT DEFAULT '',
                    folder TEXT DEFAULT 'Общие',
                    tags_json TEXT DEFAULT '[]',
                    is_favorite INTEGER DEFAULT 0,
                    sample_rate INTEGER DEFAULT 48000,
                    duration_sec REAL DEFAULT 0.0,
                    notes TEXT DEFAULT '',
                    edl_json TEXT DEFAULT '{}',
                    dsp_json TEXT DEFAULT '{}',
                    transcript_json TEXT DEFAULT '[]',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_projects_folder ON projects(folder);
                CREATE INDEX IF NOT EXISTS idx_projects_updated ON projects(updated_at DESC);
                """
            )

    def save_project(self, meta: ProjectMetadata, edl_dict: Dict[str, Any],
                     dsp_dict: Dict[str, Any], transcript_list: List[Dict[str, Any]]) -> None:
        meta.updated_at = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO projects (
                    project_id, title, speaker, description, folder, tags_json,
                    is_favorite, sample_rate, duration_sec, notes,
                    edl_json, dsp_json, transcript_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    title=excluded.title,
                    speaker=excluded.speaker,
                    description=excluded.description,
                    folder=excluded.folder,
                    tags_json=excluded.tags_json,
                    is_favorite=excluded.is_favorite,
                    sample_rate=excluded.sample_rate,
                    duration_sec=excluded.duration_sec,
                    notes=excluded.notes,
                    edl_json=excluded.edl_json,
                    dsp_json=excluded.dsp_json,
                    transcript_json=excluded.transcript_json,
                    updated_at=excluded.updated_at;
                """,
                (
                    meta.project_id,
                    meta.title,
                    meta.speaker,
                    meta.description,
                    meta.folder,
                    json.dumps(meta.tags, ensure_ascii=False),
                    int(meta.is_favorite),
                    meta.sample_rate,
                    meta.duration_sec,
                    meta.notes,
                    json.dumps(edl_dict, ensure_ascii=False),
                    json.dumps(dsp_dict, ensure_ascii=False),
                    json.dumps(transcript_list, ensure_ascii=False),
                    meta.created_at,
                    meta.updated_at,
                ),
            )

    def load_project_full(self, project_id: str) -> Optional[Tuple[ProjectMetadata, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]]:
        """Загружает метаданные, состояние дорожек, цепочку эффектов и транскрипт проекта."""
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,)).fetchone()
            if not r:
                return None
            meta = ProjectMetadata(
                project_id=r["project_id"],
                title=r["title"],
                speaker=r["speaker"],
                description=r["description"],
                folder=r["folder"],
                tags=json.loads(r["tags_json"]),
                is_favorite=bool(r["is_favorite"]),
                sample_rate=r["sample_rate"],
                duration_sec=r["duration_sec"],
                notes=r["notes"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            edl_dict = json.loads(r["edl_json"] or "{}")
            dsp_dict = json.loads(r["dsp_json"] or "{}")
            transcript_list = json.loads(r["transcript_json"] or "[]")
            return meta, edl_dict, dsp_dict, transcript_list

    def delete_project(self, project_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
            return cur.rowcount > 0

    def search_projects(self, query: str = "", folder: Optional[str] = None,
                        only_favorites: bool = False) -> List[ProjectMetadata]:
        sql = "SELECT * FROM projects WHERE 1=1"
        params: List[Any] = []
        if query:
            sql += " AND (title LIKE ? OR speaker LIKE ? OR description LIKE ? OR tags_json LIKE ?)"
            q = f"%{query}%"
            params.extend([q, q, q, q])
        if folder:
            sql += " AND folder = ?"
            params.append(folder)
        if only_favorites:
            sql += " AND is_favorite = 1"
        sql += " ORDER BY updated_at DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                ProjectMetadata(
                    project_id=r["project_id"],
                    title=r["title"],
                    speaker=r["speaker"],
                    description=r["description"],
                    folder=r["folder"],
                    tags=json.loads(r["tags_json"]),
                    is_favorite=bool(r["is_favorite"]),
                    sample_rate=r["sample_rate"],
                    duration_sec=r["duration_sec"],
                    notes=r["notes"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
                for r in rows
            ]

    def export_project_bundle(self, project_id: str, source_files: List[Path],
                              target_archive: Path) -> Path:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,)).fetchone()
            if not row:
                raise KeyError(f"Проект {project_id} не найден")
            manifest = dict(row)

        target_archive = target_archive.with_suffix(".ltproj")
        with zipfile.ZipFile(target_archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            for src in source_files:
                if src.exists():
                    zf.write(src, arcname=f"sources/{src.name}")
        return target_archive

    def import_project_bundle(self, archive_path: Path, extract_sources_dir: Path) -> str:
        extract_sources_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as zf:
            manifest = json.loads(zf.read("project_manifest.json").decode("utf-8"))
            for member in zf.namelist():
                if member.startswith("sources/") and not member.endswith("/"):
                    dest = extract_sources_dir / Path(member).name
                    with zf.open(member) as src_f, open(dest, "wb") as dst_f:
                        shutil.copyfileobj(src_f, dst_f)

        meta = ProjectMetadata(
            project_id=manifest["project_id"],
            title=manifest["title"],
            speaker=manifest.get("speaker", ""),
            description=manifest.get("description", ""),
            folder=manifest.get("folder", "Импортированные"),
            tags=json.loads(manifest.get("tags_json", "[]")),
            is_favorite=bool(manifest.get("is_favorite", 0)),
            sample_rate=manifest.get("sample_rate", 48000),
            duration_sec=manifest.get("duration_sec", 0.0),
            notes=manifest.get("notes", ""),
        )
        self.save_project(
            meta,
            json.loads(manifest.get("edl_json", "{}")),
            json.loads(manifest.get("dsp_json", "{}")),
            json.loads(manifest.get("transcript_json", "[]")),
        )
        return meta.project_id