"""Antigravity (agy) storage readers and writers.

Handles:
- ~/.gemini/config/projects/<uuid>.json
- ~/Library/Application Support/Antigravity/app_storage.json
- ~/.gemini/antigravity/conversations/<uuid>.db
- ~/.gemini/antigravity/conversation_summaries.db (summaries + FTS5 + parents)
- ~/.gemini/antigravity/agyhub_summaries_proto.pb
- ~/.gemini/antigravity/annotations/<uuid>.pbtxt
- ~/.gemini/antigravity/brain/<uuid>/ (transcripts, chunks, .user_uploaded, steps)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote, urlparse

from .agy_proto import (
    OUTSIDE_OF_PROJECT_ID,
    STEP_STATUS_DONE,
    STEP_TYPE_PLANNER_RESPONSE,
    STEP_TYPE_USER_INPUT,
    TRAJECTORY_SOURCE_CASCADE_CLIENT,
    TRAJECTORY_TYPE_CASCADE,
    WorkspaceInfo,
    decode_agyhub_summaries,
    encode_agyhub_summaries,
    encode_annotations_pbtxt,
    encode_cascade_trajectory_summary,
    encode_conversation_annotations,
    encode_trajectory_metadata,
)
from .safety import (
    BackupManager,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    load_migration_manifest,
    save_migration_manifest,
)

# --- Exact SQLite DDLs matching Antigravity ---

CONVERSATION_DB_DDL = """
CREATE TABLE IF NOT EXISTS `trajectory_meta` (
    `trajectory_id` text,
    `cascade_id` text,
    `trajectory_type` integer,
    `source` integer,
    PRIMARY KEY (`trajectory_id`)
);
CREATE TABLE IF NOT EXISTS `steps` (
    `idx` integer,
    `step_type` integer NOT NULL DEFAULT 0,
    `status` integer NOT NULL DEFAULT 0,
    `has_subtrajectory` numeric NOT NULL DEFAULT false,
    `metadata` blob,
    `error_details` blob,
    `permissions` blob,
    `task_details` blob,
    `render_info` blob,
    `step_payload` blob,
    `step_format` integer NOT NULL DEFAULT 0,
    PRIMARY KEY (`idx`)
);
CREATE INDEX IF NOT EXISTS `idx_steps_status` ON `steps`(`status`);
CREATE INDEX IF NOT EXISTS `idx_steps_step_type` ON `steps`(`step_type`);
CREATE TABLE IF NOT EXISTS `gen_metadata` (
    `idx` integer,
    `data` blob,
    `size` integer NOT NULL DEFAULT 0,
    PRIMARY KEY (`idx`)
);
CREATE TABLE IF NOT EXISTS `executor_metadata` (
    `idx` integer,
    `data` blob,
    PRIMARY KEY (`idx`)
);
CREATE TABLE IF NOT EXISTS `parent_references` (
    `idx` integer,
    `data` blob,
    PRIMARY KEY (`idx`)
);
CREATE TABLE IF NOT EXISTS `trajectory_metadata_blob` (
    `id` text DEFAULT "main",
    `data` blob,
    PRIMARY KEY (`id`)
);
CREATE TABLE IF NOT EXISTS `battle_mode_infos` (
    `idx` integer,
    `data` blob,
    PRIMARY KEY (`idx`)
);
"""

SUMMARIES_DB_DDL = """
CREATE TABLE IF NOT EXISTS `conversation_summaries` (
    `conversation_id` text,
    `title` text NOT NULL DEFAULT "",
    `preview` text NOT NULL DEFAULT "",
    `step_count` integer NOT NULL DEFAULT 0,
    `last_modified_time` datetime NOT NULL,
    `workspace_uris` text NOT NULL,
    `status` text NOT NULL DEFAULT "",
    `source` text NOT NULL DEFAULT "",
    `project_id` text NOT NULL DEFAULT "",
    `agent_name` text NOT NULL DEFAULT "",
    `parent_conversation_id` text NOT NULL DEFAULT "",
    `nesting_depth` integer NOT NULL DEFAULT 0,
    `battle_id` text NOT NULL DEFAULT "",
    `winning_conversation_id` text NOT NULL DEFAULT "",
    `not_fully_idle` numeric NOT NULL DEFAULT false,
    `killed` numeric NOT NULL DEFAULT false,
    `last_user_input_time` datetime NOT NULL,
    `last_user_input_step_index` integer NOT NULL DEFAULT -1,
    `app_data_dir` text NOT NULL DEFAULT "",
    `raw_summary` blob,
    `group_id` text NOT NULL DEFAULT "",
    PRIMARY KEY (`conversation_id`)
);
CREATE INDEX IF NOT EXISTS `idx_conversation_summaries_last_user_input_time`
    ON `conversation_summaries`(`last_user_input_time`);
CREATE INDEX IF NOT EXISTS `idx_conversation_summaries_last_modified_time`
    ON `conversation_summaries`(`last_modified_time`);
CREATE TABLE IF NOT EXISTS search_schema_version (
    version INTEGER PRIMARY KEY
);
CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
    conversation_id UNINDEXED,
    chunk_index UNINDEXED,
    chunk_first_step UNINDEXED,
    chunk_last_step UNINDEXED,
    last_modified_time UNINDEXED,
    step_count UNINDEXED,
    last_step_created_at UNINDEXED,
    archived UNINDEXED,
    user_step_byte_offsets UNINDEXED,
    agent_step_byte_offsets UNINDEXED,
    title,
    workspace,
    project,
    user_body,
    agent_body,
    tokenize = 'unicode61 remove_diacritics 2',
    prefix = '1 2 3'
);
CREATE TABLE IF NOT EXISTS conversation_parents (
    conversation_id TEXT PRIMARY KEY,
    parent_id TEXT NOT NULL
) WITHOUT ROWID;
"""

FTS_CHUNK_MAX_BYTES = 200_000

# Antigravity language_server (v2.34+) hard-codes a 500-file pruning cap on conversations/*.db
# (counting BOTH root conversations and separate subagent DBs by OS filesystem mtime).
AGY_HARD_CONVERSATION_LIMIT = 500
DEFAULT_RESERVE_SLOTS = 50

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def default_state_dir() -> Path:
    """Return ./migration_state inside the cloned repo, or CWD/migration_state when pip-installed."""
    if (PROJECT_ROOT / "pyproject.toml").exists():
        return PROJECT_ROOT / "migration_state"
    return Path.cwd() / "migration_state"


def default_app_storage_path() -> Path:
    """Return platform-appropriate Antigravity Electron app_storage.json path."""
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Antigravity" / "app_storage.json"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else (home / "AppData" / "Roaming")
        return base / "Antigravity" / "app_storage.json"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else (home / ".config")
    return base / "Antigravity" / "app_storage.json"


DEFAULT_STATE_DIR = default_state_dir()

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
_WIN_DRIVE_RE = re.compile(r"^[a-zA-Z]:[\\/]")
_URI_WIN_DRIVE_RE = re.compile(r"^/[a-zA-Z]:/")


def path_to_file_uri(path: str) -> str:
    """Convert an absolute filesystem path to a file:///... URI matching Antigravity's convention."""
    raw = (path or "").strip()
    if _WIN_DRIVE_RE.match(raw):
        norm = "/" + raw.replace("\\", "/")
    else:
        norm = os.path.normpath(raw).replace("\\", "/")
        if not norm.startswith("/"):
            norm = "/" + norm
    return "file://" + quote(norm, safe="/:@-._~!$&'()*+,;=")


def file_uri_to_path(uri: str) -> str:
    if uri.startswith("file://"):
        parsed = urlparse(uri)
        p = unquote(parsed.path)
        if _URI_WIN_DRIVE_RE.match(p):
            p = p[1:]
        return os.path.normpath(p)
    return os.path.normpath(uri)


def format_go_utc_datetime(ms: int) -> str:
    """Format epoch ms as Go/GORM SQLite UTC datetime string: 'YYYY-MM-DD HH:MM:SS.ffffff+00:00'."""
    if ms <= 0:
        return "0001-01-01 00:00:00+00:00"
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    if dt.microsecond:
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f+00:00")
    return dt.strftime("%Y-%m-%d %H:%M:%S+00:00")


def parse_go_utc_datetime(dt_str: str | None) -> Optional[float]:
    """Parse Go/GORM UTC datetime string back into Unix epoch seconds (or None if sentinel/invalid)."""
    if not dt_str:
        return None
    s = dt_str.strip()
    if s.startswith("0001-01-01"):
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def set_path_mtime_ms(path: Path, ts_ms: int) -> None:
    """Set a file or directory's OS atime/mtime to historical ts_ms (clamped to <= current wall clock)."""
    if ts_ms <= 0 or not path.exists():
        return
    ts_sec = min(ts_ms / 1000.0, time.time())
    os.utime(path, (ts_sec, ts_sec))


@dataclass
class AgyPaths:
    gemini_dir: Path
    app_storage_json: Path
    state_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.state_dir is None:
            try:
                if self.gemini_dir.resolve() == (Path.home() / ".gemini").resolve():
                    self.state_dir = default_state_dir()
                else:
                    self.state_dir = self.gemini_dir.parent / "migration_state"
            except Exception:
                self.state_dir = default_state_dir()

    @classmethod
    def default(cls) -> "AgyPaths":
        home = Path.home()
        return cls(
            gemini_dir=home / ".gemini",
            app_storage_json=default_app_storage_path(),
            state_dir=default_state_dir(),
        )

    @classmethod
    def custom(cls, root: Path) -> "AgyPaths":
        return cls(
            gemini_dir=root / ".gemini",
            app_storage_json=root / "app_storage.json",
            state_dir=root / "migration_state",
        )

    @property
    def antigravity_dir(self) -> Path:
        return self.gemini_dir / "antigravity"

    @property
    def projects_dir(self) -> Path:
        return self.gemini_dir / "config" / "projects"

    @property
    def conversations_dir(self) -> Path:
        return self.antigravity_dir / "conversations"

    @property
    def annotations_dir(self) -> Path:
        return self.antigravity_dir / "annotations"

    @property
    def brain_dir(self) -> Path:
        return self.antigravity_dir / "brain"

    @property
    def presence_dir(self) -> Path:
        return self.antigravity_dir / "presence"

    @property
    def summaries_db(self) -> Path:
        return self.antigravity_dir / "conversation_summaries.db"

    @property
    def hub_summaries_pb(self) -> Path:
        return self.antigravity_dir / "agyhub_summaries_proto.pb"

    @property
    def backups_dir(self) -> Path:
        assert self.state_dir is not None
        return self.state_dir / "backups"

    @property
    def migration_manifest(self) -> Path:
        assert self.state_dir is not None
        return self.state_dir / "manifest.json"

    @property
    def legacy_backups_dir(self) -> Path:
        return self.antigravity_dir / "codex_2_agy_backups"

    @property
    def legacy_migration_manifest(self) -> Path:
        return self.antigravity_dir / "codex_2_agy_manifest.json"


@dataclass
class AgyProject:
    id: str
    name: str
    raw: Dict[str, Any]
    is_sidecar: bool = False
    folder_paths: List[str] = field(default_factory=list)


def load_existing_agy_projects(paths: AgyPaths) -> Dict[str, AgyProject]:
    """Read all ~/.gemini/config/projects/*.json files."""
    projects: Dict[str, AgyProject] = {}
    if not paths.projects_dir.exists():
        return projects
    for p in sorted(paths.projects_dir.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            continue
        pid = raw.get("id") or p.stem
        name = raw.get("name") or ""
        is_sidecar = bool(raw.get("sidecar"))
        folders: List[str] = []
        resources = (raw.get("projectResources") or {}).get("resources") or []
        for res in resources:
            if not isinstance(res, dict):
                continue
            if "folderUri" in res and res["folderUri"]:
                folders.append(file_uri_to_path(res["folderUri"]))
            elif "gitFolder" in res and isinstance(res["gitFolder"], dict):
                furi = res["gitFolder"].get("folderUri")
                if furi:
                    folders.append(file_uri_to_path(furi))
        projects[pid] = AgyProject(
            id=pid,
            name=name,
            raw=raw,
            is_sidecar=is_sidecar,
            folder_paths=folders,
        )
    return projects


def load_existing_agy_disk_conversation_ids(paths: AgyPaths) -> set[str]:
    """Return set of conversation IDs that have an actual .db file in conversations/ (matches SqliteStore.List)."""
    cids: set[str] = set()
    if paths.conversations_dir.exists():
        for p in paths.conversations_dir.glob("*.db"):
            cids.add(p.stem)
    return cids


def open_agy_sqlite_ro(path: Path) -> sqlite3.Connection:
    """Open an Antigravity SQLite DB read-only while respecting active WAL journals."""
    try:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def load_existing_agy_conversation_ids(paths: AgyPaths) -> set[str]:
    """Return set of conversation IDs already present in Antigravity (disk .db OR summaries DB)."""
    cids = load_existing_agy_disk_conversation_ids(paths)
    if paths.summaries_db.exists():
        try:
            conn = open_agy_sqlite_ro(paths.summaries_db)
            try:
                for (cid,) in conn.execute("SELECT conversation_id FROM conversation_summaries"):
                    cids.add(cid)
            finally:
                conn.close()
        except Exception:
            pass
    return cids


@dataclass
class EncodedStep:
    idx: int
    step_type: int
    status: int
    created_at_ms: int
    metadata_bytes: bytes
    payload_bytes: bytes
    error_details_bytes: Optional[bytes] = None
    # Plaintext fields for FTS5 indexing & transcript generation
    fts_user_text: Optional[str] = None
    fts_agent_text: Optional[str] = None
    transcript_entry: Optional[Dict[str, Any]] = None
    large_output_file: Optional[Tuple[int, str]] = None  # (step_idx, full_text)


@dataclass
class ExtractedAsset:
    rel_path: str  # relative to brain/<cid>/, e.g. ".user_uploaded/img_001.png"
    data: bytes


@dataclass
class MappedConversation:
    conversation_id: str
    trajectory_id: str
    init_state_id: str
    title: str
    preview: str
    project_id: str
    project_name: str
    workspaces: List[WorkspaceInfo]
    created_at_ms: int
    last_modified_ms: int
    last_user_input_ms: int
    last_user_input_step_index: int
    steps: List[EncodedStep]
    assets: List[ExtractedAsset] = field(default_factory=list)
    archived: bool = False
    pinned: bool = False
    # Subagent metadata
    parent_conversation_id: str = ""
    root_conversation_id: str = ""
    nesting_depth: int = 0
    agent_name: str = ""
    subagent_role: str = ""
    subagent_prompt: str = ""


class AgyWriter:
    """Atomic, backed-up writer for Antigravity projects, conversations, summaries, and FTS5."""

    def __init__(
        self,
        paths: AgyPaths,
        backup: BackupManager,
        *,
        max_total_dbs: Optional[int] = AGY_HARD_CONVERSATION_LIMIT,
    ) -> None:
        self.paths = paths
        self.backup = backup
        self.max_total_dbs = max_total_dbs
        self._summaries_conn: Optional[sqlite3.Connection] = None
        self._hub_summaries: Dict[str, bytes] = {}
        self._hub_dirty = False
        self._live_disk_cids: set[str] = set()

    def open(self) -> None:
        self.paths.projects_dir.mkdir(parents=True, exist_ok=True)
        self.paths.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.paths.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.paths.brain_dir.mkdir(parents=True, exist_ok=True)
        self._live_disk_cids = load_existing_agy_disk_conversation_ids(self.paths)

        # Back up shared index files before opening for write
        self.backup.backup_file(self.paths.summaries_db)
        self.backup.backup_file(self.paths.hub_summaries_pb)
        self.backup.backup_file(self.paths.app_storage_json)
        self.backup.backup_file(self.paths.migration_manifest)

        # Open & ensure schema for conversation_summaries.db
        existed = self.paths.summaries_db.exists()
        conn = sqlite3.connect(str(self.paths.summaries_db))
        conn.executescript(SUMMARIES_DB_DDL)
        if not existed:
            conn.execute("PRAGMA user_version = 3;")
            conn.execute("INSERT OR IGNORE INTO search_schema_version(version) VALUES (3);")
            conn.commit()
            if os.name == "posix":
                os.chmod(self.paths.summaries_db, 0o644)
        self._summaries_conn = conn

        # Load existing agyhub_summaries_proto.pb so we never lose existing entries
        if self.paths.hub_summaries_pb.exists():
            self._hub_summaries = decode_agyhub_summaries(self.paths.hub_summaries_pb.read_bytes())
        else:
            # Seed from existing DB if pb didn't exist
            for cid, raw in conn.execute(
                "SELECT conversation_id, raw_summary FROM conversation_summaries WHERE raw_summary IS NOT NULL"
            ):
                self._hub_summaries[cid] = raw

    def close(self) -> None:
        if self._summaries_conn is not None:
            self._summaries_conn.commit()
            try:
                self._summaries_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except Exception:
                pass
            self._summaries_conn.close()
            self._summaries_conn = None
        if self._hub_dirty:
            atomic_write_bytes(
                self.paths.hub_summaries_pb,
                encode_agyhub_summaries(self._hub_summaries),
                mode=0o600,
            )
            self._hub_dirty = False

    def write_project(self, project_id: str, project_json: Dict[str, Any], *, is_new: bool) -> None:
        pfile = self.paths.projects_dir / f"{project_id}.json"
        if is_new:
            self.backup.record_created_project(project_id)
        else:
            self.backup.backup_file(pfile)
        atomic_write_json(pfile, project_json)

    def update_projects_order(self, new_project_ids: List[str]) -> None:
        """Append newly created project IDs into app_storage.json's projectsOrder without clobbering existing order."""
        if not new_project_ids:
            return
        if self.paths.app_storage_json.exists():
            with open(self.paths.app_storage_json, "r", encoding="utf-8") as f:
                storage = json.load(f)
        else:
            storage = {}
        raw_order = storage.get("projectsOrder", "[]")
        try:
            order = json.loads(raw_order) if isinstance(raw_order, str) else list(raw_order)
        except Exception:
            order = []
        existing = set(order)
        changed = False
        for pid in new_project_ids:
            if pid not in existing:
                order.append(pid)
                existing.add(pid)
                changed = True
        if changed:
            storage["projectsOrder"] = json.dumps(order, ensure_ascii=False)
            atomic_write_json(self.paths.app_storage_json, storage)

    def write_conversation(self, conv: MappedConversation, *, is_overwrite: bool = False) -> None:
        """Write conversation .db, brain/ files, annotation .pbtxt, summary DB row, FTS5 chunks, and hub pb."""
        assert self._summaries_conn is not None
        cid = conv.conversation_id
        db_path = self.paths.conversations_dir / f"{cid}.db"
        ann_path = self.paths.annotations_dir / f"{cid}.pbtxt"

        # Hard slot guard: never allow creating a new .db that exceeds max_total_dbs (e.g. 500 - reserve)
        if cid not in self._live_disk_cids and self.max_total_dbs is not None:
            if len(self._live_disk_cids) + 1 > self.max_total_dbs:
                raise RuntimeError(
                    f"Slot budget exceeded: writing {cid}.db would bring conversations/*.db to "
                    f"{len(self._live_disk_cids) + 1}, exceeding max_total_dbs={self.max_total_dbs} "
                    f"(Antigravity hard limit is {AGY_HARD_CONVERSATION_LIMIT})."
                )

        if is_overwrite or db_path.exists():
            self.backup.backup_file(db_path)
            self.backup.backup_file(ann_path)
            self.backup.backup_dir_tree(self.paths.brain_dir / cid)
            self.backup.record_overwritten_conversation(cid)
        else:
            self.backup.record_created_conversation(cid)

        # 1. Build TrajectoryMetadata & CascadeTrajectorySummary protobufs
        is_subagent = bool(conv.parent_conversation_id)
        root_cid = conv.root_conversation_id or cid
        traj_meta_blob = encode_trajectory_metadata(
            workspaces=conv.workspaces,
            created_at_ms=conv.created_at_ms,
            initialization_state_id=conv.init_state_id,
            root_conversation_id=root_cid,
            project_id=conv.project_id,
            parent_conversation_id=conv.parent_conversation_id,
            nesting_depth=conv.nesting_depth,
            subagent_type_name=conv.agent_name if is_subagent else "",
            subagent_role=conv.subagent_role if is_subagent else "",
            subagent_prompt=conv.subagent_prompt if is_subagent else "",
        )

        annotations_pb = (
            b""
            if is_subagent
            else encode_conversation_annotations(
                title=conv.title,
                last_user_view_time_ms=conv.last_modified_ms,
                archived=conv.archived,
                pinned=conv.pinned,
            )
        )

        raw_summary = encode_cascade_trajectory_summary(
            summary="" if is_subagent else conv.title,
            step_count=len(conv.steps),
            last_modified_ms=conv.last_modified_ms,
            trajectory_id=conv.trajectory_id,
            created_ms=conv.created_at_ms,
            workspaces=conv.workspaces,
            last_user_input_ms=0 if is_subagent else conv.last_user_input_ms,
            last_user_input_step_index=-1 if is_subagent else conv.last_user_input_step_index,
            annotations_bytes=annotations_pb,
            trajectory_metadata_bytes=traj_meta_blob,
            is_subagent=is_subagent,
        )

        # 2. Write conversations/<cid>.db via temp file + atomic replace
        tmp_db = self.paths.conversations_dir / f".{cid}.db.tmp"
        if tmp_db.exists():
            tmp_db.unlink()
        for side in ("-wal", "-shm"):
            sp = Path(str(tmp_db) + side)
            if sp.exists():
                sp.unlink()

        c_conn = sqlite3.connect(str(tmp_db))
        try:
            c_conn.execute("PRAGMA page_size = 32768;")
            c_conn.execute("PRAGMA journal_mode = WAL;")
            c_conn.execute("PRAGMA synchronous = NORMAL;")
            c_conn.execute("PRAGMA user_version = 1;")
            c_conn.executescript(CONVERSATION_DB_DDL)
            c_conn.execute(
                "INSERT INTO trajectory_meta(trajectory_id, cascade_id, trajectory_type, source) VALUES (?, ?, ?, ?)",
                (
                    conv.trajectory_id,
                    cid,
                    TRAJECTORY_TYPE_CASCADE,
                    TRAJECTORY_SOURCE_CASCADE_CLIENT,
                ),
            )
            c_conn.execute(
                "INSERT INTO trajectory_metadata_blob(id, data) VALUES ('main', ?)",
                (traj_meta_blob,),
            )
            c_conn.executemany(
                """
                INSERT INTO steps(
                    idx, step_type, status, has_subtrajectory,
                    metadata, error_details, permissions, task_details, render_info,
                    step_payload, step_format
                ) VALUES (?, ?, ?, 0, ?, ?, NULL, NULL, NULL, ?, 0)
                """,
                [
                    (
                        s.idx,
                        s.step_type,
                        s.status,
                        s.metadata_bytes,
                        s.error_details_bytes,
                        s.payload_bytes,
                    )
                    for s in conv.steps
                ],
            )
            c_conn.commit()
            c_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        finally:
            c_conn.close()

        # Clean up temp WAL/SHM sidecars after checkpoint(TRUNCATE)
        for side in ("-wal", "-shm"):
            sp = Path(str(tmp_db) + side)
            if sp.exists():
                sp.unlink()
            old_sp = Path(str(db_path) + side)
            if old_sp.exists():
                old_sp.unlink()

        target_mtime_ms = conv.last_modified_ms or conv.created_at_ms

        os.chmod(tmp_db, 0o600)
        os.replace(tmp_db, db_path)
        # CRITICAL: Sync OS filesystem mtime to real historical last_modified_ms so Antigravity's
        # startup PruneAll(500) never treats migrated Codex chats as newer than native agy sessions!
        set_path_mtime_ms(db_path, target_mtime_ms)
        self._live_disk_cids.add(cid)

        # 3. Write annotations/<cid>.pbtxt (root conversations only, matching native agy)
        if not is_subagent:
            pbtxt = encode_annotations_pbtxt(
                title=conv.title,
                last_user_view_time_ms=conv.last_modified_ms,
                archived=conv.archived,
                pinned=conv.pinned,
            )
            atomic_write_text(ann_path, pbtxt, mode=0o644)
            set_path_mtime_ms(ann_path, target_mtime_ms)

        # 4. Write brain/<cid>/ directory (assets, large step outputs, transcripts)
        self._write_brain_dir(conv, is_overwrite=is_overwrite, target_mtime_ms=target_mtime_ms)

        # 5. Upsert into conversation_summaries.db
        ws_uris_json = (
            ""
            if not conv.workspaces or conv.project_id == OUTSIDE_OF_PROJECT_ID
            else json.dumps([w.folder_uri for w in conv.workspaces], ensure_ascii=False)
        )
        sql_last_user_idx = (
            conv.last_user_input_step_index
            if (not is_subagent and conv.last_user_input_step_index > 0)
            else -1
        )
        self._summaries_conn.execute(
            """
            INSERT OR REPLACE INTO conversation_summaries(
                conversation_id, title, preview, step_count, last_modified_time,
                workspace_uris, status, source, project_id, agent_name,
                parent_conversation_id, nesting_depth, battle_id, winning_conversation_id,
                not_fully_idle, killed, last_user_input_time, last_user_input_step_index,
                app_data_dir, raw_summary, group_id
            ) VALUES (?, ?, ?, ?, ?, ?, 'CASCADE_RUN_STATUS_IDLE', '', ?, ?, ?, ?, '', '', 0, 0, ?, ?, 'antigravity', ?, '')
            """,
            (
                cid,
                "" if is_subagent else conv.title,
                "" if is_subagent else conv.preview,
                len(conv.steps),
                format_go_utc_datetime(conv.last_modified_ms),
                ws_uris_json,
                conv.project_id,
                conv.agent_name if is_subagent else "",
                conv.parent_conversation_id,
                conv.nesting_depth,
                format_go_utc_datetime(0 if is_subagent else conv.last_user_input_ms),
                sql_last_user_idx,
                raw_summary,
            ),
        )

        # 6. Update FTS5 index (root conversations; matching native agy)
        self._summaries_conn.execute("DELETE FROM conversations_fts WHERE conversation_id = ?", (cid,))
        if not is_subagent:
            self._insert_fts_rows(conv)

        # 7. Stage agyhub_summaries_proto.pb entry
        self._hub_summaries[cid] = raw_summary
        self._hub_dirty = True

    def _insert_fts_rows(self, conv: MappedConversation) -> None:
        assert self._summaries_conn is not None
        cid = conv.conversation_id
        ws_basename = ""
        if conv.workspaces:
            ws_basename = os.path.basename(file_uri_to_path(conv.workspaces[0].folder_uri))
        last_step_created_ms = conv.steps[-1].created_at_ms if conv.steps else conv.last_modified_ms
        # Metadata header row (chunk_index = -1)
        self._summaries_conn.execute(
            """
            INSERT INTO conversations_fts(
                conversation_id, chunk_index, chunk_first_step, chunk_last_step,
                last_modified_time, step_count, last_step_created_at, archived,
                user_step_byte_offsets, agent_step_byte_offsets,
                title, workspace, project, user_body, agent_body
            ) VALUES (?, -1, 0, 0, ?, ?, ?, ?, '', '', ?, ?, ?, '', '')
            """,
            (
                cid,
                conv.last_modified_ms * 1_000_000,
                len(conv.steps),
                last_step_created_ms * 1_000_000,
                1 if conv.archived else 0,
                conv.title,
                ws_basename,
                conv.project_name if conv.project_id != OUTSIDE_OF_PROJECT_ID else "",
            ),
        )

        # Build chunked body rows (chunk_index = 0, 1, ...)
        chunk_idx = 0
        first_step = 0
        last_step = 0
        user_parts: List[str] = []
        agent_parts: List[str] = []
        user_offsets: List[List[int]] = []
        agent_offsets: List[List[int]] = []
        user_bytes = 0
        agent_bytes = 0
        has_any = False

        def flush_chunk() -> None:
            nonlocal chunk_idx, user_parts, agent_parts, user_offsets, agent_offsets, user_bytes, agent_bytes, has_any
            if not has_any:
                return
            assert self._summaries_conn is not None
            self._summaries_conn.execute(
                """
                INSERT INTO conversations_fts(
                    conversation_id, chunk_index, chunk_first_step, chunk_last_step,
                    last_modified_time, step_count, last_step_created_at, archived,
                    user_step_byte_offsets, agent_step_byte_offsets,
                    title, workspace, project, user_body, agent_body
                ) VALUES (?, ?, ?, ?, 0, 0, 0, 0, ?, ?, '', '', '', ?, ?)
                """,
                (
                    cid,
                    chunk_idx,
                    first_step,
                    last_step,
                    json.dumps(user_offsets, separators=(",", ":")),
                    json.dumps(agent_offsets, separators=(",", ":")),
                    "".join(user_parts),
                    "".join(agent_parts),
                ),
            )
            chunk_idx += 1
            user_parts = []
            agent_parts = []
            user_offsets = []
            agent_offsets = []
            user_bytes = 0
            agent_bytes = 0
            has_any = False

        for step in conv.steps:
            if not has_any:
                first_step = step.idx
            last_step = step.idx
            if step.step_type == STEP_TYPE_USER_INPUT and step.fts_user_text is not None:
                chunk_str = step.fts_user_text + "\n"
                chunk_len = len(chunk_str.encode("utf-8"))
                if has_any and (user_bytes + agent_bytes + chunk_len > FTS_CHUNK_MAX_BYTES):
                    flush_chunk()
                    first_step = step.idx
                    last_step = step.idx
                user_offsets.append([step.idx, user_bytes])
                user_parts.append(chunk_str)
                user_bytes += chunk_len
                has_any = True
            elif step.step_type == STEP_TYPE_PLANNER_RESPONSE and step.fts_agent_text is not None:
                chunk_str = step.fts_agent_text + "\n"
                chunk_len = len(chunk_str.encode("utf-8"))
                if has_any and (user_bytes + agent_bytes + chunk_len > FTS_CHUNK_MAX_BYTES):
                    flush_chunk()
                    first_step = step.idx
                    last_step = step.idx
                agent_offsets.append([step.idx, agent_bytes])
                agent_parts.append(chunk_str)
                agent_bytes += chunk_len
                has_any = True

        flush_chunk()

    def _write_brain_dir(
        self,
        conv: MappedConversation,
        *,
        is_overwrite: bool = False,
        target_mtime_ms: int = 0,
    ) -> None:
        bdir = self.paths.brain_dir / conv.conversation_id
        logs_dir = bdir / ".system_generated" / "logs"
        chunks_t = logs_dir / "chunks" / "transcript"
        chunks_tf = logs_dir / "chunks" / "transcript_full"
        uploaded_dir = bdir / ".user_uploaded"
        subagents_dir = bdir / "subagents"
        steps_dir = bdir / ".system_generated" / "steps"
        if is_overwrite and uploaded_dir.exists():
            for old_f in uploaded_dir.iterdir():
                if old_f.is_file():
                    old_f.unlink()
        if is_overwrite and subagents_dir.exists():
            for old_f in subagents_dir.iterdir():
                if old_f.is_file():
                    old_f.unlink()
        if is_overwrite and steps_dir.exists():
            shutil.rmtree(steps_dir, ignore_errors=True)
        uploaded_dir.mkdir(parents=True, exist_ok=True)
        (bdir / "scratch").mkdir(parents=True, exist_ok=True)
        chunks_t.mkdir(parents=True, exist_ok=True)
        chunks_tf.mkdir(parents=True, exist_ok=True)

        # Write extracted images / artifacts
        for asset in conv.assets:
            target = bdir / asset.rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(target, asset.data)

        # Write large step output files
        for step in conv.steps:
            if step.large_output_file is not None:
                s_idx, full_text = step.large_output_file
                out_path = bdir / ".system_generated" / "steps" / str(s_idx) / "output.txt"
                atomic_write_text(out_path, full_text)

        # Write transcript_full.jsonl and compact transcript.jsonl (+ chunk 00000000.jsonl)
        full_lines: List[str] = []
        compact_lines: List[str] = []
        for step in conv.steps:
            if not step.transcript_entry:
                continue
            entry = step.transcript_entry
            full_lines.append(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
            # Compact version truncates large fields (> 2000 chars) just like agy; transcript_full.jsonl is lossless
            c_entry = dict(entry)
            trunc = []
            for key in ("content", "thinking"):
                val = c_entry.get(key)
                if isinstance(val, str) and len(val) > 2000:
                    c_entry[key] = val[:2000] + "\n... [truncated]"
                    trunc.append(key)
            if trunc:
                c_entry["truncated_fields"] = trunc
            compact_lines.append(json.dumps(c_entry, ensure_ascii=False, separators=(",", ":")))

        full_text = ("\n".join(full_lines) + "\n") if full_lines else ""
        compact_text = ("\n".join(compact_lines) + "\n") if compact_lines else ""
        tf_path = logs_dir / "transcript_full.jsonl"
        t_path = logs_dir / "transcript.jsonl"
        atomic_write_text(tf_path, full_text)
        atomic_write_text(t_path, compact_text)
        # Hardlink chunk 00000000.jsonl to avoid duplicating transcript bytes on disk
        for src_p, dst_p, txt in (
            (tf_path, chunks_tf / "00000000.jsonl", full_text),
            (t_path, chunks_t / "00000000.jsonl", compact_text),
        ):
            try:
                if dst_p.exists():
                    dst_p.unlink()
                os.link(src_p, dst_p)
            except OSError:
                atomic_write_text(dst_p, txt)

        if target_mtime_ms > 0:
            set_path_mtime_ms(bdir, target_mtime_ms)


def clean_orphans(
    paths: AgyPaths,
    backup: Optional[BackupManager] = None,
    *,
    dry_run: bool = False,
    relocate_legacy_state: bool = True,
) -> Dict[str, Any]:
    """Find and remove orphaned summary rows, hub entries, annotations, presence locks, brain UUID dirs,
    stale manifest entries, and legacy tool files inside ~/.gemini/antigravity/.
    """
    live_cids = load_existing_agy_disk_conversation_ids(paths)
    report: Dict[str, Any] = {
        "live_disk_dbs": len(live_cids),
        "orphan_summary_cids": [],
        "orphan_fts_cids": [],
        "orphan_parent_cids": [],
        "orphan_hub_cids": [],
        "orphan_annotations": [],
        "orphan_presence_locks": [],
        "orphan_brain_dirs": [],
        "orphan_app_storage_layout_keys": [],
        "orphan_app_storage_aux_panes": 0,
        "stale_manifest_cids": [],
        "legacy_manifest_found": paths.legacy_migration_manifest.exists(),
        "legacy_backups_found": paths.legacy_backups_dir.exists(),
        "legacy_backups_moved_to": None,
    }

    # 1. Inspect conversation_summaries.db
    if paths.summaries_db.exists():
        ro_conn = open_agy_sqlite_ro(paths.summaries_db)
        try:
            report["orphan_summary_cids"] = sorted(
                cid
                for (cid,) in ro_conn.execute("SELECT conversation_id FROM conversation_summaries")
                if cid not in live_cids
            )
            try:
                report["orphan_fts_cids"] = sorted(
                    cid
                    for (cid,) in ro_conn.execute("SELECT DISTINCT conversation_id FROM conversations_fts")
                    if cid not in live_cids
                )
            except Exception:
                pass
            try:
                report["orphan_parent_cids"] = sorted(
                    cid
                    for (cid,) in ro_conn.execute("SELECT conversation_id FROM conversation_parents")
                    if cid not in live_cids
                )
            except Exception:
                pass
        finally:
            ro_conn.close()

    # 2. Inspect agyhub_summaries_proto.pb
    hub_map: Dict[str, bytes] = {}
    if paths.hub_summaries_pb.exists():
        hub_map = decode_agyhub_summaries(paths.hub_summaries_pb.read_bytes())
        report["orphan_hub_cids"] = sorted(cid for cid in hub_map if cid not in live_cids)

    # 3. Inspect annotations/*.pbtxt
    if paths.annotations_dir.exists():
        report["orphan_annotations"] = sorted(
            str(p) for p in paths.annotations_dir.glob("*.pbtxt") if p.stem not in live_cids
        )

    # 4. Inspect presence/*.lock
    if paths.presence_dir.exists():
        report["orphan_presence_locks"] = sorted(
            str(p) for p in paths.presence_dir.glob("*.lock") if p.stem not in live_cids
        )

    # 5. Inspect brain/<uuid>/ (strictly UUID-named directories only; never touch tempmediaStorage etc.)
    if paths.brain_dir.exists():
        report["orphan_brain_dirs"] = sorted(
            str(p)
            for p in paths.brain_dir.iterdir()
            if p.is_dir() and _UUID_RE.match(p.name) and p.name not in live_cids
        )

    # 6. Inspect app_storage.json for stale per-conversation layout & aux-pane entries
    app_storage_data: Optional[Dict[str, Any]] = None
    app_storage_dirty = False
    layout_prefix = "antigravity-multi-conversation-layout-v3-"
    if paths.app_storage_json.exists():
        try:
            app_storage_data = json.loads(paths.app_storage_json.read_text(encoding="utf-8"))
        except Exception:
            app_storage_data = None
    if isinstance(app_storage_data, dict):
        stale_layout_keys: List[str] = []
        for k in list(app_storage_data.keys()):
            if k.startswith(layout_prefix):
                suffix = k[len(layout_prefix) :]
                if _UUID_RE.match(suffix) and suffix not in live_cids:
                    stale_layout_keys.append(k)
        stale_layout_keys.sort()
        report["orphan_app_storage_layout_keys"] = stale_layout_keys
        if stale_layout_keys:
            app_storage_dirty = True

        # Check index list if it contains dead UUIDs
        idx_key = f"{layout_prefix}index"
        if idx_key in app_storage_data:
            raw_idx = app_storage_data[idx_key]
            try:
                idx_list = json.loads(raw_idx) if isinstance(raw_idx, str) else raw_idx
                if isinstance(idx_list, list) and any(
                    isinstance(x, str) and _UUID_RE.match(x) and x not in live_cids for x in idx_list
                ):
                    app_storage_dirty = True
            except Exception:
                pass

        # Check aux-pane-session & aux-pane-v2-session conversationPanes maps
        stale_aux_count = 0
        for aux_key in ("aux-pane-session", "aux-pane-v2-session"):
            raw_aux = app_storage_data.get(aux_key)
            if raw_aux is None:
                continue
            try:
                aux_obj = json.loads(raw_aux) if isinstance(raw_aux, str) else raw_aux
            except Exception:
                continue
            if isinstance(aux_obj, dict) and isinstance(aux_obj.get("conversationPanes"), dict):
                cp = aux_obj["conversationPanes"]
                stale_cp = [cid for cid in cp if _UUID_RE.match(cid) and cid not in live_cids]
                stale_aux_count += len(stale_cp)
                if stale_cp:
                    app_storage_dirty = True
        report["orphan_app_storage_aux_panes"] = stale_aux_count

    # 7. Inspect current migration manifest for stale entries whose .db no longer exists
    manifest = load_migration_manifest(paths.migration_manifest)
    manifest_convos = manifest.get("conversations", {})
    report["stale_manifest_cids"] = sorted(cid for cid in manifest_convos if cid not in live_cids)

    has_changes = bool(
        report["orphan_summary_cids"]
        or report["orphan_fts_cids"]
        or report["orphan_parent_cids"]
        or report["orphan_hub_cids"]
        or report["orphan_annotations"]
        or report["orphan_presence_locks"]
        or report["orphan_brain_dirs"]
        or app_storage_dirty
        or report["stale_manifest_cids"]
        or (relocate_legacy_state and (report["legacy_manifest_found"] or report["legacy_backups_found"]))
    )
    if dry_run or not has_changes:
        return report

    # Apply cleanup with full backups
    if backup is not None:
        if report["orphan_summary_cids"] or report["orphan_fts_cids"] or report["orphan_parent_cids"]:
            backup.backup_file(paths.summaries_db)
        if report["orphan_hub_cids"]:
            backup.backup_file(paths.hub_summaries_pb)
        if app_storage_dirty:
            backup.backup_file(paths.app_storage_json)
        if report["stale_manifest_cids"]:
            backup.backup_file(paths.migration_manifest)
        for ann_s in report["orphan_annotations"]:
            backup.backup_file(Path(ann_s))

    # Clean SQLite tables
    if paths.summaries_db.exists() and (
        report["orphan_summary_cids"] or report["orphan_fts_cids"] or report["orphan_parent_cids"]
    ):
        conn = sqlite3.connect(str(paths.summaries_db))
        try:
            for cid in report["orphan_summary_cids"]:
                conn.execute("DELETE FROM conversation_summaries WHERE conversation_id = ?", (cid,))
            for cid in report["orphan_fts_cids"]:
                conn.execute("DELETE FROM conversations_fts WHERE conversation_id = ?", (cid,))
            for cid in report["orphan_parent_cids"]:
                conn.execute("DELETE FROM conversation_parents WHERE conversation_id = ?", (cid,))
            conn.commit()
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except Exception:
                pass
        finally:
            conn.close()

    # Clean agyhub_summaries_proto.pb
    if paths.hub_summaries_pb.exists() and report["orphan_hub_cids"]:
        cleaned_hub = {cid: raw for cid, raw in hub_map.items() if cid in live_cids}
        atomic_write_bytes(paths.hub_summaries_pb, encode_agyhub_summaries(cleaned_hub), mode=0o600)

    # Clean orphan annotations & presence locks & brain UUID dirs
    for ann_s in report["orphan_annotations"]:
        Path(ann_s).unlink(missing_ok=True)
    for lock_s in report["orphan_presence_locks"]:
        Path(lock_s).unlink(missing_ok=True)
    for bdir_s in report["orphan_brain_dirs"]:
        shutil.rmtree(bdir_s, ignore_errors=True)

    # Clean stale conversation layout & aux-pane entries from app_storage.json
    if app_storage_dirty and isinstance(app_storage_data, dict):
        for k in report["orphan_app_storage_layout_keys"]:
            app_storage_data.pop(k, None)
        idx_key = f"{layout_prefix}index"
        if idx_key in app_storage_data:
            raw_idx = app_storage_data[idx_key]
            try:
                was_str = isinstance(raw_idx, str)
                idx_list = json.loads(raw_idx) if was_str else raw_idx
                if isinstance(idx_list, list):
                    cleaned_idx = [
                        x for x in idx_list if not (isinstance(x, str) and _UUID_RE.match(x) and x not in live_cids)
                    ]
                    app_storage_data[idx_key] = (
                        json.dumps(cleaned_idx, ensure_ascii=False, separators=(",", ":")) if was_str else cleaned_idx
                    )
            except Exception:
                pass
        for aux_key in ("aux-pane-session", "aux-pane-v2-session"):
            raw_aux = app_storage_data.get(aux_key)
            if raw_aux is None:
                continue
            try:
                was_str = isinstance(raw_aux, str)
                aux_obj = json.loads(raw_aux) if was_str else raw_aux
            except Exception:
                continue
            if isinstance(aux_obj, dict) and isinstance(aux_obj.get("conversationPanes"), dict):
                cp = aux_obj["conversationPanes"]
                for cid in [c for c in list(cp.keys()) if _UUID_RE.match(c) and c not in live_cids]:
                    cp.pop(cid, None)
                app_storage_data[aux_key] = (
                    json.dumps(aux_obj, ensure_ascii=False, separators=(",", ":")) if was_str else aux_obj
                )
        atomic_write_json(paths.app_storage_json, app_storage_data)

    # Clean stale manifest entries
    if report["stale_manifest_cids"] and paths.migration_manifest.exists():
        for cid in report["stale_manifest_cids"]:
            manifest_convos.pop(cid, None)
        save_migration_manifest(paths.migration_manifest, manifest)

    # Relocate legacy tool state out of ~/.gemini/antigravity/ into project state_dir
    if relocate_legacy_state:
        if paths.legacy_migration_manifest.exists():
            paths.legacy_migration_manifest.unlink(missing_ok=True)
        if paths.legacy_backups_dir.exists() and paths.state_dir is not None:
            dst = paths.state_dir / "legacy_backups"
            dst.mkdir(parents=True, exist_ok=True)
            for item in sorted(paths.legacy_backups_dir.iterdir()):
                target = dst / item.name
                if target.exists():
                    continue
                shutil.move(str(item), str(target))
            # Remove now-empty legacy backups dir
            try:
                paths.legacy_backups_dir.rmdir()
            except OSError:
                pass
            report["legacy_backups_moved_to"] = str(dst)

    return report
