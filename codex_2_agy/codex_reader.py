"""Read-only Codex (App + CLI) data reader.

Strictly read-only: opens SQLite DBs with `?mode=ro&immutable=1` and decompresses
`.jsonl.zst` rollouts in memory without ever writing to or modifying `~/.codex/`.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    from compression import zstd as _stdlib_zstd  # type: ignore[import-not-found]
except ImportError:
    _stdlib_zstd = None

try:
    import zstandard as _zstd  # type: ignore
except ImportError:
    _zstd = None


IMAGE_TAG_RE = re.compile(r'<image\s+[^>]*path="([^"]+)"')
DATA_URI_RE = re.compile(r"^data:([^;,]+);base64,(.+)$", re.DOTALL)


def iso_to_ms(iso_str: str | None) -> int:
    if not iso_str:
        return 0
    try:
        s = iso_str.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        return int(dt.timestamp() * 1000)
    except Exception:
        return 0


def open_sqlite_ro(path: Path) -> sqlite3.Connection:
    """Open SQLite database strictly read-only & immutable (no WAL/SHM creation or locks)."""
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def iter_rollout_lines(path: Path) -> Iterator[str]:
    """Stream lines from a plain .jsonl or zstd-compressed .jsonl.zst rollout file (read-only)."""
    if path.suffix == ".zst" or path.name.endswith(".jsonl.zst"):
        if _stdlib_zstd is not None:
            with _stdlib_zstd.open(path, mode="rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    yield line
            return
        if _zstd is not None:
            dctx = _zstd.ZstdDecompressor()
            with open(path, "rb") as fh:
                with dctx.stream_reader(fh) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                    for line in text_stream:
                        yield line
            return
        # Fallback to zstd CLI
        try:
            proc = subprocess.Popen(
                ["zstd", "-dc", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Cannot decompress {path}: install 'zstandard' (`pip install zstandard`), "
                "install the `zstd` CLI, or use Python 3.14+."
            ) from exc
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                yield line
        finally:
            proc.stdout.close()
            proc.wait()
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line


@dataclass
class CodexProject:
    id: str
    name: str
    roots: List[str]
    position: int
    created_at_ms: int
    updated_at_ms: int
    legacy_ids: List[str] = field(default_factory=list)


@dataclass
class CodexThread:
    id: str
    rollout_path: str
    created_at_ms: int
    updated_at_ms: int
    source_raw: str
    source_kind: str  # "vscode", "cli", "exec", "custom", "subagent_spawn", "subagent_guardian", "subagent_other"
    cwd: str
    title: str
    name: Optional[str]
    session_index_name: Optional[str]
    preview: str
    first_user_message: str
    has_user_event: bool
    archived: bool
    is_pinned: bool
    tokens_used: int
    model: Optional[str]
    git_sha: Optional[str]
    git_branch: Optional[str]
    git_origin_url: Optional[str]
    db_project_id: Optional[str]
    # Subagent fields (populated when source_kind == "subagent_spawn")
    parent_thread_id: Optional[str] = None
    nesting_depth: int = 0
    agent_path: Optional[str] = None
    agent_nickname: Optional[str] = None
    agent_role: Optional[str] = None
    # Worktree attachment sourceCwd if any
    worktree_source_cwd: Optional[str] = None

    @property
    def is_subagent(self) -> bool:
        return self.source_kind.startswith("subagent_")

    @property
    def is_guardian(self) -> bool:
        return self.source_kind == "subagent_guardian"

    @property
    def is_thread_spawn(self) -> bool:
        return self.source_kind == "subagent_spawn"


@dataclass
class CodexGlobalState:
    local_projects: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    project_order: List[str] = field(default_factory=list)
    legacy_to_app_project_id: Dict[str, str] = field(default_factory=dict)
    thread_project_assignments: Dict[str, str] = field(default_factory=dict)  # tid -> app or legacy project_id
    projectless_thread_ids: set[str] = field(default_factory=set)
    workspace_root_hints: Dict[str, str] = field(default_factory=dict)
    projectless_output_dirs: Dict[str, str] = field(default_factory=dict)


@dataclass
class CodexTurn:
    turn_id: str
    rollout_ordinal: int
    status: str
    started_at_ms: int
    completed_at_ms: int
    duration_ms: int


@dataclass
class CodexItem:
    turn_id: str
    item_id: str
    item_type: str
    rollout_ordinal: int
    created_at_ms: int
    started_at_ms: int
    completed_at_ms: int
    data: Dict[str, Any]


@dataclass
class RolloutImage:
    mime_type: str
    data: bytes
    path_hint: str = ""
    ordinal: int = 0


@dataclass
class RolloutUsage:
    ordinal: int
    turn_id: str
    response_id: str
    input_tokens: int
    cached_input_tokens: int
    cache_write_tokens: int
    output_tokens: int
    reasoning_output_tokens: int


@dataclass
class RolloutAugmentation:
    """Extra lossless data extracted from .jsonl / .jsonl.zst rollout files."""

    resolved_path: Optional[Path] = None
    # User message images grouped by path, turn_id, and user message occurrence
    images_by_path: Dict[str, RolloutImage] = field(default_factory=dict)
    images_by_turn_id: Dict[str, List[RolloutImage]] = field(default_factory=dict)
    images_by_user_msg_index: Dict[int, List[RolloutImage]] = field(default_factory=dict)
    usages: List[RolloutUsage] = field(default_factory=list)
    turn_models: Dict[str, str] = field(default_factory=dict)
    # Original question options from request_user_input[_async] function_calls keyed by call_id
    question_specs_by_call_id: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # Fallback items if thread_history_1.sqlite has no rows for this thread
    fallback_items: List[CodexItem] = field(default_factory=list)


@dataclass
class ThreadHistoryData:
    turns: List[CodexTurn]
    items: List[CodexItem]
    augmentation: RolloutAugmentation


def parse_source(source_raw: str) -> Tuple[str, Optional[str], int, Optional[str], Optional[str], Optional[str]]:
    """Parse threads.source into (kind, parent_thread_id, depth, agent_path, agent_nickname, agent_role)."""
    s = (source_raw or "").strip()
    if s in ("vscode", "cli", "exec"):
        return s, None, 0, None, None, None
    if s.startswith("{"):
        try:
            obj = json.loads(s)
        except Exception:
            return "custom", None, 0, None, None, None
        if "subagent" in obj:
            sub = obj["subagent"]
            if isinstance(sub, dict):
                if sub.get("other") == "guardian":
                    return "subagent_guardian", None, 1, None, None, "guardian"
                if "thread_spawn" in sub and isinstance(sub["thread_spawn"], dict):
                    ts = sub["thread_spawn"]
                    return (
                        "subagent_spawn",
                        ts.get("parent_thread_id"),
                        int(ts.get("depth") or 1),
                        ts.get("agent_path"),
                        ts.get("agent_nickname"),
                        ts.get("agent_role"),
                    )
            return "subagent_other", None, 1, None, None, None
        return "custom", None, 0, None, None, None
    return s or "unknown", None, 0, None, None, None


class CodexReader:
    """Strictly read-only facade over ~/.codex."""

    def __init__(self, codex_home: Path | None = None) -> None:
        self.codex_home = codex_home or (Path.home() / ".codex")
        self.state_db_path = self.codex_home / "state_5.sqlite"
        self.history_db_path = self.codex_home / "thread_history_1.sqlite"
        self.global_state_path = self.codex_home / ".codex-global-state.json"
        self.session_index_path = self.codex_home / "session_index.jsonl"
        self.attachments_dir = self.codex_home / "attachments"
        self.generated_images_dir = self.codex_home / "generated_images"

        self._state_conn: Optional[sqlite3.Connection] = None
        self._history_conn: Optional[sqlite3.Connection] = None

    def open(self) -> None:
        if self.state_db_path.exists():
            self._state_conn = open_sqlite_ro(self.state_db_path)
        if self.history_db_path.exists():
            self._history_conn = open_sqlite_ro(self.history_db_path)

    def close(self) -> None:
        if self._state_conn is not None:
            self._state_conn.close()
            self._state_conn = None
        if self._history_conn is not None:
            self._history_conn.close()
            self._history_conn = None

    def __enter__(self) -> "CodexReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def load_global_state(self) -> CodexGlobalState:
        gs = CodexGlobalState()
        if not self.global_state_path.exists():
            return gs
        with open(self.global_state_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return gs

        gs.local_projects = raw.get("local-projects") or {}
        gs.project_order = list(raw.get("project-order") or [])
        m_by_host = raw.get("app-server-project-id-by-legacy-project-id-by-host") or {}
        for _host, mapping in m_by_host.items():
            if isinstance(mapping, dict):
                for legacy_id, app_id in mapping.items():
                    if isinstance(legacy_id, str) and isinstance(app_id, str):
                        gs.legacy_to_app_project_id[legacy_id] = app_id

        tpa = raw.get("thread-project-assignments") or {}
        for tid, entry in tpa.items():
            if isinstance(entry, dict) and entry.get("projectId"):
                lpid = str(entry["projectId"])
                gs.thread_project_assignments[tid] = gs.legacy_to_app_project_id.get(lpid, lpid)
            elif isinstance(entry, str):
                gs.thread_project_assignments[tid] = gs.legacy_to_app_project_id.get(entry, entry)

        for tid in raw.get("projectless-thread-ids") or []:
            if isinstance(tid, str):
                gs.projectless_thread_ids.add(tid)

        for k, v in raw.items():
            if not isinstance(v, dict):
                continue
            if k.startswith("electron-owner-value:thread-workspace-root-hints:"):
                for tid, path in v.items():
                    if isinstance(path, str) and path:
                        gs.workspace_root_hints[tid] = path
            elif k.startswith("electron-owner-value:thread-projectless-output-directories:"):
                for tid, path in v.items():
                    if isinstance(path, str) and path:
                        gs.projectless_output_dirs[tid] = path
                        gs.projectless_thread_ids.add(tid)

        return gs

    def load_session_index_titles(self) -> Dict[str, str]:
        titles: Dict[str, str] = {}
        if not self.session_index_path.exists():
            return titles
        with open(self.session_index_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                tid = obj.get("id")
                tname = obj.get("thread_name")
                if isinstance(tid, str) and isinstance(tname, str) and tname.strip():
                    titles[tid] = tname.strip()
        return titles

    def load_projects(self, global_state: CodexGlobalState) -> List[CodexProject]:
        projects: Dict[str, CodexProject] = {}
        # Build reverse legacy map: app_id -> [legacy_ids]
        app_to_legacy: Dict[str, List[str]] = {}
        for lid, aid in global_state.legacy_to_app_project_id.items():
            app_to_legacy.setdefault(aid, []).append(lid)

        if self._state_conn is not None:
            cur = self._state_conn.cursor()
            roots_by_pid: Dict[str, List[str]] = {}
            for pid, path in cur.execute(
                "SELECT project_id, path FROM project_roots ORDER BY project_id, position ASC"
            ):
                roots_by_pid.setdefault(pid, []).append(os.path.normpath(path))
            for pid, name, pos, cat, uat in cur.execute(
                "SELECT id, name, position, created_at_ms, updated_at_ms FROM projects ORDER BY position ASC, id ASC"
            ):
                projects[pid] = CodexProject(
                    id=pid,
                    name=name,
                    roots=roots_by_pid.get(pid, []),
                    position=int(pos),
                    created_at_ms=int(cat or 0),
                    updated_at_ms=int(uat or 0),
                    legacy_ids=app_to_legacy.get(pid, []),
                )

        # Also include any local-projects from global state not present in state_5.sqlite
        next_pos = max((p.position for p in projects.values()), default=-1) + 1
        for lid, obj in global_state.local_projects.items():
            aid = global_state.legacy_to_app_project_id.get(lid, lid)
            if aid in projects:
                continue
            roots = [os.path.normpath(r) for r in (obj.get("rootPaths") or []) if isinstance(r, str)]
            projects[aid] = CodexProject(
                id=aid,
                name=str(obj.get("name") or (os.path.basename(roots[0]) if roots else aid)),
                roots=roots,
                position=next_pos,
                created_at_ms=int(obj.get("createdAt") or 0),
                updated_at_ms=int(obj.get("updatedAt") or 0),
                legacy_ids=[lid],
            )
            next_pos += 1

        return sorted(projects.values(), key=lambda p: (p.position, p.id))

    def load_threads(self, session_titles: Dict[str, str]) -> List[CodexThread]:
        if self._state_conn is None:
            return []
        cur = self._state_conn.cursor()

        # Load worktree sourceCwd hints from thread_attachments
        worktree_source_by_tid: Dict[str, str] = {}
        try:
            for tid, payload in cur.execute(
                "SELECT thread_id, payload FROM thread_attachments WHERE attachment_type = 'worktree'"
            ):
                try:
                    obj = json.loads(payload)
                    scwd = obj.get("sourceCwd")
                    if isinstance(scwd, str) and scwd:
                        worktree_source_by_tid[tid] = os.path.normpath(scwd)
                except Exception:
                    pass
        except sqlite3.OperationalError:
            pass

        threads: List[CodexThread] = []
        rows = cur.execute(
            """
            SELECT
                id, rollout_path, created_at, updated_at, created_at_ms, updated_at_ms,
                source, cwd, title, name, preview, first_user_message,
                has_user_event, archived, is_pinned, tokens_used, model,
                git_sha, git_branch, git_origin_url, project_id,
                agent_nickname, agent_role, agent_path
            FROM threads
            ORDER BY created_at_ms ASC, id ASC
            """
        ).fetchall()

        for r in rows:
            (
                tid, rpath, cat, uat, cat_ms, uat_ms,
                src_raw, cwd, title, name, preview, fum,
                has_ue, archived, is_pinned, tok, model,
                git_sha, git_branch, git_origin_url, db_pid,
                col_nick, col_role, col_apath,
            ) = r
            kind, parent_id, depth, apath, anick, arole = parse_source(src_raw)
            threads.append(
                CodexThread(
                    id=tid,
                    rollout_path=rpath or "",
                    created_at_ms=int(cat_ms if cat_ms else (cat or 0) * 1000),
                    updated_at_ms=int(uat_ms if uat_ms else (uat or 0) * 1000),
                    source_raw=src_raw or "",
                    source_kind=kind,
                    cwd=os.path.normpath(cwd) if cwd else "",
                    title=title or "",
                    name=name if name else None,
                    session_index_name=session_titles.get(tid),
                    preview=preview or "",
                    first_user_message=fum or "",
                    has_user_event=bool(has_ue),
                    archived=bool(archived),
                    is_pinned=bool(is_pinned),
                    tokens_used=int(tok or 0),
                    model=model if model else None,
                    git_sha=git_sha,
                    git_branch=git_branch,
                    git_origin_url=git_origin_url,
                    db_project_id=db_pid if db_pid else None,
                    parent_thread_id=parent_id,
                    nesting_depth=depth,
                    agent_path=apath or col_apath,
                    agent_nickname=anick or col_nick,
                    agent_role=arole or col_role,
                    worktree_source_cwd=worktree_source_by_tid.get(tid),
                )
            )
        return threads

    def resolve_rollout_file(self, rollout_path: str) -> Optional[Path]:
        """Locate plain .jsonl or compressed .jsonl.zst rollout (in sessions or archived_sessions)."""
        if not rollout_path:
            return None
        p = Path(rollout_path)
        candidates = [p]
        if not p.name.endswith(".zst"):
            candidates.append(Path(str(p) + ".zst"))
        else:
            candidates.append(Path(str(p)[:-4]))
        base = p.name[:-4] if p.name.endswith(".zst") else p.name
        candidates.append(self.codex_home / "archived_sessions" / base)
        candidates.append(self.codex_home / "archived_sessions" / (base + ".zst"))
        for c in candidates:
            if c.exists():
                return c
        return None

    def get_thread_turn_counts(self) -> Dict[str, Tuple[int, int]]:
        """Return {thread_id: (turn_count, item_count)} quickly from thread_history_1.sqlite."""
        if self._history_conn is None:
            return {}
        turns: Dict[str, int] = {}
        items: Dict[str, int] = {}
        for tid, cnt in self._history_conn.execute(
            "SELECT thread_id, COUNT(*) FROM thread_turns GROUP BY thread_id"
        ):
            turns[tid] = cnt
        for tid, cnt in self._history_conn.execute(
            "SELECT thread_id, COUNT(*) FROM thread_items GROUP BY thread_id"
        ):
            items[tid] = cnt
        all_tids = set(turns.keys()) | set(items.keys())
        return {tid: (turns.get(tid, 0), items.get(tid, 0)) for tid in all_tids}

    def load_thread_history(
        self,
        thread: CodexThread,
        *,
        parse_rollout: bool = True,
        prefer_rollout_only: bool = False,
    ) -> ThreadHistoryData:
        """Load full structured history from thread_history_1.sqlite augmented by .jsonl/.jsonl.zst rollout."""
        turns: List[CodexTurn] = []
        items: List[CodexItem] = []

        if self._history_conn is not None and not prefer_rollout_only:
            for tid, turn_id, ordinal, status, s_at, c_at, dur_ms in self._history_conn.execute(
                """
                SELECT thread_id, turn_id, rollout_ordinal, status, started_at, completed_at, duration_ms
                FROM thread_turns
                WHERE thread_id = ?
                ORDER BY rollout_ordinal ASC
                """,
                (thread.id,),
            ):
                turns.append(
                    CodexTurn(
                        turn_id=turn_id,
                        rollout_ordinal=int(ordinal or 0),
                        status=status or "completed",
                        started_at_ms=int((s_at or 0) * 1000),
                        completed_at_ms=int((c_at or 0) * 1000),
                        duration_ms=int(dur_ms or 0),
                    )
                )

            for turn_id, item_id, item_type, ordinal, c_ms, s_ms, comp_ms, ijson in self._history_conn.execute(
                """
                SELECT turn_id, item_id, item_type, rollout_ordinal,
                       created_at_ms, started_at_ms, completed_at_ms, item_json
                FROM thread_items
                WHERE thread_id = ?
                ORDER BY rollout_ordinal ASC, created_at_ms ASC
                """,
                (thread.id,),
            ):
                try:
                    data = json.loads(ijson)
                except Exception:
                    continue
                items.append(
                    CodexItem(
                        turn_id=turn_id,
                        item_id=item_id,
                        item_type=item_type or str(data.get("type") or ""),
                        rollout_ordinal=int(ordinal or 0),
                        created_at_ms=int(c_ms or 0),
                        started_at_ms=int(s_ms or c_ms or 0),
                        completed_at_ms=int(comp_ms or c_ms or 0),
                        data=data,
                    )
                )

        aug = RolloutAugmentation()
        if parse_rollout or not items:
            rfile = self.resolve_rollout_file(thread.rollout_path)
            if rfile is not None:
                aug = self._parse_rollout(rfile, need_fallback_items=(len(items) == 0))
                if not items and aug.fallback_items:
                    items = aug.fallback_items

        return ThreadHistoryData(turns=turns, items=items, augmentation=aug)

    def _parse_rollout(self, rfile: Path, *, need_fallback_items: bool) -> RolloutAugmentation:
        aug = RolloutAugmentation(resolved_path=rfile)
        user_msg_idx = 0
        current_turn_id = "turn-0"
        item_completed_items: List[CodexItem] = []
        response_fallback_items: List[CodexItem] = []
        pending_fn_calls: Dict[str, CodexItem] = {}

        for line_no, raw_line in enumerate(iter_rollout_lines(rfile)):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except Exception:
                continue
            rtype = obj.get("type")
            ts_ms = iso_to_ms(obj.get("timestamp"))
            ordinal = int(obj.get("ordinal") if obj.get("ordinal") is not None else line_no)
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                continue

            if rtype == "turn_context":
                tid = payload.get("turn_id") or current_turn_id
                current_turn_id = str(tid)
                model = payload.get("model")
                if isinstance(model, str) and model:
                    aug.turn_models[current_turn_id] = model

            elif rtype == "token_usage_record":
                usage = payload.get("usage")
                if isinstance(usage, dict):
                    aug.usages.append(
                        RolloutUsage(
                            ordinal=ordinal,
                            turn_id=str(payload.get("turn_id") or current_turn_id),
                            response_id=str(payload.get("response_id") or ""),
                            input_tokens=int(usage.get("input_tokens") or 0),
                            cached_input_tokens=int(usage.get("cached_input_tokens") or 0),
                            cache_write_tokens=int(usage.get("cache_write_input_tokens") or 0),
                            output_tokens=int(usage.get("output_tokens") or 0),
                            reasoning_output_tokens=int(usage.get("reasoning_output_tokens") or 0),
                        )
                    )

            elif rtype == "event_msg":
                ptype = payload.get("type")
                if ptype == "token_count" and not aug.usages:
                    info = payload.get("info")
                    if isinstance(info, dict):
                        last_u = info.get("last_token_usage")
                        if isinstance(last_u, dict):
                            aug.usages.append(
                                RolloutUsage(
                                    ordinal=ordinal,
                                    turn_id=current_turn_id,
                                    response_id="",
                                    input_tokens=int(last_u.get("input_tokens") or 0),
                                    cached_input_tokens=int(last_u.get("cached_input_tokens") or 0),
                                    cache_write_tokens=int(last_u.get("cache_write_input_tokens") or 0),
                                    output_tokens=int(last_u.get("output_tokens") or 0),
                                    reasoning_output_tokens=int(last_u.get("reasoning_output_tokens") or 0),
                                )
                            )
                elif need_fallback_items and ptype == "item_completed":
                    item = payload.get("item")
                    if isinstance(item, dict) and item.get("type"):
                        item_completed_items.append(
                            CodexItem(
                                turn_id=str(payload.get("turn_id") or current_turn_id),
                                item_id=str(item.get("id") or f"rollout-{ordinal}"),
                                item_type=str(item.get("type")),
                                rollout_ordinal=ordinal,
                                created_at_ms=ts_ms,
                                started_at_ms=ts_ms,
                                completed_at_ms=ts_ms,
                                data=item,
                            )
                        )

            elif rtype == "response_item":
                ptype = payload.get("type")
                if ptype == "message" and payload.get("role") == "user":
                    content = payload.get("content") or []
                    if not isinstance(content, list):
                        continue
                    # Skip injected environment/system wrapper messages
                    first_txt = ""
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "input_text":
                            first_txt = str(part.get("text") or "").lstrip()
                            break
                    if first_txt.startswith(
                        (
                            "<environment_context>",
                            "<permissions instructions>",
                            "# AGENTS.md",
                            "<INSTRUCTIONS>",
                            "<collaboration_mode>",
                            "<turn_aborted>",
                            "<external_codex_apps_open_page>",
                            "<recommended_plugins>",
                            "<user_instructions>",
                            "<personality_spec>",
                            "<app_context>",
                            "<apps_context>",
                            "<skill>",
                            "Side conversation boundary.",
                        )
                    ):
                        continue

                    pending_path_hint = ""
                    msg_imgs: List[RolloutImage] = []
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        pt = part.get("type")
                        if pt == "input_text":
                            txt = str(part.get("text") or "")
                            m = IMAGE_TAG_RE.search(txt)
                            if m:
                                pending_path_hint = m.group(1)
                        elif pt == "input_image":
                            url = str(part.get("image_url") or "")
                            dm = DATA_URI_RE.match(url)
                            if dm:
                                mime = dm.group(1).strip()
                                b64 = dm.group(2).strip()
                                try:
                                    raw_bytes = base64.b64decode(b64)
                                except Exception:
                                    raw_bytes = b""
                                if raw_bytes:
                                    rimg = RolloutImage(
                                        mime_type=mime,
                                        data=raw_bytes,
                                        path_hint=pending_path_hint,
                                        ordinal=ordinal,
                                    )
                                    msg_imgs.append(rimg)
                                    if pending_path_hint:
                                        aug.images_by_path[pending_path_hint] = rimg
                                pending_path_hint = ""
                    if msg_imgs:
                        aug.images_by_user_msg_index[user_msg_idx] = msg_imgs
                        aug.images_by_turn_id.setdefault(current_turn_id, []).extend(msg_imgs)
                    user_msg_idx += 1

                    # Fallback userMessage for rollouts without thread_history or item_completed
                    if need_fallback_items:
                        txt_parts = [
                            {"type": "text", "text": str(p.get("text") or "")}
                            for p in content
                            if isinstance(p, dict)
                            and p.get("type") == "input_text"
                            and not str(p.get("text") or "").startswith("<image ")
                            and str(p.get("text") or "").strip() != "</image>"
                        ]
                        if txt_parts or msg_imgs:
                            response_fallback_items.append(
                                CodexItem(
                                    turn_id=current_turn_id,
                                    item_id=str(payload.get("id") or f"u-{ordinal}"),
                                    item_type="userMessage",
                                    rollout_ordinal=ordinal,
                                    created_at_ms=ts_ms,
                                    started_at_ms=ts_ms,
                                    completed_at_ms=ts_ms,
                                    data={"type": "userMessage", "content": txt_parts},
                                )
                            )
                elif need_fallback_items and ptype == "message" and payload.get("role") == "assistant":
                    content = payload.get("content") or []
                    texts = [
                        str(p.get("text") or "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") in ("output_text", "text")
                    ]
                    full_t = "\n".join(t for t in texts if t)
                    if full_t:
                        response_fallback_items.append(
                            CodexItem(
                                turn_id=current_turn_id,
                                item_id=str(payload.get("id") or f"a-{ordinal}"),
                                item_type="agentMessage",
                                rollout_ordinal=ordinal,
                                created_at_ms=ts_ms,
                                started_at_ms=ts_ms,
                                completed_at_ms=ts_ms,
                                data={"type": "agentMessage", "text": full_t, "phase": payload.get("phase") or "final_answer"},
                            )
                        )
                elif need_fallback_items and ptype == "reasoning":
                    summary_parts = [
                        str(s.get("text") or "")
                        for s in (payload.get("summary") or [])
                        if isinstance(s, dict) and s.get("text")
                    ]
                    if summary_parts:
                        response_fallback_items.append(
                            CodexItem(
                                turn_id=current_turn_id,
                                item_id=str(payload.get("id") or f"r-{ordinal}"),
                                item_type="reasoning",
                                rollout_ordinal=ordinal,
                                created_at_ms=ts_ms,
                                started_at_ms=ts_ms,
                                completed_at_ms=ts_ms,
                                data={"type": "reasoning", "summary": summary_parts, "content": []},
                            )
                        )
                elif ptype == "function_call" and payload.get("name") in ("request_user_input_async", "request_user_input"):
                    call_id = str(payload.get("call_id") or "")
                    raw_args = payload.get("arguments")
                    if call_id and isinstance(raw_args, str):
                        try:
                            parsed_q = json.loads(raw_args)
                            qlist = parsed_q.get("questions") if isinstance(parsed_q, dict) else None
                            if isinstance(qlist, list):
                                aug.question_specs_by_call_id[call_id] = [
                                    q for q in qlist if isinstance(q, dict)
                                ]
                        except Exception:
                            pass
                elif need_fallback_items and ptype in ("function_call", "custom_tool_call"):
                    call_id = str(payload.get("call_id") or f"fc-{ordinal}")
                    tool_name = str(payload.get("name") or "tool")
                    raw_args = payload.get("arguments") if ptype == "function_call" else payload.get("input")
                    parsed_args: Any = raw_args
                    if isinstance(raw_args, str):
                        try:
                            parsed_args = json.loads(raw_args)
                        except Exception:
                            parsed_args = {"input": raw_args}
                    if tool_name in ("exec_command", "shell_command", "shell") and isinstance(parsed_args, dict):
                        cmd_val = parsed_args.get("cmd") or parsed_args.get("command") or ""
                        if isinstance(cmd_val, list):
                            cmd_val = " ".join(str(x) for x in cmd_val)
                        ci = CodexItem(
                            turn_id=current_turn_id,
                            item_id=call_id,
                            item_type="commandExecution",
                            rollout_ordinal=ordinal,
                            created_at_ms=ts_ms,
                            started_at_ms=ts_ms,
                            completed_at_ms=ts_ms,
                            data={
                                "type": "commandExecution",
                                "command": str(cmd_val),
                                "cwd": str(parsed_args.get("workdir") or parsed_args.get("cwd") or ""),
                                "exitCode": 0,
                                "status": "completed",
                                "aggregatedOutput": "",
                            },
                        )
                    else:
                        ci = CodexItem(
                            turn_id=current_turn_id,
                            item_id=call_id,
                            item_type="dynamicToolCall",
                            rollout_ordinal=ordinal,
                            created_at_ms=ts_ms,
                            started_at_ms=ts_ms,
                            completed_at_ms=ts_ms,
                            data={
                                "type": "dynamicToolCall",
                                "tool": tool_name,
                                "arguments": parsed_args if isinstance(parsed_args, dict) else {"input": str(raw_args or "")},
                                "contentItems": [],
                                "status": "completed",
                            },
                        )
                    pending_fn_calls[call_id] = ci
                    response_fallback_items.append(ci)
                elif need_fallback_items and ptype in ("function_call_output", "custom_tool_call_output"):
                    call_id = str(payload.get("call_id") or "")
                    out_val = payload.get("output")
                    out_text = out_val if isinstance(out_val, str) else json.dumps(out_val, ensure_ascii=False)
                    if call_id in pending_fn_calls:
                        ci = pending_fn_calls[call_id]
                        ci.completed_at_ms = max(ci.completed_at_ms, ts_ms)
                        if ci.item_type == "commandExecution":
                            ci.data["aggregatedOutput"] = out_text
                        else:
                            ci.data["contentItems"] = [{"type": "inputText", "text": out_text}]

        if need_fallback_items:
            aug.fallback_items = item_completed_items if item_completed_items else response_fallback_items
        return aug
