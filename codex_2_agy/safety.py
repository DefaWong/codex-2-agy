"""Safety guards, atomic writes, backups, manifest tracking, and rollback."""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class LiveAntigravityError(RuntimeError):
    """Raised when attempting to write to the live Antigravity directory while agy is running."""


def detect_running_antigravity_pids() -> List[tuple[int, str]]:
    """Return list of (pid, cmdline) for running Antigravity desktop app or language_server."""
    my_pid = os.getpid()
    found: List[tuple[int, str]] = []
    try:
        if sys.platform == "win32":
            proc = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode != 0:
                return []
            for row in csv.reader(io.StringIO(proc.stdout)):
                if len(row) < 2:
                    continue
                img = row[0].strip().lower()
                if img in ("antigravity.exe", "language_server.exe", "language_server_windows_x64.exe"):
                    try:
                        pid = int(row[1].strip())
                    except ValueError:
                        continue
                    if pid != my_pid:
                        found.append((pid, row[0].strip()))
            return found

        ps_cmd = (
            ["ps", "-ax", "-o", "pid=,command="]
            if sys.platform == "darwin"
            else ["ps", "-eo", "pid=,args="]
        )
        proc = subprocess.run(
            ps_cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            if pid == my_pid:
                continue
            cmd = parts[1]
            cmd_low = cmd.lower()
            if "codex-2-agy" in cmd_low or "codex_2_agy" in cmd_low:
                continue
            # Match Antigravity Electron app or its language_server binary (macOS & Linux)
            if (
                "/Applications/Antigravity.app/" in cmd
                or "Antigravity.app/Contents/MacOS/Antigravity" in cmd
                or "Antigravity.app/Contents/Resources/bin/language_server" in cmd
                or "/antigravity/antigravity" in cmd_low
                or ("antigravity" in cmd_low and "language_server" in cmd_low)
            ):
                found.append((pid, cmd[:140]))
    except Exception:
        return []
    return found


def is_default_live_agy_dir(gemini_dir: Path) -> bool:
    """Return True if gemini_dir points to the current user's real ~/.gemini directory."""
    try:
        real_default = (Path.home() / ".gemini").resolve()
        return gemini_dir.resolve() == real_default
    except Exception:
        return False


def assert_safe_to_write(gemini_dir: Path, *, allow_live_agy: bool = False) -> None:
    """Block writes to live ~/.gemini while Antigravity is running unless explicitly overridden."""
    if allow_live_agy:
        return
    if not is_default_live_agy_dir(gemini_dir):
        return
    running = detect_running_antigravity_pids()
    if running:
        sample = "\n".join(f"  PID {pid}: {cmd}" for pid, cmd in running[:5])
        raise LiveAntigravityError(
            "检测到 Antigravity (agy) 正在运行！为避免覆盖/损坏正在运行的 SQLite 与 Protobuf 状态，"
            "请先完全退出 Antigravity（Cmd+Q），然后在外部终端（如 Terminal / iTerm2）执行迁移命令。\n"
            f"检测到的进程：\n{sample}\n"
            "（如果只是预览方案，请使用 `plan` 或 `--dry-run`；如果目标是独立测试目录，请指定 `--agy-home /tmp/...`。）"
        )


def atomic_write_bytes(path: Path, data: bytes, mode: int | None = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def atomic_write_text(path: Path, text: str, mode: int | None = 0o644) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def atomic_write_json(path: Path, obj: Any, mode: int | None = 0o644) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(path, text, mode=mode)


@dataclass
class RunBackupManifest:
    run_id: str
    created_at: str
    gemini_dir: str
    app_storage_json: str
    backed_up_files: Dict[str, str] = field(default_factory=dict)  # orig_abs -> rel_path in backup_dir
    backed_up_dirs: Dict[str, str] = field(default_factory=dict)  # orig_abs_dir -> rel_path in backup_dir
    created_files: List[str] = field(default_factory=list)  # abs paths that did not exist before run
    created_projects: List[str] = field(default_factory=list)
    created_conversations: List[str] = field(default_factory=list)
    overwritten_conversations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "gemini_dir": self.gemini_dir,
            "app_storage_json": self.app_storage_json,
            "backed_up_files": self.backed_up_files,
            "backed_up_dirs": self.backed_up_dirs,
            "created_files": self.created_files,
            "created_projects": self.created_projects,
            "created_conversations": self.created_conversations,
            "overwritten_conversations": self.overwritten_conversations,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunBackupManifest":
        return cls(
            run_id=d["run_id"],
            created_at=d["created_at"],
            gemini_dir=d["gemini_dir"],
            app_storage_json=d.get("app_storage_json", ""),
            backed_up_files=dict(d.get("backed_up_files", {})),
            backed_up_dirs=dict(d.get("backed_up_dirs", {})),
            created_files=list(d.get("created_files", [])),
            created_projects=list(d.get("created_projects", [])),
            created_conversations=list(d.get("created_conversations", [])),
            overwritten_conversations=list(d.get("overwritten_conversations", [])),
        )


class BackupManager:
    """Creates timestamped pre-write snapshots and supports clean rollback."""

    def __init__(self, backup_root: Path, gemini_dir: Path, app_storage_json: Path) -> None:
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.backup_dir = backup_root / self.run_id
        # Avoid collision if run twice within the same second
        suffix = 1
        while self.backup_dir.exists():
            self.backup_dir = backup_root / f"{self.run_id}_{suffix}"
            suffix += 1
        self.files_dir = self.backup_dir / "files"
        self.manifest = RunBackupManifest(
            run_id=self.backup_dir.name,
            created_at=datetime.now(timezone.utc).isoformat(),
            gemini_dir=str(gemini_dir),
            app_storage_json=str(app_storage_json),
        )
        self._counter = 0

    def ensure_dir(self) -> None:
        self.files_dir.mkdir(parents=True, exist_ok=True)

    def backup_file(self, path: Path) -> None:
        """Back up an existing file before modifying it, or record it for deletion on rollback if new."""
        abs_str = str(path.resolve()) if path.exists() else str(path.absolute())
        if abs_str in self.manifest.backed_up_files or abs_str in self.manifest.created_files:
            return
        if not path.exists():
            self.manifest.created_files.append(abs_str)
            return
        self.ensure_dir()
        self._counter += 1
        rel = f"files/{self._counter:04d}_{path.name}"
        dst = self.backup_dir / rel
        shutil.copy2(path, dst)
        self.manifest.backed_up_files[abs_str] = rel
        # Also back up SQLite sidecars if present
        if path.suffix == ".db":
            for side in ("-wal", "-shm"):
                sp = Path(str(path) + side)
                if sp.exists():
                    self.backup_file(sp)

    def backup_dir_tree(self, path: Path) -> None:
        """Back up an existing directory tree before overwriting its contents."""
        if not path.exists() or not path.is_dir():
            return
        abs_str = str(path.resolve())
        if abs_str in self.manifest.backed_up_dirs:
            return
        self.ensure_dir()
        self._counter += 1
        rel = f"dirs/{self._counter:04d}_{path.name}"
        dst = self.backup_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(path, dst, symlinks=True)
        self.manifest.backed_up_dirs[abs_str] = rel

    def record_created_project(self, project_id: str) -> None:
        if project_id not in self.manifest.created_projects:
            self.manifest.created_projects.append(project_id)

    def record_created_conversation(self, cid: str) -> None:
        if cid not in self.manifest.created_conversations:
            self.manifest.created_conversations.append(cid)

    def record_overwritten_conversation(self, cid: str) -> None:
        if cid not in self.manifest.overwritten_conversations:
            self.manifest.overwritten_conversations.append(cid)

    def save_manifest(self) -> Path:
        self.ensure_dir()
        manifest_path = self.backup_dir / "run_manifest.json"
        atomic_write_json(manifest_path, self.manifest.to_dict())
        return manifest_path


def load_migration_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 1, "projects": {}, "conversations": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("version", 1)
    data.setdefault("projects", {})
    data.setdefault("conversations", {})
    return data


def save_migration_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    atomic_write_json(path, manifest)


def rollback_backup(backup_dir: Path, *, allow_live_agy: bool = False) -> Dict[str, int]:
    """Restore backed-up files and delete newly created projects/conversations from a run."""
    manifest_path = backup_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Backup manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        rm = RunBackupManifest.from_dict(json.load(f))

    gemini_dir = Path(rm.gemini_dir)
    assert_safe_to_write(gemini_dir, allow_live_agy=allow_live_agy)

    stats = {
        "restored_files": 0,
        "deleted_projects": 0,
        "deleted_conversations": 0,
    }

    # 1. Remove newly created conversation files & directories
    agy_dir = gemini_dir / "antigravity"
    for cid in rm.created_conversations:
        removed_any = False
        for suffix in (".db", ".db-wal", ".db-shm"):
            p = agy_dir / "conversations" / f"{cid}{suffix}"
            if p.exists():
                p.unlink()
                removed_any = True
        ann = agy_dir / "annotations" / f"{cid}.pbtxt"
        if ann.exists():
            ann.unlink()
            removed_any = True
        bdir = agy_dir / "brain" / cid
        if bdir.exists():
            shutil.rmtree(bdir)
            removed_any = True
        if removed_any:
            stats["deleted_conversations"] += 1

    # 2. Remove newly created project JSONs
    for pid in rm.created_projects:
        pjson = gemini_dir / "config" / "projects" / f"{pid}.json"
        if pjson.exists():
            pjson.unlink()
            stats["deleted_projects"] += 1

    # 2b. Remove files that did not exist before this run (e.g. fresh manifest / summaries / app_storage)
    for created_abs in rm.created_files:
        cp = Path(created_abs)
        if cp.exists():
            cp.unlink()
            stats["restored_files"] += 1
        if cp.suffix == ".db":
            for side in ("-wal", "-shm"):
                sp = Path(str(cp) + side)
                if sp.exists():
                    sp.unlink()

    # 3. Restore all backed-up files (conversation_summaries.db, agyhub_summaries_proto.pb,
    #    app_storage.json, merged project JSONs, migration manifest, etc.)
    # Remove any stale -wal/-shm before restoring .db files
    restored_dbs = set()
    for orig_abs, rel in rm.backed_up_files.items():
        orig = Path(orig_abs)
        if orig.suffix == ".db":
            restored_dbs.add(orig)
            for side in ("-wal", "-shm"):
                sp = Path(str(orig) + side)
                if sp.exists() and str(sp.resolve()) not in rm.backed_up_files:
                    sp.unlink()

    for orig_abs, rel in rm.backed_up_files.items():
        src = backup_dir / rel
        dst = Path(orig_abs)
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            stats["restored_files"] += 1

    # 4. Restore any backed-up directories (e.g. brain/<cid> on --update-existing overwrites)
    for orig_abs, rel in rm.backed_up_dirs.items():
        src = backup_dir / rel
        dst = Path(orig_abs)
        if src.exists() and src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, symlinks=True)
            stats["restored_files"] += 1

    return stats
