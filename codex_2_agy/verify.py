"""Post-migration integrity and lossless verification suite for Antigravity storage."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .agy_proto import (
    OUTSIDE_OF_PROJECT_ID,
    STEP_TYPE_GENERIC,
    STEP_TYPE_PLANNER_RESPONSE,
    STEP_TYPE_USER_INPUT,
    decode_agyhub_summaries,
)
from .agy_store import (
    AGY_HARD_CONVERSATION_LIMIT,
    AgyPaths,
    file_uri_to_path,
    load_existing_agy_disk_conversation_ids,
    load_existing_agy_projects,
    open_agy_sqlite_ro,
    parse_go_utc_datetime,
)
from .protobuf_wire import decode_fields


@dataclass
class VerificationReport:
    checked_projects: int = 0
    checked_conversations: int = 0
    checked_steps: int = 0
    checked_media_files: int = 0
    checked_mtimes: int = 0
    disk_db_count: int = 0
    fts_rows: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def verify_agy_target(
    paths: AgyPaths,
    *,
    only_cids: Optional[List[str]] = None,
) -> VerificationReport:
    """Verify Antigravity projects, conversation DBs, protobuf wire integrity, tool pairing, FTS5, and media."""
    rep = VerificationReport()

    # 1. Projects JSON integrity
    projects = load_existing_agy_projects(paths)
    rep.checked_projects = len(projects)
    seen_folders: Dict[str, str] = {}
    for pid, ap in projects.items():
        if ap.is_sidecar:
            continue
        if not ap.name:
            rep.warnings.append(f"Project {pid} has empty name")
        for fp in ap.folder_paths:
            nfp = os.path.normpath(fp)
            if nfp in seen_folders and seen_folders[nfp] != pid:
                rep.warnings.append(f"Duplicate project folder {nfp} in projects {seen_folders[nfp]} and {pid}")
            seen_folders[nfp] = pid

    # 1b. Hard 500-file cap check on conversations/*.db
    disk_cids = load_existing_agy_disk_conversation_ids(paths)
    rep.disk_db_count = len(disk_cids)
    if rep.disk_db_count > AGY_HARD_CONVERSATION_LIMIT:
        rep.errors.append(
            f"conversations/*.db count ({rep.disk_db_count}) exceeds Antigravity hard limit "
            f"({AGY_HARD_CONVERSATION_LIMIT}); language_server will prune oldest .db files on next startup!"
        )

    # 2. conversation_summaries.db + agyhub_summaries_proto.pb consistency
    if not paths.summaries_db.exists():
        rep.errors.append(f"Missing conversation_summaries.db at {paths.summaries_db}")
        return rep

    conn = open_agy_sqlite_ro(paths.summaries_db)
    try:
        ic = conn.execute("PRAGMA integrity_check").fetchone()
        if not ic or ic[0] != "ok":
            rep.errors.append(f"conversation_summaries.db integrity_check failed: {ic}")

        db_rows = conn.execute(
            """
            SELECT conversation_id, title, step_count, project_id, parent_conversation_id, raw_summary, last_modified_time
            FROM conversation_summaries
            """
        ).fetchall()
        summaries_by_cid = {r[0]: r for r in db_rows}

        fts_cids = {
            r[0]
            for r in conn.execute("SELECT DISTINCT conversation_id FROM conversations_fts")
        }
        rep.fts_rows = conn.execute("SELECT COUNT(*) FROM conversations_fts").fetchone()[0]
    finally:
        conn.close()

    hub_map: Dict[str, bytes] = {}
    if paths.hub_summaries_pb.exists():
        try:
            hub_map = decode_agyhub_summaries(paths.hub_summaries_pb.read_bytes())
        except Exception as e:
            rep.errors.append(f"Failed to decode agyhub_summaries_proto.pb: {e}")
    else:
        rep.errors.append(f"Missing agyhub_summaries_proto.pb at {paths.hub_summaries_pb}")

    target_cids = only_cids if only_cids is not None else sorted(summaries_by_cid.keys())

    # 3. Verify each conversation DB, protobuf blobs, tool call pairing, annotations, FTS5, and media
    for cid in target_cids:
        rep.checked_conversations += 1
        row = summaries_by_cid.get(cid)
        if row is None:
            rep.errors.append(f"Conversation {cid} missing from conversation_summaries.db")
            continue
        _cid, title, step_count, project_id, parent_cid, raw_summary, last_mod_str = row
        is_subagent = bool(parent_cid)

        # Project reference valid
        if project_id != OUTSIDE_OF_PROJECT_ID and project_id not in projects:
            rep.errors.append(f"Conversation {cid} references unknown project_id {project_id}")

        # Parent reference valid
        if is_subagent and parent_cid not in summaries_by_cid:
            rep.errors.append(f"Subagent {cid} references missing parent_conversation_id {parent_cid}")

        # raw_summary matches agyhub_summaries_proto.pb
        if cid not in hub_map:
            rep.errors.append(f"Conversation {cid} missing from agyhub_summaries_proto.pb")
        elif hub_map[cid] != raw_summary:
            rep.errors.append(f"Conversation {cid} raw_summary mismatch between DB and agyhub_summaries_proto.pb")

        # FTS5 & annotations check for root conversations
        if not is_subagent:
            if cid not in fts_cids:
                rep.warnings.append(f"Root conversation {cid} has no conversations_fts rows")
            ann_path = paths.annotations_dir / f"{cid}.pbtxt"
            if not ann_path.exists():
                rep.warnings.append(f"Root conversation {cid} missing annotations/{cid}.pbtxt")

        # Transcript check
        tfull = paths.brain_dir / cid / ".system_generated" / "logs" / "transcript_full.jsonl"
        if not tfull.exists():
            rep.warnings.append(f"Conversation {cid} missing transcript_full.jsonl")

        # Conversation DB integrity, historical mtime & protobuf verification
        db_path = paths.conversations_dir / f"{cid}.db"
        if not db_path.exists():
            rep.errors.append(f"Conversation {cid} missing {db_path}")
            continue

        expected_mtime = parse_go_utc_datetime(last_mod_str)
        if expected_mtime is not None:
            rep.checked_mtimes += 1
            actual_mtime = db_path.stat().st_mtime
            clamped_expected = min(expected_mtime, time.time())
            if only_cids is not None:
                if abs(actual_mtime - clamped_expected) > 5.0:
                    rep.errors.append(
                        f"{cid}.db mtime mismatch: disk={actual_mtime:.1f}, expected={clamped_expected:.1f} "
                        f"(summary={last_mod_str})"
                    )
            elif actual_mtime - clamped_expected > 300.0:
                rep.warnings.append(
                    f"{cid}.db disk mtime ({actual_mtime:.0f}) is >5min newer than summary last_modified_time ({last_mod_str})"
                )

        c_conn = open_agy_sqlite_ro(db_path)
        try:
            ic = c_conn.execute("PRAGMA integrity_check").fetchone()
            if not ic or ic[0] != "ok":
                rep.errors.append(f"{cid}.db integrity_check failed: {ic}")
                continue
            tm = c_conn.execute("SELECT trajectory_id, cascade_id FROM trajectory_meta").fetchall()
            if len(tm) != 1 or tm[0][1] != cid:
                rep.errors.append(f"{cid}.db invalid trajectory_meta: {tm}")
            tmb = c_conn.execute("SELECT data FROM trajectory_metadata_blob WHERE id='main'").fetchone()
            if not tmb or not tmb[0]:
                rep.errors.append(f"{cid}.db missing trajectory_metadata_blob['main']")
            else:
                decode_fields(tmb[0])

            s_rows = c_conn.execute(
                "SELECT idx, step_type, status, metadata, step_payload FROM steps ORDER BY idx ASC"
            ).fetchall()
            if len(s_rows) != step_count:
                rep.errors.append(
                    f"{cid}.db step_count mismatch: summaries={step_count}, actual={len(s_rows)}"
                )

            expected_tool_ids: List[str] = []
            for seq, (idx, st, status, meta, payload) in enumerate(s_rows):
                rep.checked_steps += 1
                if idx != seq:
                    rep.errors.append(f"{cid}.db non-contiguous step idx {idx} at position {seq}")
                if not payload:
                    rep.errors.append(f"{cid}.db step {idx} has empty step_payload")
                    continue
                try:
                    pfields = decode_fields(payload)
                except Exception as e:
                    rep.errors.append(f"{cid}.db step {idx} corrupt step_payload protobuf: {e}")
                    continue

                f5 = [v for f, wt, v in pfields if f == 5 and isinstance(v, bytes)]
                if not f5 or f5[0] != meta:
                    rep.errors.append(f"{cid}.db step {idx} metadata != step_payload.field_5")

                # Verify strict PLANNER_RESPONSE.tool_calls <-> GENERIC tool_call pairing
                if st == STEP_TYPE_PLANNER_RESPONSE:
                    if expected_tool_ids:
                        rep.errors.append(
                            f"{cid}.db step {idx} started new PLANNER_RESPONSE with unpaired tool calls: {expected_tool_ids}"
                        )
                        expected_tool_ids = []
                    for f, wt, v in pfields:
                        if f == 20 and isinstance(v, bytes):
                            for sf, swt, sv in decode_fields(v):
                                if sf == 7 and isinstance(sv, bytes):
                                    tc_id = ""
                                    for tf, twt, tv in decode_fields(sv):
                                        if tf == 1 and isinstance(tv, bytes):
                                            tc_id = tv.decode("utf-8", "replace")
                                    if tc_id:
                                        expected_tool_ids.append(tc_id)
                elif st == STEP_TYPE_GENERIC:
                    tc_id = ""
                    if meta:
                        for mf, mwt, mv in decode_fields(meta):
                            if mf == 4 and isinstance(mv, bytes):
                                for tf, twt, tv in decode_fields(mv):
                                    if tf == 1 and isinstance(tv, bytes):
                                        tc_id = tv.decode("utf-8", "replace")
                    if expected_tool_ids:
                        exp = expected_tool_ids.pop(0)
                        if exp != tc_id:
                            rep.errors.append(
                                f"{cid}.db step {idx} tool_call id mismatch: expected {exp}, got {tc_id}"
                            )
                    else:
                        rep.errors.append(f"{cid}.db step {idx} orphan GENERIC step without preceding PLANNER_RESPONSE tool_call")
                else:
                    if expected_tool_ids:
                        rep.errors.append(
                            f"{cid}.db step {idx} (type={st}) interrupted unpaired tool calls: {expected_tool_ids}"
                        )
                        expected_tool_ids = []

                # Verify media URIs on USER_INPUT exist on disk
                if st == STEP_TYPE_USER_INPUT:
                    for f, wt, v in pfields:
                        if f == 19 and isinstance(v, bytes):
                            for sf, swt, sv in decode_fields(v):
                                if sf == 9 and isinstance(sv, bytes):
                                    for mf, mwt, mv in decode_fields(sv):
                                        if mf == 5 and isinstance(mv, bytes):
                                            uri = mv.decode("utf-8", "replace")
                                            fp = file_uri_to_path(uri)
                                            rep.checked_media_files += 1
                                            if not os.path.isfile(fp) or os.path.getsize(fp) == 0:
                                                rep.errors.append(f"{cid}.db step {idx} missing media file: {fp}")

            if expected_tool_ids:
                rep.errors.append(f"{cid}.db ended with unpaired tool calls: {expected_tool_ids}")
        finally:
            c_conn.close()

    return rep
