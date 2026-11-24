"""End-to-end synthetic migration, verification, idempotency, rollback, and safety tests."""

from __future__ import annotations

import base64
import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from compression import zstd as _stdlib_zstd  # type: ignore[import-not-found]
except ImportError:
    _stdlib_zstd = None

try:
    import zstandard as _zstd_pkg  # type: ignore
except ImportError:
    _zstd_pkg = None

from codex_2_agy.agy_proto import (
    OUTSIDE_OF_PROJECT_ID,
    WorkspaceInfo,
    decode_agyhub_summaries,
    encode_agyhub_summaries,
    encode_cascade_trajectory_summary,
    encode_conversation_annotations,
    encode_trajectory_metadata,
)
from codex_2_agy.agy_store import AgyPaths, SUMMARIES_DB_DDL
from codex_2_agy.cli import main as cli_main
from codex_2_agy.safety import LiveAntigravityError, assert_safe_to_write, rollback_backup
from codex_2_agy.verify import verify_agy_target

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
TINY_PNG_B64 = base64.b64encode(TINY_PNG).decode("ascii")


def _write_zst_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = ("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n").encode("utf-8")
    if _stdlib_zstd is not None:
        compressed = _stdlib_zstd.compress(raw)
    elif _zstd_pkg is not None:
        compressed = _zstd_pkg.ZstdCompressor().compress(raw)
    else:
        proc = subprocess.run(["zstd", "-c"], input=raw, capture_output=True, check=True)
        compressed = proc.stdout
    path.write_bytes(compressed)


def _build_synthetic_codex(codex_home: Path, repo_a: Path, repo_b: Path, repo_c: Path) -> dict[str, str]:
    codex_home.mkdir(parents=True, exist_ok=True)
    state_db = codex_home / "state_5.sqlite"
    hist_db = codex_home / "thread_history_1.sqlite"

    tid_root = "aaaaaaaa-1111-4111-8111-111111111111"
    tid_sub1 = "bbbbbbbb-2222-4222-8222-222222222222"
    tid_standalone = "cccccccc-3333-4333-8333-333333333333"
    tid_conflict = "dddddddd-4444-4444-8444-444444444444"
    pid_shared = "10000000-0000-4000-8000-000000000001"
    pid_new = "20000000-0000-4000-8000-000000000002"

    rollout_root = codex_home / "sessions" / "2026" / "10" / "01" / f"rollout-{tid_root}.jsonl.zst"
    rollout_sub1 = codex_home / "sessions" / "2026" / "10" / "01" / f"rollout-{tid_sub1}.jsonl.zst"
    rollout_sa = codex_home / "sessions" / "2026" / "10" / "01" / f"rollout-{tid_standalone}.jsonl.zst"

    expired_clip = "/var/folders/zz/yy/T/codex-clipboard-999.png"
    _write_zst_jsonl(
        rollout_root,
        [
            {"timestamp": "2026-10-01T10:00:00Z", "ordinal": 1, "type": "turn_context", "payload": {"turn_id": "t1", "model": "gpt-5.3-codex"}},
            {
                "timestamp": "2026-10-01T10:00:01Z",
                "ordinal": 2,
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f'<image path="{expired_clip}">'},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{TINY_PNG_B64}"},
                        {"type": "input_text", "text": "</image>"},
                        {"type": "input_text", "text": "帮我排查软路由转发丢包问题"},
                    ],
                },
            },
            {
                "timestamp": "2026-10-01T10:00:05Z",
                "ordinal": 3,
                "type": "token_usage_record",
                "payload": {
                    "turn_id": "t1",
                    "response_id": "resp-root-1",
                    "usage": {
                        "input_tokens": 1200,
                        "cached_input_tokens": 400,
                        "output_tokens": 350,
                        "reasoning_output_tokens": 100,
                    },
                },
            },
        ],
    )
    # Rollout-only subagent (NOT present in thread_history_1.sqlite!)
    _write_zst_jsonl(
        rollout_sub1,
        [
            {"timestamp": "2026-10-01T10:00:02Z", "ordinal": 1, "type": "turn_context", "payload": {"turn_id": "st1", "model": "gpt-5.3-codex"}},
            {
                "timestamp": "2026-10-01T10:00:02Z",
                "ordinal": 2,
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Inspect nftables rules"}]},
            },
            {
                "timestamp": "2026-10-01T10:00:04Z",
                "ordinal": 3,
                "type": "response_item",
                "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checked nftables, MTU clamping missing."}]},
            },
        ],
    )
    _write_zst_jsonl(
        rollout_sa,
        [
            {"timestamp": "2026-10-02T10:00:00Z", "ordinal": 1, "type": "turn_context", "payload": {"turn_id": "sa1", "model": "gpt-5.3-codex"}},
        ],
    )

    # Create state_5.sqlite
    sconn = sqlite3.connect(str(state_db))
    sconn.executescript(
        """
        CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT, position INTEGER, created_at_ms INTEGER, updated_at_ms INTEGER);
        CREATE TABLE project_roots (project_id TEXT, path TEXT, position INTEGER);
        CREATE TABLE thread_attachments (thread_id TEXT, attachment_type TEXT, payload TEXT);
        CREATE TABLE threads (
            id TEXT PRIMARY KEY, rollout_path TEXT, created_at INTEGER, updated_at INTEGER,
            created_at_ms INTEGER, updated_at_ms INTEGER, source TEXT, cwd TEXT, title TEXT,
            name TEXT, preview TEXT, first_user_message TEXT, has_user_event INTEGER,
            archived INTEGER, is_pinned INTEGER, tokens_used INTEGER, model TEXT,
            git_sha TEXT, git_branch TEXT, git_origin_url TEXT, project_id TEXT,
            agent_nickname TEXT, agent_role TEXT, agent_path TEXT
        );
        """
    )
    sconn.execute("INSERT INTO projects VALUES (?, 'SharedProj', 0, 1000, 2000)", (pid_shared,))
    sconn.execute("INSERT INTO projects VALUES (?, '软路由项目', 1, 1000, 2000)", (pid_new,))
    sconn.execute("INSERT INTO project_roots VALUES (?, ?, 0)", (pid_shared, str(repo_a)))
    sconn.execute("INSERT INTO project_roots VALUES (?, ?, 1)", (pid_shared, str(repo_b)))
    sconn.execute("INSERT INTO project_roots VALUES (?, ?, 0)", (pid_new, str(repo_c)))

    sub_source = json.dumps(
        {
            "subagent": {
                "thread_spawn": {
                    "parent_thread_id": tid_root,
                    "depth": 1,
                    "agent_path": "/root/netcheck",
                    "agent_nickname": "Euler",
                    "agent_role": "explorer",
                }
            }
        }
    )
    sconn.execute(
        """
        INSERT INTO threads VALUES
        (?, ?, 1760000000, 1760000050, 1760000000000, 1760000050000, 'vscode', ?, '软路由转发排查', '软路由丢包分析', '帮我排查软路由转发丢包问题', '帮我排查软路由转发丢包问题', 1, 0, 1, 1550, 'gpt-5.3-codex', NULL, 'main', NULL, ?, NULL, NULL, NULL),
        (?, ?, 1760000002, 1760000010, 1760000002000, 1760000010000, ?, ?, 'Inspect nftables', NULL, 'Inspect nftables', 'Inspect nftables', 1, 0, 0, 200, 'gpt-5.3-codex', NULL, 'main', NULL, NULL, 'Euler', 'explorer', '/root/netcheck'),
        (?, ?, 1760001000, 1760001050, 1760001000000, 1760001050000, 'cli', '/tmp/nowhere', '普通问答测试', NULL, '普通问答测试', '普通问答测试', 1, 0, 0, 100, 'gpt-5.3-codex', NULL, NULL, NULL, NULL, NULL, NULL, NULL),
        (?, '', 1760002000, 1760002050, 1760002000000, 1760002050000, 'vscode', ?, 'Conflict Native', NULL, 'Conflict', 'Conflict', 1, 0, 0, 50, 'gpt-5.3-codex', NULL, NULL, NULL, ?, NULL, NULL, NULL)
        """,
        (
            tid_root, str(rollout_root), str(repo_b), pid_shared,
            tid_sub1, str(rollout_sub1), sub_source, str(repo_b),
            tid_standalone, str(rollout_sa),
            tid_conflict, str(repo_c), pid_new,
        ),
    )
    sconn.commit()
    sconn.close()

    # Create thread_history_1.sqlite (intentionally omitting tid_sub1 to test rollout fallback!)
    hconn = sqlite3.connect(str(hist_db))
    hconn.executescript(
        """
        CREATE TABLE thread_turns (thread_id TEXT, turn_id TEXT, rollout_ordinal INTEGER, status TEXT, started_at INTEGER, completed_at INTEGER, duration_ms INTEGER);
        CREATE TABLE thread_items (thread_id TEXT, turn_id TEXT, item_id TEXT, item_type TEXT, rollout_ordinal INTEGER, created_at_ms INTEGER, started_at_ms INTEGER, completed_at_ms INTEGER, item_json TEXT);
        """
    )
    hconn.execute("INSERT INTO thread_turns VALUES (?, 't1', 1, 'completed', 1760000000, 1760000050, 50000)", (tid_root,))
    hconn.execute("INSERT INTO thread_turns VALUES (?, 'sa1', 1, 'completed', 1760001000, 1760001050, 50000)", (tid_standalone,))
    root_items = [
        (
            tid_root, "t1", "u1", "userMessage", 1, 1760000001000, 1760000001000, 1760000001000,
            json.dumps({
                "type": "userMessage",
                "content": [
                    {"type": "text", "text": "帮我排查软路由转发丢包问题"},
                    {"type": "localImage", "path": expired_clip},
                ],
            }, ensure_ascii=False),
        ),
        (
            tid_root, "t1", "col1", "collabAgentToolCall", 2, 1760000002000, 1760000002000, 1760000010000,
            json.dumps({
                "type": "collabAgentToolCall",
                "tool": "spawn_agent",
                "receiverThreadIds": [tid_sub1],
                "prompt": "Inspect nftables rules",
                "status": "completed",
            }),
        ),
        (
            tid_root, "t1", "fc1", "fileChange", 3, 1760000015000, 1760000015000, 1760000016000,
            json.dumps({
                "type": "fileChange",
                "status": "completed",
                "changes": [{"path": str(repo_b / "router.conf"), "kind": {"type": "update"}, "diff": "@@ -1 +1 @@\n-mtu=1500\n+mtu=1452"}],
            }),
        ),
        (
            tid_root, "t1", "am1", "agentMessage", 4, 1760000020000, 1760000020000, 1760000021000,
            json.dumps({"type": "agentMessage", "text": "已定位原因：PPPoE MTU 未做 MSS Clamping，已修复 router.conf。", "phase": "final_answer"}, ensure_ascii=False),
        ),
        (
            tid_standalone, "sa1", "u1", "userMessage", 1, 1760001001000, 1760001001000, 1760001001000,
            json.dumps({"type": "userMessage", "content": [{"type": "text", "text": "SQLite FTS5 supports Chinese?"}]}),
        ),
        (
            tid_standalone, "sa1", "a1", "agentMessage", 2, 1760001005000, 1760001005000, 1760001005000,
            json.dumps({"type": "agentMessage", "text": "Yes, with unicode61 tokenizer."}),
        ),
    ]
    hconn.executemany("INSERT INTO thread_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", root_items)
    hconn.commit()
    hconn.close()

    # Global state JSON
    (codex_home / ".codex-global-state.json").write_text(
        json.dumps(
            {
                "projectless-thread-ids": [tid_standalone],
                "project-order": [pid_new, pid_shared],
            }
        ),
        encoding="utf-8",
    )
    return {
        "root": tid_root,
        "sub1": tid_sub1,
        "standalone": tid_standalone,
        "conflict": tid_conflict,
        "pid_shared": pid_shared,
        "pid_new": pid_new,
    }


def _seed_existing_agy(agy_home: Path, repo_a: Path, conflict_cid: str) -> dict[str, bytes]:
    """Pre-populate Antigravity with 1 overlapping project, 1 sidecar, 1 native conversation, and app_storage.json."""
    paths = AgyPaths.custom(agy_home)
    paths.projects_dir.mkdir(parents=True, exist_ok=True)
    paths.conversations_dir.mkdir(parents=True, exist_ok=True)
    paths.annotations_dir.mkdir(parents=True, exist_ok=True)
    paths.brain_dir.mkdir(parents=True, exist_ok=True)

    existing_proj = {
        "id": "agy-existing-shared",
        "name": "Existing Shared Name",
        "projectResources": {"resources": [{"folderUri": f"file://{repo_a}"}]},
        "settings": {"preserveMe": 123},
    }
    proj_file = paths.projects_dir / "agy-existing-shared.json"
    proj_file.write_text(json.dumps(existing_proj, indent=2) + "\n", encoding="utf-8")

    app_storage = {"projectsOrder": json.dumps(["agy-existing-shared"]), "otherKey": "keep-intact"}
    paths.app_storage_json.write_text(json.dumps(app_storage, indent=2) + "\n", encoding="utf-8")

    # Seed conversation_summaries.db + agyhub_summaries_proto.pb with 1 native conversation (conflict_cid)
    conn = sqlite3.connect(str(paths.summaries_db))
    conn.executescript(SUMMARIES_DB_DDL)
    conn.execute("PRAGMA user_version = 3;")
    conn.execute("INSERT OR IGNORE INTO search_schema_version(version) VALUES (3);")
    raw_sum = encode_cascade_trajectory_summary(
        summary="Native Agy Convo",
        step_count=0,
        last_modified_ms=1760000000000,
        trajectory_id="native-traj",
        created_ms=1760000000000,
        workspaces=[],
        last_user_input_ms=1760000000000,
        last_user_input_step_index=-1,
        annotations_bytes=encode_conversation_annotations(title="Native Agy Convo", last_user_view_time_ms=1760000000000),
        trajectory_metadata_bytes=encode_trajectory_metadata(
            workspaces=[], created_at_ms=1760000000000, initialization_state_id="n", root_conversation_id=conflict_cid, project_id=OUTSIDE_OF_PROJECT_ID
        ),
    )
    conn.execute(
        """
        INSERT INTO conversation_summaries VALUES
        (?, 'Native Agy Convo', 'Native', 0, '2026-10-01 00:00:00+00:00', '', 'CASCADE_RUN_STATUS_IDLE', '', ?, '', '', 0, '', '', 0, 0, '2026-10-01 00:00:00+00:00', -1, 'antigravity', ?, '')
        """,
        (conflict_cid, OUTSIDE_OF_PROJECT_ID, raw_sum),
    )
    conn.commit()
    conn.close()
    paths.hub_summaries_pb.write_bytes(encode_agyhub_summaries({conflict_cid: raw_sum}))

    return {
        "proj_file": proj_file.read_bytes(),
        "app_storage": paths.app_storage_json.read_bytes(),
        "hub_pb": paths.hub_summaries_pb.read_bytes(),
        "summaries_db": paths.summaries_db.read_bytes(),
    }


class TestEndToEnd(unittest.TestCase):
    def test_full_plan_dryrun_migrate_verify_idempotent_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo_a = root / "repos" / "repo-a"
            repo_b = root / "repos" / "repo-b"
            repo_c = root / "repos" / "软路由 空格目录"
            for p in (repo_a, repo_b, repo_c):
                p.mkdir(parents=True, exist_ok=True)

            codex_home = root / "codex_home"
            agy_home = root / "agy_home"
            ids = _build_synthetic_codex(codex_home, repo_a, repo_b, repo_c)
            orig_bytes = _seed_existing_agy(agy_home, repo_a, ids["conflict"])
            paths = AgyPaths.custom(agy_home)

            # 1. Plan & dry-run must not modify anything
            rc = cli_main(["plan", "--codex-home", str(codex_home), "--agy-home", str(agy_home), "-v"])
            self.assertEqual(rc, 0)
            rc = cli_main(["migrate", "--codex-home", str(codex_home), "--agy-home", str(agy_home), "--dry-run"])
            self.assertEqual(rc, 0)
            self.assertFalse(paths.migration_manifest.exists())
            self.assertEqual(paths.hub_summaries_pb.read_bytes(), orig_bytes["hub_pb"])

            # 2. Real migration into isolated agy_home (default inline subagent mode!)
            rc = cli_main(["migrate", "--codex-home", str(codex_home), "--agy-home", str(agy_home)])
            self.assertEqual(rc, 0)

            # Verify backups & manifest live in migration_state/ (NOT inside .gemini/antigravity/!)
            self.assertTrue(paths.migration_manifest.exists())
            self.assertFalse(paths.legacy_migration_manifest.exists())
            self.assertFalse(paths.legacy_backups_dir.exists())

            # Default inline mode: ONLY 2 root .db files written (sub1 is inlined at 0 .db cost!)
            self.assertTrue((paths.conversations_dir / f"{ids['root']}.db").exists())
            self.assertTrue((paths.conversations_dir / f"{ids['standalone']}.db").exists())
            self.assertFalse((paths.conversations_dir / f"{ids['sub1']}.db").exists())
            # Subagent transcript archived under parent's brain/subagents/
            sub_md_files = list((paths.brain_dir / ids["root"] / "subagents").glob("*.md"))
            self.assertEqual(len(sub_md_files), 1)
            self.assertIn("Checked nftables, MTU clamping missing.", sub_md_files[0].read_text("utf-8"))

            # CRITICAL: Verify OS filesystem mtime of migrated .db matches real historical last_modified_ms!
            root_db_mtime = (paths.conversations_dir / f"{ids['root']}.db").stat().st_mtime
            sa_db_mtime = (paths.conversations_dir / f"{ids['standalone']}.db").stat().st_mtime
            self.assertAlmostEqual(root_db_mtime, 1760000050.0, delta=2.0)
            self.assertAlmostEqual(sa_db_mtime, 1760001050.0, delta=2.0)

            # Verify migrated conversations pass verifier with 0 errors (including mtime & 500-cap check!)
            rep = verify_agy_target(paths, only_cids=[ids["root"], ids["standalone"]])
            self.assertTrue(rep.ok, f"verify errors: {rep.errors}")
            self.assertEqual(rep.checked_conversations, 2)
            self.assertEqual(rep.checked_mtimes, 2)
            self.assertEqual(rep.checked_media_files, 1)  # recovered PNG from .jsonl.zst!

            # Verify existing overlapping project was merged & preserved settings + native conversation untouched
            merged_proj = json.loads((paths.projects_dir / "agy-existing-shared.json").read_text("utf-8"))
            self.assertEqual(merged_proj["name"], "Existing Shared Name")
            self.assertEqual(merged_proj["settings"], {"preserveMe": 123})
            self.assertEqual(len(merged_proj["projectResources"]["resources"]), 2)

            # Verify new Chinese+space project created and added to app_storage.json without clobbering otherKey
            new_proj_path = paths.projects_dir / f"{ids['pid_new']}.json"
            self.assertTrue(new_proj_path.exists())
            app_st = json.loads(paths.app_storage_json.read_text("utf-8"))
            self.assertEqual(app_st["otherKey"], "keep-intact")
            self.assertEqual(json.loads(app_st["projectsOrder"]), ["agy-existing-shared", ids["pid_new"]])

            # Verify native conflict conversation was NOT overwritten
            hub = decode_agyhub_summaries(paths.hub_summaries_pb.read_bytes())
            self.assertIn(ids["conflict"], hub)
            self.assertIn(ids["root"], hub)
            self.assertNotIn(ids["sub1"], hub)
            self.assertIn(ids["standalone"], hub)

            # Verify FTS5 full-text search works for both Chinese (prefix query, matching unicode61 + prefix='1 2 3') & English
            conn = sqlite3.connect(f"file:{paths.summaries_db}?mode=ro&immutable=1", uri=True)
            zh_hits = conn.execute(
                "SELECT conversation_id FROM conversations_fts WHERE conversations_fts MATCH '软路由*'"
            ).fetchall()
            en_hits = conn.execute(
                "SELECT conversation_id FROM conversations_fts WHERE conversations_fts MATCH 'unicode61'"
            ).fetchall()
            conn.close()
            self.assertTrue(any(r[0] == ids["root"] for r in zh_hits))
            self.assertTrue(any(r[0] == ids["standalone"] for r in en_hits))

            # 3. Second migrate run is 100% idempotent
            rc2 = cli_main(["migrate", "--codex-home", str(codex_home), "--agy-home", str(agy_home)])
            self.assertEqual(rc2, 0)

            # 3b. Simulate user deleting one migrated conversation from sidebar -> next migrate re-creates it!
            (paths.conversations_dir / f"{ids['standalone']}.db").unlink()
            rc_remig = cli_main(["migrate", "--codex-home", str(codex_home), "--agy-home", str(agy_home)])
            self.assertEqual(rc_remig, 0)
            self.assertTrue((paths.conversations_dir / f"{ids['standalone']}.db").exists())

            # 4. Rollback first migration backup -> restores pre-migration state cleanly
            backup_dirs = sorted(paths.backups_dir.iterdir())
            first_backup = backup_dirs[0]
            rc_rb = cli_main(["rollback", "--backup-dir", str(first_backup)])
            self.assertEqual(rc_rb, 0)

            self.assertEqual((paths.projects_dir / "agy-existing-shared.json").read_bytes(), orig_bytes["proj_file"])
            self.assertEqual(paths.app_storage_json.read_bytes(), orig_bytes["app_storage"])
            self.assertEqual(paths.hub_summaries_pb.read_bytes(), orig_bytes["hub_pb"])
            self.assertFalse(new_proj_path.exists())
            self.assertFalse((paths.conversations_dir / f"{ids['root']}.db").exists())
            self.assertFalse((paths.brain_dir / ids["root"]).exists())
            self.assertFalse(paths.migration_manifest.exists())

    def test_separate_subagent_mode_and_slot_cap_and_clean_orphans(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo_a = root / "repos" / "repo-a"
            repo_b = root / "repos" / "repo-b"
            repo_c = root / "repos" / "repo-c"
            for p in (repo_a, repo_b, repo_c):
                p.mkdir(parents=True, exist_ok=True)
            codex_home = root / "codex_home"
            agy_home = root / "agy_home"
            ids = _build_synthetic_codex(codex_home, repo_a, repo_b, repo_c)
            _seed_existing_agy(agy_home, repo_a, ids["conflict"])
            paths = AgyPaths.custom(agy_home)

            # 1. Explicit separate mode migrates sub1 as its own .db with historical mtime
            rc = cli_main([
                "migrate",
                "--codex-home", str(codex_home),
                "--agy-home", str(agy_home),
                "--subagent-mode", "separate",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((paths.conversations_dir / f"{ids['sub1']}.db").exists())
            self.assertAlmostEqual(
                (paths.conversations_dir / f"{ids['sub1']}.db").stat().st_mtime,
                1790848804.0,  # rollout_sub1 last step timestamp: 2026-10-01T10:00:04Z
                delta=2.0,
            )
            rep = verify_agy_target(paths, only_cids=[ids["root"], ids["sub1"], ids["standalone"]])
            self.assertTrue(rep.ok, f"verify errors: {rep.errors}")

            # 2. Seed orphans + app_storage layout/aux caches + legacy state and test clean-orphans
            (paths.brain_dir / "tempmediaStorage").mkdir(parents=True, exist_ok=True)
            orphan_uuid = "99999999-9999-4999-8999-999999999999"
            (paths.brain_dir / orphan_uuid).mkdir(parents=True, exist_ok=True)
            paths.presence_dir.mkdir(parents=True, exist_ok=True)
            (paths.presence_dir / f"{orphan_uuid}.lock").write_text("", encoding="utf-8")
            (paths.presence_dir / f"{ids['root']}.lock").write_text("", encoding="utf-8")
            st_obj = json.loads(paths.app_storage_json.read_text("utf-8"))
            st_obj[f"antigravity-multi-conversation-layout-v3-{orphan_uuid}"] = '{"focusedPaneId":"pane-1"}'
            st_obj[f"antigravity-multi-conversation-layout-v3-{ids['root']}"] = '{"focusedPaneId":"pane-1"}'
            st_obj["antigravity-multi-conversation-layout-v3-index"] = json.dumps([orphan_uuid, ids["root"]])
            st_obj["aux-pane-session"] = json.dumps({"conversationPanes": {orphan_uuid: {}, ids["root"]: {}}})
            st_obj["aux-pane-v2-session"] = json.dumps({"version": 1, "conversationPanes": {orphan_uuid: {}, ids["root"]: {}}})
            paths.app_storage_json.write_text(json.dumps(st_obj, indent=2) + "\n", encoding="utf-8")
            paths.legacy_migration_manifest.write_text("{}", encoding="utf-8")
            (paths.legacy_backups_dir / "old_run").mkdir(parents=True, exist_ok=True)
            (paths.legacy_backups_dir / "old_run" / "run_manifest.json").write_text("{}", encoding="utf-8")

            # Note: ids["conflict"] was seeded in summaries DB without a .db file -> should be cleaned as orphan!
            rc_clean = cli_main(["clean-orphans", "--agy-home", str(agy_home)])
            self.assertEqual(rc_clean, 0)
            self.assertTrue((paths.brain_dir / "tempmediaStorage").exists())  # non-UUID preserved!
            self.assertFalse((paths.brain_dir / orphan_uuid).exists())
            self.assertFalse((paths.presence_dir / f"{orphan_uuid}.lock").exists())
            self.assertTrue((paths.presence_dir / f"{ids['root']}.lock").exists())
            cleaned_st = json.loads(paths.app_storage_json.read_text("utf-8"))
            self.assertNotIn(f"antigravity-multi-conversation-layout-v3-{orphan_uuid}", cleaned_st)
            self.assertIn(f"antigravity-multi-conversation-layout-v3-{ids['root']}", cleaned_st)
            self.assertEqual(json.loads(cleaned_st["antigravity-multi-conversation-layout-v3-index"]), [ids["root"]])
            self.assertEqual(list(json.loads(cleaned_st["aux-pane-session"])["conversationPanes"].keys()), [ids["root"]])
            self.assertEqual(list(json.loads(cleaned_st["aux-pane-v2-session"])["conversationPanes"].keys()), [ids["root"]])
            self.assertFalse(paths.legacy_migration_manifest.exists())
            self.assertFalse(paths.legacy_backups_dir.exists())
            self.assertTrue((paths.state_dir / "legacy_backups" / "old_run" / "run_manifest.json").exists())
            hub = decode_agyhub_summaries(paths.hub_summaries_pb.read_bytes())
            self.assertNotIn(ids["conflict"], hub)
            self.assertIn(ids["root"], hub)

    def test_max_total_budget_selects_newest_and_blocks_explicit_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo_a = root / "repos" / "repo-a"
            repo_b = root / "repos" / "repo-b"
            repo_c = root / "repos" / "repo-c"
            for p in (repo_a, repo_b, repo_c):
                p.mkdir(parents=True, exist_ok=True)
            codex_home = root / "codex_home"
            agy_home = root / "agy_home"
            ids = _build_synthetic_codex(codex_home, repo_a, repo_b, repo_c)
            _seed_existing_agy(agy_home, repo_a, ids["conflict"])
            paths = AgyPaths.custom(agy_home)
            # Pre-seed 1 existing native .db on disk (conflict_cid is native -> skipped, takes 1 disk slot)
            (paths.conversations_dir / f"{ids['conflict']}.db").write_bytes(b"")

            # --max-total 2 with 1 existing disk .db -> only 1 new slot -> must pick newest (standalone, ts=1760001050) over root (ts=1760000050)!
            rc = cli_main([
                "migrate",
                "--codex-home", str(codex_home),
                "--agy-home", str(agy_home),
                "--max-total", "2",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((paths.conversations_dir / f"{ids['standalone']}.db").exists())
            self.assertFalse((paths.conversations_dir / f"{ids['root']}.db").exists())

            # Explicit --only when 0 slots remain must hard-block with rc=2!
            rc_blocked = cli_main([
                "migrate",
                "--codex-home", str(codex_home),
                "--agy-home", str(agy_home),
                "--max-total", "2",
                "--only", ids["root"],
            ])
            self.assertEqual(rc_blocked, 2)
            self.assertFalse((paths.conversations_dir / f"{ids['root']}.db").exists())

    def test_live_antigravity_safety_guard(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            # Custom directory is ALWAYS allowed even if Antigravity is running
            with mock.patch("codex_2_agy.safety.detect_running_antigravity_pids", return_value=[(1234, "/Applications/Antigravity.app/Contents/MacOS/Antigravity")]):
                assert_safe_to_write(Path(td) / ".gemini")
                # Default ~/.gemini MUST raise LiveAntigravityError!
                with self.assertRaises(LiveAntigravityError):
                    assert_safe_to_write(Path.home() / ".gemini")

    def test_overwrite_cleans_stale_steps_dir_and_rollback_restores_brain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo_a = root / "repos" / "repo-a"
            repo_b = root / "repos" / "repo-b"
            repo_c = root / "repos" / "repo-c"
            for p in (repo_a, repo_b, repo_c):
                p.mkdir(parents=True, exist_ok=True)
            codex_home = root / "codex_home"
            agy_home = root / "agy_home"
            ids = _build_synthetic_codex(codex_home, repo_a, repo_b, repo_c)
            _seed_existing_agy(agy_home, repo_a, ids["conflict"])
            paths = AgyPaths.custom(agy_home)

            rc1 = cli_main([
                "migrate",
                "--codex-home", str(codex_home),
                "--agy-home", str(agy_home),
                "--only", ids["root"],
            ])
            self.assertEqual(rc1, 0)

            # Seed a stale spilled step output file simulating shifted step index before overwrite
            stale_out = paths.brain_dir / ids["root"] / ".system_generated" / "steps" / "999" / "output.txt"
            stale_out.parent.mkdir(parents=True, exist_ok=True)
            stale_out.write_text("stale output", encoding="utf-8")

            rc2 = cli_main([
                "migrate",
                "--codex-home", str(codex_home),
                "--agy-home", str(agy_home),
                "--only", ids["root"],
                "--update-existing",
                "--force-update",
            ])
            self.assertEqual(rc2, 0)
            self.assertFalse(stale_out.exists())

            # Roll back the overwrite run -> stale_out must be restored from backed_up_dirs!
            backups = sorted(paths.backups_dir.iterdir())
            self.assertGreaterEqual(len(backups), 2)
            rollback_backup(backups[-1])
            self.assertTrue(stale_out.exists())
            self.assertEqual(stale_out.read_text(encoding="utf-8"), "stale output")

    def test_verify_reads_uncheckpointed_wal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo_a = root / "repos" / "repo-a"
            repo_b = root / "repos" / "repo-b"
            repo_c = root / "repos" / "repo-c"
            for p in (repo_a, repo_b, repo_c):
                p.mkdir(parents=True, exist_ok=True)
            codex_home = root / "codex_home"
            agy_home = root / "agy_home"
            ids = _build_synthetic_codex(codex_home, repo_a, repo_b, repo_c)
            _seed_existing_agy(agy_home, repo_a, ids["conflict"])
            paths = AgyPaths.custom(agy_home)

            rc = cli_main([
                "migrate",
                "--codex-home", str(codex_home),
                "--agy-home", str(agy_home),
                "--only", ids["standalone"],
            ])
            self.assertEqual(rc, 0)

            # Put summaries.db into WAL mode and make an uncheckpointed write while holding a reader
            wconn = sqlite3.connect(str(paths.summaries_db))
            wconn.execute("PRAGMA journal_mode = WAL;")
            wconn.execute("PRAGMA wal_autocheckpoint = 0;")
            wconn.execute(
                "UPDATE conversation_summaries SET preview = 'wal-updated-preview' WHERE conversation_id = ?",
                (ids["standalone"],),
            )
            wconn.commit()
            try:
                # verify_agy_target must still open cleanly and pass with 0 errors
                rep = verify_agy_target(paths, only_cids=[ids["standalone"]])
                self.assertTrue(rep.ok, f"verify failed under active WAL: {rep.errors}")
            finally:
                wconn.close()


if __name__ == "__main__":
    unittest.main()
