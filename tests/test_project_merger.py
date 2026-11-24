"""Tests for project merging, sidecar protection, and 9-level thread classification."""

from __future__ import annotations

import unittest

from codex_2_agy.agy_proto import OUTSIDE_OF_PROJECT_ID
from codex_2_agy.agy_store import AgyProject
from codex_2_agy.codex_reader import CodexGlobalState, CodexProject, CodexThread
from codex_2_agy.project_merger import build_migration_plan


def _thread(tid: str, **kw) -> CodexThread:
    d = dict(
        id=tid,
        rollout_path="",
        created_at_ms=1000,
        updated_at_ms=2000,
        source_raw="vscode",
        source_kind="vscode",
        cwd="/tmp/other",
        title="t",
        name=None,
        session_index_name=None,
        preview="hello",
        first_user_message="hello",
        has_user_event=True,
        archived=False,
        is_pinned=False,
        tokens_used=10,
        model="gpt-5.3-codex",
        git_sha=None,
        git_branch=None,
        git_origin_url=None,
        db_project_id=None,
    )
    d.update(kw)
    return CodexThread(**d)


class TestProjectMerger(unittest.TestCase):
    def test_merge_existing_reuse_add_roots_and_sidecar_protection(self) -> None:
        existing_agy = {
            "agy-alpha": AgyProject(
                id="agy-alpha",
                name="Alpha Suite",
                raw={
                    "id": "agy-alpha",
                    "name": "Alpha Suite",
                    "projectResources": {"resources": [{"folderUri": "file:///work/alpha-core"}]},
                    "settings": {"custom": True},
                },
                is_sidecar=False,
                folder_paths=["/work/alpha-core"],
            ),
            "agy-beta": AgyProject(
                id="agy-beta",
                name="Beta App",
                raw={
                    "id": "agy-beta",
                    "name": "Beta App",
                    "projectResources": {"resources": [{"folderUri": "file:///work/beta"}]},
                },
                is_sidecar=False,
                folder_paths=["/work/beta"],
            ),
            "agy-sidecar": AgyProject(
                id="agy-sidecar",
                name="Vitals",
                raw={
                    "id": "agy-sidecar",
                    "name": "Vitals",
                    "sidecar": {"sidecarId": "antigravity-vitals/vitals"},
                    "projectResources": {"resources": [{"folderUri": "file:///work/newproj"}]},
                },
                is_sidecar=True,
                folder_paths=["/work/newproj"],
            ),
        }
        codex_projects = [
            CodexProject("cx-alpha", "Alpha Suite", ["/work/alpha-core", "/work/alpha-cloud"], 0, 1, 1),
            CodexProject("cx-beta", "Beta", ["/work/beta"], 1, 1, 1),
            CodexProject("cx-new", "NewProj", ["/work/newproj"], 2, 1, 1),
        ]
        gs = CodexGlobalState(
            projectless_thread_ids={"t-projless"},
            thread_project_assignments={"t-tpa": "cx-beta"},
            workspace_root_hints={"t-hint": "/work/newproj"},
        )
        threads = [
            _thread("t-db", db_project_id="cx-alpha", cwd="/work/alpha-cloud"),
            _thread("t-tpa", cwd="/somewhere/else"),
            _thread("t-projless", cwd="/work/beta"),
            _thread("t-hint", cwd="/somewhere/else"),
            _thread("t-wt", cwd="/Users/me/.codex/worktrees/ab12/alpha-core", worktree_source_cwd="/work/alpha-core"),
            _thread("t-subdir", cwd="/work/newproj/sub/dir"),
            _thread("t-wt-base", cwd="/Users/me/.codex/worktrees/zz99/alpha-cloud"),
            _thread("t-standalone", cwd="/Users/me/RandomDir"),
            _thread("t-exec", source_kind="exec", cwd="/work/beta"),
            _thread("t-guardian", source_kind="subagent_guardian", cwd="/work/beta"),
            _thread("t-empty", cwd="/work/beta", preview="", first_user_message="", has_user_event=False),
            # Rollout-only subagent (not in thread_history_1.sqlite, so turn_counts=(0,0) but has_user_event=True)
            _thread(
                "t-sub1",
                source_kind="subagent_spawn",
                parent_thread_id="t-db",
                nesting_depth=1,
                cwd="/work/alpha-cloud",
            ),
            # Depth-2 subagent under t-sub1
            _thread(
                "t-sub2",
                source_kind="subagent_spawn",
                parent_thread_id="t-sub1",
                nesting_depth=2,
                cwd="/work/alpha-cloud",
            ),
            # Empty depth-1 subagent + child depth-2 (depth-2 must not become orphan!)
            _thread(
                "t-sub-empty",
                source_kind="subagent_spawn",
                parent_thread_id="t-db",
                nesting_depth=1,
                preview="",
                first_user_message="",
                has_user_event=False,
            ),
            _thread(
                "t-sub-orphan",
                source_kind="subagent_spawn",
                parent_thread_id="t-sub-empty",
                nesting_depth=2,
            ),
        ]
        turn_counts = {t.id: (1, 2) for t in threads if t.id not in ("t-empty", "t-sub1", "t-sub-empty")}

        plan = build_migration_plan(
            codex_projects=codex_projects,
            codex_threads=threads,
            global_state=gs,
            existing_agy_projects=existing_agy,
            thread_turn_counts=turn_counts,
            subagent_mode="separate",
        )

        by_cx = {a.codex_project_id: a for a in plan.project_actions}
        # Alpha Suite merged + preserved settings + added /work/alpha-cloud
        self.assertEqual(by_cx["cx-alpha"].action, "merge_add_roots")
        self.assertEqual(by_cx["cx-alpha"].target_agy_project_id, "agy-alpha")
        self.assertEqual(by_cx["cx-alpha"].added_roots, ["/work/alpha-cloud"])
        self.assertEqual(by_cx["cx-alpha"].project_json["settings"], {"custom": True})
        # Beta reused
        self.assertEqual(by_cx["cx-beta"].action, "reuse")
        self.assertEqual(by_cx["cx-beta"].target_agy_project_id, "agy-beta")
        # NewProj NOT matched to sidecar, created fresh with cx-new UUID
        self.assertEqual(by_cx["cx-new"].action, "create")
        self.assertEqual(by_cx["cx-new"].target_agy_project_id, "cx-new")

        # Verify thread resolutions
        res = plan.thread_resolutions
        self.assertEqual(res["t-db"].project_id, "agy-alpha")
        self.assertEqual(res["t-tpa"].project_id, "agy-beta")
        self.assertEqual(res["t-projless"].project_id, OUTSIDE_OF_PROJECT_ID)
        self.assertEqual(res["t-hint"].project_id, "cx-new")
        self.assertEqual(res["t-wt"].project_id, "agy-alpha")
        self.assertEqual(res["t-subdir"].project_id, "cx-new")
        self.assertEqual(res["t-wt-base"].project_id, "agy-alpha")
        self.assertEqual(res["t-standalone"].project_id, OUTSIDE_OF_PROJECT_ID)
        self.assertEqual(res["t-sub1"].project_id, "agy-alpha")
        self.assertEqual(res["t-sub2"].root_conversation_id, "t-db")

        selected_ids = [t.id for t in plan.selected_threads]
        self.assertIn("t-sub1", selected_ids)
        self.assertIn("t-sub2", selected_ids)
        self.assertNotIn("t-exec", selected_ids)
        self.assertNotIn("t-guardian", selected_ids)
        self.assertNotIn("t-empty", selected_ids)
        self.assertNotIn("t-sub-empty", selected_ids)
        self.assertNotIn("t-sub-orphan", selected_ids)

    def test_slot_budget_reverse_chronological_and_inline_subagents(self) -> None:
        # 5 root threads with distinct updated_at_ms + 3 subagents under newest root
        roots = [
            _thread(f"root-{i}", updated_at_ms=1000 * i, created_at_ms=1000 * i)
            for i in range(1, 6)  # root-1 (oldest) .. root-5 (newest)
        ]
        subs = [
            _thread(
                f"sub-{k}",
                source_kind="subagent_spawn",
                parent_thread_id="root-5",
                nesting_depth=1,
                updated_at_ms=5000 + k,
            )
            for k in range(1, 4)
        ]
        all_threads = roots + subs
        turn_counts = {t.id: (1, 2) for t in all_threads}

        # Case 1: hard_limit=15, reserve=10 -> max_total=5; existing_disk=2 -> available_new_slots=3
        # Default inline mode: subagents cost 0 slots -> picks newest 3 roots (root-5, root-4, root-3)!
        plan_inline = build_migration_plan(
            codex_projects=[],
            codex_threads=all_threads,
            global_state=CodexGlobalState(),
            existing_agy_projects={},
            thread_turn_counts=turn_counts,
            existing_disk_cids={"native-1", "native-2"},
            hard_limit=15,
            reserve_slots=10,
        )
        self.assertEqual(plan_inline.subagent_mode, "inline")
        self.assertEqual([t.id for t in plan_inline.selected_threads], ["root-5", "root-4", "root-3"])
        self.assertEqual(len(plan_inline.inline_subagents_by_root["root-5"]), 3)
        self.assertEqual(plan_inline.budget.available_new_slots, 3)
        self.assertEqual(plan_inline.budget.planned_new_dbs, 3)
        self.assertEqual(plan_inline.budget.projected_total_dbs, 5)
        self.assertEqual(plan_inline.budget.trimmed_roots_by_budget, 2)
        self.assertEqual(plan_inline.budget.cutoff_updated_at_ms, 3000)

        # Case 2: separate mode with same 3 available slots:
        # root-5 needs 1 + 3 = 4 slots (> 3, trimmed!), so root-4, root-3, root-2 fit (3 slots total)!
        plan_sep = build_migration_plan(
            codex_projects=[],
            codex_threads=all_threads,
            global_state=CodexGlobalState(),
            existing_agy_projects={},
            thread_turn_counts=turn_counts,
            subagent_mode="separate",
            existing_disk_cids={"native-1", "native-2"},
            hard_limit=15,
            reserve_slots=10,
        )
        self.assertEqual([t.id for t in plan_sep.selected_threads], ["root-4", "root-3", "root-2"])
        self.assertEqual(plan_sep.budget.planned_new_dbs, 3)
        self.assertEqual(plan_sep.budget.trimmed_roots_by_budget, 2)

        # Case 3: explicit --only exceeding available slots flags budget_exceeded_explicit!
        plan_over = build_migration_plan(
            codex_projects=[],
            codex_threads=all_threads,
            global_state=CodexGlobalState(),
            existing_agy_projects={},
            thread_turn_counts=turn_counts,
            only_thread_ids=["root-5", "root-4", "root-3", "root-2"],
            existing_disk_cids={"native-1", "native-2"},
            hard_limit=15,
            reserve_slots=10,
        )
        self.assertTrue(plan_over.budget.budget_exceeded_explicit)
        self.assertEqual(plan_over.budget.planned_new_dbs, 4)


if __name__ == "__main__":
    unittest.main()
