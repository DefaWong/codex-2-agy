"""Command-line interface for codex-2-agy."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agy_proto import OUTSIDE_OF_PROJECT_ID
from .agy_store import (
    AGY_HARD_CONVERSATION_LIMIT,
    DEFAULT_RESERVE_SLOTS,
    AgyPaths,
    AgyWriter,
    clean_orphans,
    load_existing_agy_conversation_ids,
    load_existing_agy_disk_conversation_ids,
    load_existing_agy_projects,
)
from .codex_reader import CodexReader
from .mapper import (
    InlineSubagentSummary,
    build_inline_subagent_summary,
    map_codex_thread,
    ms_to_local_iso,
    pick_best_title,
    subagent_display_role,
)
from .project_merger import SUBAGENT_MODES, MigrationPlan, build_migration_plan
from .safety import (
    BackupManager,
    LiveAntigravityError,
    assert_safe_to_write,
    load_migration_manifest,
    rollback_backup,
    save_migration_manifest,
)
from .verify import verify_agy_target


def resolve_agy_paths(args: argparse.Namespace) -> AgyPaths:
    if getattr(args, "agy_home", None):
        paths = AgyPaths.custom(Path(args.agy_home).expanduser())
    else:
        paths = AgyPaths.default()
    if getattr(args, "gemini_dir", None):
        paths.gemini_dir = Path(args.gemini_dir).expanduser()
    if getattr(args, "app_storage", None):
        paths.app_storage_json = Path(args.app_storage).expanduser()
    if getattr(args, "state_dir", None):
        paths.state_dir = Path(args.state_dir).expanduser()
    return paths


def _resolve_subagent_mode(args: argparse.Namespace) -> str:
    if getattr(args, "no_subagents", False):
        return "skip"
    mode = getattr(args, "subagent_mode", None) or "inline"
    return mode


def _resolve_reserve_slots(args: argparse.Namespace) -> int:
    max_total = getattr(args, "max_total", None)
    if max_total is not None:
        clamped = max(0, min(AGY_HARD_CONVERSATION_LIMIT, int(max_total)))
        return AGY_HARD_CONVERSATION_LIMIT - clamped
    return max(0, int(getattr(args, "reserve_slots", DEFAULT_RESERVE_SLOTS)))


def _build_plan_from_args(
    reader: CodexReader,
    agy_paths: AgyPaths,
    args: argparse.Namespace,
    *,
    existing_disk_cids: set[str],
    existing_any_cids: set[str],
    manifest_convos: Dict[str, Any],
) -> MigrationPlan:
    gs = reader.load_global_state()
    si_titles = reader.load_session_index_titles()
    codex_projects = reader.load_projects(gs)
    codex_threads = reader.load_threads(si_titles)
    turn_counts = reader.get_thread_turn_counts()
    existing_agy_projects = load_existing_agy_projects(agy_paths)

    return build_migration_plan(
        codex_projects=codex_projects,
        codex_threads=codex_threads,
        global_state=gs,
        existing_agy_projects=existing_agy_projects,
        thread_turn_counts=turn_counts,
        subagent_mode=_resolve_subagent_mode(args),
        include_guardian=getattr(args, "include_guardian", False),
        include_exec=getattr(args, "include_exec", False),
        include_empty=getattr(args, "include_empty", False),
        only_thread_ids=getattr(args, "only", None),
        project_filter=getattr(args, "project", None),
        limit=getattr(args, "limit", None),
        existing_disk_cids=existing_disk_cids,
        existing_any_cids=existing_any_cids,
        manifest_convos=manifest_convos,
        update_existing=getattr(args, "update_existing", False),
        force_update=getattr(args, "force_update", False),
        hard_limit=AGY_HARD_CONVERSATION_LIMIT,
        reserve_slots=_resolve_reserve_slots(args),
        enforce_slot_budget=True,
    )


def print_plan_summary(
    plan: MigrationPlan,
    agy_paths: AgyPaths,
    existing_disk_cids: set[str],
    existing_any_cids: set[str],
    manifest: Dict[str, Any],
    *,
    update_existing: bool = False,
    force_update: bool = False,
    verbose: bool = False,
) -> None:
    b = plan.budget
    print("=== Antigravity 500 槽位预算 (Slot Budget) ===")
    print(
        f"  agy 硬上限: {b.hard_limit} | 预留槽位: {b.reserve_slots} | 本次总上限: {b.max_total_dbs}"
    )
    print(
        f"  磁盘现有 .db: {b.existing_disk_dbs} (原生 {b.existing_native_dbs} + 已迁 {b.existing_migrated_dbs}) "
        f"| 剩余可新增槽位: {b.available_new_slots}"
    )
    print(
        f"  本次预计新增 .db: {b.planned_new_dbs} | 更新已有: {b.planned_overwrite_dbs} "
        f"| 迁移后预计总数: {b.projected_total_dbs}/{b.hard_limit} (余量 {max(0, b.hard_limit - b.projected_total_dbs)})"
    )
    inlined_total = sum(len(v) for v in plan.inline_subagents_by_root.values())
    if plan.subagent_mode == "inline":
        print(f"  Subagent 策略: inline（{inlined_total} 个子代理将内联折叠进父会话并存档至 brain/<父id>/subagents/，占 0 槽位）")
    elif plan.subagent_mode == "separate":
        print("  Subagent 策略: separate（每个子代理独立生成 .db，计入 500 槽位）")
    else:
        print("  Subagent 策略: skip（跳过子代理，父会话以纯摘要卡片展示，占 0 槽位）")
    if b.trimmed_roots_by_budget > 0:
        cutoff_str = ms_to_local_iso(b.cutoff_updated_at_ms) if b.cutoff_updated_at_ms else "-"
        print(
            f"  [按时间倒序截断] 因槽位上限跳过较旧主会话: {b.trimmed_roots_by_budget} 条 "
            f"(对应子代理 {b.trimmed_subagents_by_budget} 条); 本次最旧入选时间: {cutoff_str}"
        )
    if b.budget_exceeded_explicit:
        print(
            f"  [ERROR] --only 指定的新增会话需要 {b.planned_new_dbs} 个槽位，超出当前可用槽位 {b.available_new_slots}！"
        )
    print(f"  工具状态与备份目录: {agy_paths.state_dir}")

    # Notice if legacy state or orphan summaries exist
    orphan_summaries = len(existing_any_cids - existing_disk_cids)
    legacy_exists = agy_paths.legacy_migration_manifest.exists() or agy_paths.legacy_backups_dir.exists()
    if orphan_summaries > 0 or legacy_exists:
        hints = []
        if orphan_summaries > 0:
            hints.append(f"{orphan_summaries} 条无 .db 孤儿索引")
        if legacy_exists:
            hints.append("~/.gemini/antigravity/ 下存在旧版迁移备份/清单残留")
        print(f"  [提示] 检测到 {'、'.join(hints)}，可先运行 `python3 -m codex_2_agy clean-orphans` 清理。")

    print("\n=== 项目合并计划 (Project Merge Plan) ===")
    reused = [a for a in plan.project_actions if a.action == "reuse"]
    merged = [a for a in plan.project_actions if a.action == "merge_add_roots"]
    created = [a for a in plan.project_actions if a.action == "create"]
    print(f"  复用现有 agy 项目: {len(reused)} | 合并新增目录: {len(merged)} | 新建项目: {len(created)}")
    for a in merged:
        print(
            f"  [MERGE] {a.codex_project_name} -> agy:{a.target_agy_project_name} ({a.target_agy_project_id}) "
            f"+roots={a.added_roots}"
        )
    for a in reused:
        print(f"  [REUSE] {a.codex_project_name} -> agy:{a.target_agy_project_name} ({a.target_agy_project_id})")
    if verbose:
        for a in created:
            print(f"  [CREATE] {a.target_agy_project_name} ({a.target_agy_project_id}) roots={a.all_roots}")

    roots = [t for t in plan.selected_threads if not t.is_thread_spawn]
    subs = [t for t in plan.selected_threads if t.is_thread_spawn]
    already_migrated = 0
    will_update = 0
    will_create = 0
    conflict_native = 0
    manifest_convos = manifest.get("conversations", {})

    for t in plan.selected_threads:
        if t.id in manifest_convos and t.id in existing_disk_cids:
            prev_ts = manifest_convos[t.id].get("updated_at_ms", 0)
            if update_existing and (force_update or t.updated_at_ms > prev_ts):
                will_update += 1
            else:
                already_migrated += 1
        elif t.id in existing_any_cids and t.id not in manifest_convos:
            conflict_native += 1
        elif t.id in existing_disk_cids:
            conflict_native += 1
        else:
            will_create += 1

    print("\n=== 会话迁移计划 (Conversation Migration Plan) ===")
    print(
        f"  选中顶层对话: {len(roots)} | 独立 Subagent .db: {len(subs)} | "
        f"内联 Subagent: {inlined_total} | 将写入/检查合计: {len(plan.selected_threads)}"
    )
    print(
        f"  本次新增: {will_create} | 本次更新: {will_update} | 已迁移跳过: {already_migrated} | "
        f"与 agy 原生冲突跳过: {conflict_native}"
    )
    print(f"  过滤/截断统计: {plan.skipped_counts}")

    # Distribution by target project (root threads only)
    by_proj: Dict[str, int] = {}
    for t in roots:
        res = plan.thread_resolutions[t.id]
        label = "(普通对话 / outside-of-project)" if res.project_id == OUTSIDE_OF_PROJECT_ID else res.project_name
        by_proj[label] = by_proj.get(label, 0) + 1
    print("\n=== 各项目顶层对话分布 ===")
    for label, cnt in sorted(by_proj.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {label}: {cnt}")

    if verbose and roots:
        print("\n=== 选中顶层对话明细 (按更新时间倒序，前 30 条) ===")
        for t in roots[:30]:
            res = plan.thread_resolutions[t.id]
            title = pick_best_title(t)
            pname = "outside-of-project" if res.project_id == OUTSIDE_OF_PROJECT_ID else res.project_name
            in_cnt = len(plan.inline_subagents_by_root.get(t.id, []))
            sub_tag = f" +{in_cnt}sub" if in_cnt else ""
            print(f"  {t.id} | {ms_to_local_iso(t.updated_at_ms)[:19]} | [{pname}] ({res.reason}{sub_tag}) | {title}")


def cmd_plan(args: argparse.Namespace) -> int:
    codex_home = Path(args.codex_home).expanduser()
    agy_paths = resolve_agy_paths(args)
    manifest = load_migration_manifest(agy_paths.migration_manifest)
    existing_disk_cids = load_existing_agy_disk_conversation_ids(agy_paths)
    existing_any_cids = load_existing_agy_conversation_ids(agy_paths)

    with CodexReader(codex_home) as reader:
        plan = _build_plan_from_args(
            reader,
            agy_paths,
            args,
            existing_disk_cids=existing_disk_cids,
            existing_any_cids=existing_any_cids,
            manifest_convos=manifest.get("conversations", {}),
        )
    print_plan_summary(
        plan,
        agy_paths,
        existing_disk_cids,
        existing_any_cids,
        manifest,
        update_existing=args.update_existing,
        force_update=getattr(args, "force_update", False),
        verbose=args.verbose,
    )
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    codex_home = Path(args.codex_home).expanduser()
    agy_paths = resolve_agy_paths(args)

    if not args.dry_run:
        try:
            assert_safe_to_write(agy_paths.gemini_dir, allow_live_agy=args.allow_live_agy)
        except LiveAntigravityError as e:
            print(f"[BLOCKED] {e}", file=sys.stderr)
            return 2

    manifest = load_migration_manifest(agy_paths.migration_manifest)
    existing_disk_cids = load_existing_agy_disk_conversation_ids(agy_paths)
    existing_any_cids = load_existing_agy_conversation_ids(agy_paths)

    t0 = time.time()
    with CodexReader(codex_home) as reader:
        plan = _build_plan_from_args(
            reader,
            agy_paths,
            args,
            existing_disk_cids=existing_disk_cids,
            existing_any_cids=existing_any_cids,
            manifest_convos=manifest.get("conversations", {}),
        )
        print_plan_summary(
            plan,
            agy_paths,
            existing_disk_cids,
            existing_any_cids,
            manifest,
            update_existing=args.update_existing,
            force_update=args.force_update,
            verbose=args.verbose,
        )
        if plan.budget.budget_exceeded_explicit:
            print(
                f"\n[BLOCKED] --only 指定的新增会话需要 {plan.budget.planned_new_dbs} 个槽位，"
                f"但当前仅剩 {plan.budget.available_new_slots} 个安全槽位（上限 {plan.budget.max_total_dbs}/500）。"
                "请减少指定会话数、改用 `--subagent-mode inline`，或调小 `--reserve-slots`。",
                file=sys.stderr,
            )
            return 2
        if plan.budget.projected_total_dbs > AGY_HARD_CONVERSATION_LIMIT:
            print(
                f"\n[BLOCKED] 迁移后磁盘 .db 总数将达到 {plan.budget.projected_total_dbs}，"
                f"超过 Antigravity 硬上限 {AGY_HARD_CONVERSATION_LIMIT}！已拒绝写入。",
                file=sys.stderr,
            )
            return 2

        if args.dry_run:
            print("\n[DRY-RUN] 未写入任何目标文件。")
            return 0

        # Build subagent role map for ALL known Codex subagents (semantic agent_path leaf first)
        subagent_roles: Dict[str, str] = {
            tid: subagent_display_role(st)
            for tid, st in plan.all_subagent_threads.items()
        }
        separate_subagent_ids: set[str] = (
            {t.id for t in plan.selected_threads if t.is_thread_spawn}
            if plan.subagent_mode == "separate"
            else set()
        )

        # Determine which projects actually need to be created/updated:
        # When subsetted by --only/--project/--limit or slot budget trimming, only create/merge
        # projects referenced by selected threads unless --all-projects is explicitly set.
        if args.only or args.project or args.limit is not None or plan.budget.trimmed_roots_by_budget > 0:
            needed_agy_pids = {
                plan.thread_resolutions[t.id].project_id
                for t in plan.selected_threads
                if plan.thread_resolutions[t.id].project_id != OUTSIDE_OF_PROJECT_ID
            }
            if args.all_projects:
                target_actions = plan.project_actions
            else:
                target_actions = [a for a in plan.project_actions if a.target_agy_project_id in needed_agy_pids]
        else:
            target_actions = plan.project_actions

        backup = BackupManager(
            agy_paths.backups_dir,
            agy_paths.gemini_dir,
            agy_paths.app_storage_json,
        )
        writer = AgyWriter(agy_paths, backup, max_total_dbs=plan.budget.max_total_dbs)
        writer.open()

        migrated_cids: List[str] = []
        skipped_existing = 0
        skipped_empty_after_map = 0
        total_steps = 0
        total_assets = 0
        total_inlined_subs = 0
        errors: List[str] = []

        # Track original manifest CIDs so previously migrated threads deleted by the user from the
        # sidebar are NOT mistaken for native agy UUID conflicts when we prune stale manifest entries.
        manifest_convos = manifest.setdefault("conversations", {})
        orig_manifest_cids = set(manifest_convos.keys())
        for stale_cid in [cid for cid in list(manifest_convos.keys()) if cid not in existing_disk_cids]:
            manifest_convos.pop(stale_cid, None)

        try:
            # 1. Apply project creations & root merges
            created_pids_this_run: List[str] = []
            for act in target_actions:
                if act.action == "create" and act.project_json is not None:
                    writer.write_project(act.target_agy_project_id, act.project_json, is_new=True)
                    created_pids_this_run.append(act.target_agy_project_id)
                    manifest["projects"][act.codex_project_id] = {
                        "target_agy_project_id": act.target_agy_project_id,
                        "action": act.action,
                        "name": act.target_agy_project_name,
                    }
                elif act.action == "merge_add_roots" and act.project_json is not None:
                    writer.write_project(act.target_agy_project_id, act.project_json, is_new=False)
                    manifest["projects"][act.codex_project_id] = {
                        "target_agy_project_id": act.target_agy_project_id,
                        "action": act.action,
                        "added_roots": act.added_roots,
                    }
                else:
                    manifest["projects"][act.codex_project_id] = {
                        "target_agy_project_id": act.target_agy_project_id,
                        "action": "reuse",
                    }

            if created_pids_this_run:
                ordered_new = [pid for pid in plan.new_project_order_ids if pid in created_pids_this_run]
                for pid in created_pids_this_run:
                    if pid not in ordered_new:
                        ordered_new.append(pid)
                writer.update_projects_order(ordered_new)

            # 2. Migrate selected conversations
            total_to_process = len(plan.selected_threads)
            skipped_empty_ids: set[str] = set()
            for idx_num, thread in enumerate(plan.selected_threads, start=1):
                if thread.is_thread_spawn and thread.parent_thread_id in skipped_empty_ids:
                    skipped_empty_after_map += 1
                    skipped_empty_ids.add(thread.id)
                    continue
                is_overwrite = False
                if thread.id in manifest_convos and thread.id in existing_disk_cids:
                    prev_ts = manifest_convos[thread.id].get("updated_at_ms", 0)
                    if not (args.update_existing and (args.force_update or thread.updated_at_ms > prev_ts)):
                        skipped_existing += 1
                        continue
                    is_overwrite = True
                elif thread.id in existing_any_cids and thread.id not in orig_manifest_cids:
                    # Native agy conversation with same UUID — never overwrite!
                    skipped_existing += 1
                    continue
                elif thread.id in existing_disk_cids:
                    skipped_existing += 1
                    continue

                res = plan.thread_resolutions[thread.id]
                try:
                    history = reader.load_thread_history(
                        thread,
                        parse_rollout=not args.skip_rollout_enrichment,
                    )

                    # In inline mode, load child subagent histories & summarize into parent brain/subagents/
                    inline_sub_map: Dict[str, InlineSubagentSummary] = {}
                    if plan.subagent_mode == "inline" and not thread.is_thread_spawn:
                        for sub_t in plan.inline_subagents_by_root.get(thread.id, []):
                            try:
                                sub_hist = reader.load_thread_history(
                                    sub_t,
                                    parse_rollout=not args.skip_rollout_enrichment,
                                )
                                inline_sub_map[sub_t.id] = build_inline_subagent_summary(sub_t, sub_hist)
                            except Exception as sub_exc:
                                print(
                                    f"  [WARN] 内联子代理 {sub_t.id} 解析失败（已降级为标题摘要）: {sub_exc}",
                                    file=sys.stderr,
                                )

                    mapped = map_codex_thread(
                        thread,
                        history,
                        project_id=res.project_id,
                        project_name=res.project_name,
                        workspaces=res.workspaces,
                        agy_paths=agy_paths,
                        codex_home=codex_home,
                        root_conversation_id=res.root_conversation_id,
                        subagent_roles=subagent_roles,
                        subagent_mode=plan.subagent_mode,
                        separate_subagent_ids=separate_subagent_ids,
                        inline_subagents=inline_sub_map,
                    )
                    if not mapped.steps and not args.include_empty:
                        skipped_empty_after_map += 1
                        skipped_empty_ids.add(thread.id)
                        continue

                    writer.write_conversation(mapped, is_overwrite=is_overwrite)
                    migrated_cids.append(thread.id)
                    total_steps += len(mapped.steps)
                    total_assets += len(mapped.assets)
                    total_inlined_subs += len(inline_sub_map)
                    manifest_convos[thread.id] = {
                        "updated_at_ms": thread.updated_at_ms,
                        "last_modified_ms": mapped.last_modified_ms,
                        "step_count": len(mapped.steps),
                        "asset_count": len(mapped.assets),
                        "inlined_subagents": len(inline_sub_map),
                        "subagent_mode": plan.subagent_mode,
                        "project_id": res.project_id,
                        "is_subagent": thread.is_thread_spawn,
                        "migrated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    if idx_num % 50 == 0 or idx_num == total_to_process:
                        print(
                            f"  进度: {idx_num}/{total_to_process} "
                            f"(已写入 {len(migrated_cids)}, steps={total_steps}, "
                            f"assets={total_assets}, inlined_subs={total_inlined_subs})"
                        )
                except Exception as exc:
                    errors.append(f"{thread.id}: {exc}")
                    print(f"  [ERROR] 迁移 {thread.id} 失败: {exc}", file=sys.stderr)
        finally:
            writer.close()

        save_migration_manifest(agy_paths.migration_manifest, manifest)
        backup.save_manifest()

    elapsed = time.time() - t0
    final_disk_dbs = len(load_existing_agy_disk_conversation_ids(agy_paths))
    print(f"\n=== 迁移完成 ({elapsed:.2f}s) ===")
    print(
        f"  写入会话数: {len(migrated_cids)} (总 steps={total_steps}, "
        f"提取图片/产物={total_assets}, 内联子代理={total_inlined_subs})"
    )
    print(f"  跳过已存在: {skipped_existing} | 跳过无内容: {skipped_empty_after_map} | 错误: {len(errors)}")
    print(f"  当前磁盘总 .db: {final_disk_dbs}/{AGY_HARD_CONVERSATION_LIMIT} (剩余槽位 {max(0, AGY_HARD_CONVERSATION_LIMIT - final_disk_dbs)})")
    print(f"  本次回滚快照: {backup.backup_dir}")
    print(f"  回滚命令: python3 -m codex_2_agy rollback --backup-dir {backup.backup_dir}")

    if not args.no_verify and migrated_cids:
        print("\n=== 自动完整性校验 ===")
        rep = verify_agy_target(agy_paths, only_cids=migrated_cids)
        print(
            f"  磁盘 .db={rep.disk_db_count}/{AGY_HARD_CONVERSATION_LIMIT}, 校验项目={rep.checked_projects}, "
            f"会话={rep.checked_conversations}, steps={rep.checked_steps}, "
            f"mtime校验={rep.checked_mtimes}, 媒体文件={rep.checked_media_files}, FTS行={rep.fts_rows}"
        )
        if rep.warnings:
            print(f"  警告 ({len(rep.warnings)}):")
            for w in rep.warnings[:10]:
                print(f"    - {w}")
        if rep.errors:
            print(f"  错误 ({len(rep.errors)}):", file=sys.stderr)
            for e in rep.errors[:20]:
                print(f"    - {e}", file=sys.stderr)
            return 1
        print("  校验通过: 0 错误！")

    return 1 if errors else 0


def cmd_verify(args: argparse.Namespace) -> int:
    agy_paths = resolve_agy_paths(args)
    rep = verify_agy_target(agy_paths, only_cids=args.only)
    print(
        f"校验结果: disk_dbs={rep.disk_db_count}/{AGY_HARD_CONVERSATION_LIMIT}, "
        f"projects={rep.checked_projects}, conversations={rep.checked_conversations}, "
        f"steps={rep.checked_steps}, mtimes={rep.checked_mtimes}, "
        f"media={rep.checked_media_files}, fts_rows={rep.fts_rows}"
    )
    if rep.warnings:
        print(f"警告 ({len(rep.warnings)}):")
        for w in rep.warnings[:20]:
            print(f"  - {w}")
    if rep.errors:
        print(f"错误 ({len(rep.errors)}):", file=sys.stderr)
        for e in rep.errors[:30]:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("全部校验通过 (0 错误)。")
    return 0


def cmd_clean_orphans(args: argparse.Namespace) -> int:
    agy_paths = resolve_agy_paths(args)
    if not args.dry_run:
        try:
            assert_safe_to_write(agy_paths.gemini_dir, allow_live_agy=args.allow_live_agy)
        except LiveAntigravityError as e:
            print(f"[BLOCKED] {e}", file=sys.stderr)
            return 2

    backup = None
    if not args.dry_run:
        backup = BackupManager(
            agy_paths.backups_dir,
            agy_paths.gemini_dir,
            agy_paths.app_storage_json,
        )

    rep = clean_orphans(
        agy_paths,
        backup,
        dry_run=args.dry_run,
        relocate_legacy_state=not args.keep_legacy_state,
    )

    prefix = "[DRY-RUN 预览]" if args.dry_run else "[清理完成]"
    print(f"=== 孤儿数据与残留清理 {prefix} ===")
    print(f"  磁盘有效会话 .db 数: {rep['live_disk_dbs']}")
    print(f"  孤儿 conversation_summaries 行: {len(rep['orphan_summary_cids'])}")
    print(f"  孤儿 conversations_fts 会话数: {len(rep['orphan_fts_cids'])}")
    print(f"  孤儿 conversation_parents 行: {len(rep['orphan_parent_cids'])}")
    print(f"  孤儿 agyhub_summaries_proto.pb 条目: {len(rep['orphan_hub_cids'])}")
    print(f"  孤儿 annotations/*.pbtxt 文件: {len(rep['orphan_annotations'])}")
    print(f"  孤儿 presence/*.lock 文件: {len(rep['orphan_presence_locks'])}")
    print(f"  孤儿 brain/<uuid>/ 目录: {len(rep['orphan_brain_dirs'])}")
    print(f"  失效 app_storage 窗口布局键: {len(rep['orphan_app_storage_layout_keys'])}")
    print(f"  失效 app_storage 侧边栏分屏条目: {rep['orphan_app_storage_aux_panes']}")
    print(f"  失效 manifest 会话记录: {len(rep['stale_manifest_cids'])}")
    print(f"  旧版 ~/.gemini/antigravity/codex_2_agy_manifest.json: {'发现' if rep['legacy_manifest_found'] else '无'}")
    print(f"  旧版 ~/.gemini/antigravity/codex_2_agy_backups/: {'发现' if rep['legacy_backups_found'] else '无'}")
    if rep.get("legacy_backups_moved_to"):
        print(f"  旧版备份已迁移至项目目录: {rep['legacy_backups_moved_to']}")

    if not args.dry_run and backup is not None and (backup.manifest.backed_up_files or backup.manifest.created_files):
        backup.save_manifest()
        print(f"  本次清理备份: {backup.backup_dir}")
        print(f"  回滚命令: python3 -m codex_2_agy rollback --backup-dir {backup.backup_dir}")

    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    bdir = Path(args.backup_dir).expanduser()
    try:
        stats = rollback_backup(bdir, allow_live_agy=args.allow_live_agy)
    except LiveAntigravityError as e:
        print(f"[BLOCKED] {e}", file=sys.stderr)
        return 2
    print(
        f"回滚完成: 恢复文件={stats['restored_files']}, "
        f"删除新建项目={stats['deleted_projects']}, 删除新建会话={stats['deleted_conversations']}"
    )
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    codex_home = Path(args.codex_home).expanduser()
    agy_paths = resolve_agy_paths(args)
    with CodexReader(codex_home) as reader:
        gs = reader.load_global_state()
        si = reader.load_session_index_titles()
        threads = {t.id: t for t in reader.load_threads(si)}
        t = threads.get(args.thread_id)
        if not t:
            print(f"未找到 Codex 线程: {args.thread_id}", file=sys.stderr)
            return 1
        hist = reader.load_thread_history(t, parse_rollout=True)
        plan = _build_plan_from_args(
            reader,
            agy_paths,
            argparse.Namespace(
                subagent_mode="separate",
                no_subagents=False,
                include_guardian=True,
                include_exec=True,
                include_empty=True,
                only=[args.thread_id],
                project=None,
                limit=None,
                reserve_slots=0,
                max_total=None,
                update_existing=False,
                force_update=False,
            ),
            existing_disk_cids=set(),
            existing_any_cids=set(),
            manifest_convos={},
        )
        res = plan.thread_resolutions[t.id]
        item_counts: Dict[str, int] = {}
        for it in hist.items:
            item_counts[it.item_type] = item_counts.get(it.item_type, 0) + 1
        info = {
            "id": t.id,
            "title": pick_best_title(t),
            "source_kind": t.source_kind,
            "cwd": t.cwd,
            "updated_at": ms_to_local_iso(t.updated_at_ms),
            "rollout_file": str(hist.augmentation.resolved_path),
            "target_project_id": res.project_id,
            "target_project_name": res.project_name,
            "resolution_reason": res.reason,
            "turns": len(hist.turns),
            "items": len(hist.items),
            "item_types": item_counts,
            "rollout_embedded_images": sum(len(v) for v in hist.augmentation.images_by_user_msg_index.values()),
            "rollout_usage_records": len(hist.augmentation.usages),
        }
        print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--codex-home", default="~/.codex", help="Codex 数据目录（只读，默认 ~/.codex）")
    p.add_argument("--agy-home", default=None, help="独立测试根目录（会读写 <dir>/.gemini、<dir>/app_storage.json 与 <dir>/migration_state）")
    p.add_argument("--gemini-dir", default=None, help="自定义 .gemini 目录（默认 ~/.gemini）")
    p.add_argument("--app-storage", default=None, help="自定义 Antigravity app_storage.json 路径")
    p.add_argument("--state-dir", default=None, help="工具备份与清单保存目录（默认在当前项目目录 ./migration_state）")


def _add_filter_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--subagent-mode",
        choices=SUBAGENT_MODES,
        default="inline",
        help="Subagent 处理策略：inline（默认，内联折叠到父会话并存档至 brain/<父id>/subagents/，占 0 槽位）、separate（独立建 .db，计入 500 上限）、skip（跳过子代理）",
    )
    p.add_argument("--no-subagents", action="store_true", help="等价于 --subagent-mode skip")
    p.add_argument(
        "--reserve-slots",
        type=int,
        default=DEFAULT_RESERVE_SLOTS,
        help=f"在 Antigravity {AGY_HARD_CONVERSATION_LIMIT} 会话硬上限中预留给原生会话的槽位数（默认 {DEFAULT_RESERVE_SLOTS}）",
    )
    p.add_argument(
        "--max-total",
        type=int,
        default=None,
        help=f"自定义迁移后总 .db 上限（不得超过 {AGY_HARD_CONVERSATION_LIMIT}；设置后优先于 --reserve-slots）",
    )
    p.add_argument("--include-guardian", action="store_true", help="包含 Guardian 风控审查子线程")
    p.add_argument("--include-exec", action="store_true", help="包含 codex exec 一次性测试线程")
    p.add_argument("--include-empty", action="store_true", help="包含 0 消息空白草稿线程")
    p.add_argument("--only", nargs="+", default=None, help="仅迁移指定的 Codex thread UUID（适合金丝雀试迁）")
    p.add_argument("--project", default=None, help="仅迁移指定项目（项目名、项目 ID 或 standalone）")
    p.add_argument("--limit", type=int, default=None, help="最多迁移最近 N 条顶层会话（按更新时间倒序）")
    p.add_argument("--update-existing", action="store_true", help="更新已迁移且在 Codex 端有新改动的会话")
    p.add_argument("-v", "--verbose", action="store_true", help="打印详细清单")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codex-2-agy",
        description="无损、只读地将 Codex (App/CLI) 的项目与历史会话（含 .jsonl.zst 与截图）迁移到 Antigravity (agy)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="只读预览槽位预算、项目合并与会话迁移计划（不写任何文件）")
    _add_common_flags(p_plan)
    _add_filter_flags(p_plan)
    p_plan.set_defaults(func=cmd_plan)

    p_mig = sub.add_parser("migrate", help="执行迁移（自动同步历史 mtime、受 500 槽位保护，备份存项目目录）")
    _add_common_flags(p_mig)
    _add_filter_flags(p_mig)
    p_mig.add_argument("--dry-run", action="store_true", help="仅演练，不写盘")
    p_mig.add_argument("--all-projects", action="store_true", help="即使按槽位截断或筛选也同步合并所有项目")
    p_mig.add_argument("--force-update", action="store_true", help="配合 --update-existing 强制重迁已迁移会话")
    p_mig.add_argument("--skip-rollout-enrichment", action="store_true", help="不解压 rollout（更快，但不会还原过期图片和 token 统计）")
    p_mig.add_argument("--no-verify", action="store_true", help="跳过迁移后的自动完整性校验")
    p_mig.add_argument(
        "--allow-live-agy",
        action="store_true",
        help="危险：跳过 Antigravity 运行进程检测（不建议用于真实 ~/.gemini）",
    )
    p_mig.set_defaults(func=cmd_migrate)

    p_ver = sub.add_parser("verify", help="校验 Antigravity 目录中项目、SQLite、mtime、500 上限、Protobuf、FTS5 与媒体完整性")
    _add_common_flags(p_ver)
    p_ver.add_argument("--only", nargs="+", default=None, help="仅校验指定会话 UUID")
    p_ver.set_defaults(func=cmd_verify)

    p_clean = sub.add_parser("clean-orphans", help="清理无对应 .db 的孤儿索引/锁文件，并将旧版备份移出 ~/.gemini/antigravity/")
    _add_common_flags(p_clean)
    p_clean.add_argument("--dry-run", action="store_true", help="仅扫描预览，不写盘")
    p_clean.add_argument("--keep-legacy-state", action="store_true", help="保留 ~/.gemini/antigravity/ 下的旧版 codex_2_agy_* 文件")
    p_clean.add_argument("--allow-live-agy", action="store_true", help="跳过 Antigravity 运行进程检测")
    p_clean.set_defaults(func=cmd_clean_orphans)

    p_rb = sub.add_parser("rollback", help="按备份快照目录一键回滚某次迁移")
    p_rb.add_argument("--backup-dir", required=True, help="迁移时输出的备份目录路径")
    p_rb.add_argument("--allow-live-agy", action="store_true", help="跳过 Antigravity 运行进程检测")
    p_rb.set_defaults(func=cmd_rollback)

    p_ins = sub.add_parser("inspect", help="检查单条 Codex 线程的解析详情（含 zst 图片与项目归属）")
    _add_common_flags(p_ins)
    p_ins.add_argument("thread_id", help="Codex thread UUID")
    p_ins.set_defaults(func=cmd_inspect)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
