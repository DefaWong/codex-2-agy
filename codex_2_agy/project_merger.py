"""Project merging planner and Codex thread -> Antigravity project/workspace classifier."""

from __future__ import annotations

import copy
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agy_proto import OUTSIDE_OF_PROJECT_ID, WorkspaceInfo
from .agy_store import (
    AGY_HARD_CONVERSATION_LIMIT,
    DEFAULT_RESERVE_SLOTS,
    AgyProject,
    file_uri_to_path,
    path_to_file_uri,
)
from .codex_reader import CodexGlobalState, CodexProject, CodexThread

SUBAGENT_MODES = ("inline", "separate", "skip")


@dataclass
class GitRepoInfo:
    is_git: bool
    git_root: str = ""
    default_branch: str = "main"
    origin_url: str = ""
    computed_name: str = ""


class GitProbeCache:
    """Cached, read-only local git metadata prober."""

    def __init__(self) -> None:
        self._cache: Dict[str, GitRepoInfo] = {}

    def probe(self, folder_path: str) -> GitRepoInfo:
        norm = os.path.normpath(folder_path)
        if norm in self._cache:
            return self._cache[norm]
        info = self._probe_uncached(norm)
        self._cache[norm] = info
        return info

    def _probe_uncached(self, norm: str) -> GitRepoInfo:
        if not norm or not os.path.isdir(norm):
            return GitRepoInfo(is_git=False)
        if not os.path.exists(os.path.join(norm, ".git")):
            return GitRepoInfo(is_git=False)

        branch = ""
        # Try origin/HEAD first, then current HEAD
        for args in (
            ["git", "--no-optional-locks", "-C", norm, "symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
            ["git", "--no-optional-locks", "-C", norm, "symbolic-ref", "--short", "HEAD"],
        ):
            try:
                p = subprocess.run(args, capture_output=True, text=True, timeout=3)
                if p.returncode == 0 and p.stdout.strip():
                    b = p.stdout.strip()
                    if b.startswith("origin/"):
                        b = b[len("origin/") :]
                    branch = b
                    break
            except Exception:
                pass
        if not branch:
            branch = "main"

        origin_url = ""
        try:
            p = subprocess.run(
                ["git", "--no-optional-locks", "-C", norm, "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if p.returncode == 0:
                origin_url = p.stdout.strip()
        except Exception:
            pass

        return GitRepoInfo(
            is_git=True,
            git_root=norm,
            default_branch=branch,
            origin_url=origin_url,
            computed_name=compute_repo_name(origin_url, norm),
        )


def compute_repo_name(origin_url: str, fallback_path: str = "") -> str:
    if origin_url:
        # git@host:org/repo.git or https://host/org/repo.git
        m = re.search(r"[:/]([^/:]+/[^/:]+?)(?:\.git)?/?$", origin_url.strip())
        if m:
            return m.group(1)
    return os.path.basename(fallback_path.rstrip("/")) if fallback_path else ""


def make_project_resource_entry(folder_path: str, git_info: GitRepoInfo) -> Dict[str, Any]:
    uri = path_to_file_uri(folder_path)
    if git_info.is_git:
        return {
            "gitFolder": {
                "folderUri": uri,
                "defaultBranch": git_info.default_branch or "main",
            }
        }
    return {"folderUri": uri}


@dataclass
class ProjectMergeAction:
    codex_project_id: str
    codex_project_name: str
    target_agy_project_id: str
    target_agy_project_name: str
    action: str  # "reuse", "merge_add_roots", "create"
    added_roots: List[str] = field(default_factory=list)
    all_roots: List[str] = field(default_factory=list)
    project_json: Optional[Dict[str, Any]] = None  # Non-None for "create" or "merge_add_roots"


@dataclass
class ThreadResolution:
    project_id: str  # agy project UUID or OUTSIDE_OF_PROJECT_ID
    project_name: str
    workspaces: List[WorkspaceInfo]
    reason: str
    root_conversation_id: str


@dataclass
class SlotBudget:
    hard_limit: int = AGY_HARD_CONVERSATION_LIMIT
    reserve_slots: int = DEFAULT_RESERVE_SLOTS
    max_total_dbs: int = AGY_HARD_CONVERSATION_LIMIT - DEFAULT_RESERVE_SLOTS
    existing_disk_dbs: int = 0
    existing_native_dbs: int = 0
    existing_migrated_dbs: int = 0
    available_new_slots: int = 0
    planned_new_dbs: int = 0
    planned_overwrite_dbs: int = 0
    projected_total_dbs: int = 0
    trimmed_roots_by_budget: int = 0
    trimmed_subagents_by_budget: int = 0
    cutoff_updated_at_ms: int = 0
    budget_exceeded_explicit: bool = False


@dataclass
class MigrationPlan:
    project_actions: List[ProjectMergeAction]
    new_project_order_ids: List[str]
    codex_to_agy_project_id: Dict[str, str]
    agy_project_names: Dict[str, str]
    agy_project_roots: Dict[str, List[str]]
    thread_resolutions: Dict[str, ThreadResolution]
    selected_threads: List[CodexThread]
    skipped_counts: Dict[str, int]
    subagent_mode: str = "inline"
    inline_subagents_by_root: Dict[str, List[CodexThread]] = field(default_factory=dict)
    all_subagent_threads: Dict[str, CodexThread] = field(default_factory=dict)
    budget: SlotBudget = field(default_factory=SlotBudget)


def plan_project_merges(
    codex_projects: List[CodexProject],
    existing_agy_projects: Dict[str, AgyProject],
    global_state: CodexGlobalState,
    git_cache: GitProbeCache,
) -> Tuple[List[ProjectMergeAction], List[str], Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """Match Codex projects to existing agy projects by normalized folder path and plan merges/creates."""
    # Map normalized folder path -> existing non-sidecar AgyProject
    folder_to_agy: Dict[str, AgyProject] = {}
    agy_names: Dict[str, str] = {}
    agy_roots: Dict[str, List[str]] = {}

    for pid, ap in existing_agy_projects.items():
        if ap.is_sidecar:
            continue
        agy_names[pid] = ap.name
        agy_roots[pid] = list(ap.folder_paths)
        for fp in ap.folder_paths:
            folder_to_agy[os.path.normpath(fp)] = ap

    actions: List[ProjectMergeAction] = []
    codex_to_agy: Dict[str, str] = {}
    created_in_order: List[str] = []

    for cp in codex_projects:
        matched_agy: Optional[AgyProject] = None
        for r in cp.roots:
            nr = os.path.normpath(r)
            if nr in folder_to_agy:
                matched_agy = folder_to_agy[nr]
                break

        if matched_agy is not None:
            codex_to_agy[cp.id] = matched_agy.id
            existing_set = {os.path.normpath(fp) for fp in agy_roots.get(matched_agy.id, [])}
            missing_roots = [os.path.normpath(r) for r in cp.roots if os.path.normpath(r) not in existing_set]
            if missing_roots:
                merged_json = copy.deepcopy(matched_agy.raw)
                res_obj = merged_json.setdefault("projectResources", {})
                res_list = res_obj.setdefault("resources", [])
                for mr in missing_roots:
                    ginfo = git_cache.probe(mr)
                    res_list.append(make_project_resource_entry(mr, ginfo))
                    agy_roots.setdefault(matched_agy.id, []).append(mr)
                    folder_to_agy[mr] = matched_agy
                actions.append(
                    ProjectMergeAction(
                        codex_project_id=cp.id,
                        codex_project_name=cp.name,
                        target_agy_project_id=matched_agy.id,
                        target_agy_project_name=matched_agy.name,
                        action="merge_add_roots",
                        added_roots=missing_roots,
                        all_roots=list(agy_roots[matched_agy.id]),
                        project_json=merged_json,
                    )
                )
            else:
                actions.append(
                    ProjectMergeAction(
                        codex_project_id=cp.id,
                        codex_project_name=cp.name,
                        target_agy_project_id=matched_agy.id,
                        target_agy_project_name=matched_agy.name,
                        action="reuse",
                        added_roots=[],
                        all_roots=list(agy_roots.get(matched_agy.id, [])),
                        project_json=None,
                    )
                )
        else:
            # Create new agy project reusing Codex project's UUID (idempotent!)
            target_id = cp.id
            resources = [
                make_project_resource_entry(os.path.normpath(r), git_cache.probe(r))
                for r in cp.roots
            ]
            new_json: Dict[str, Any] = {
                "id": target_id,
                "name": cp.name,
                "projectResources": {"resources": resources},
                "permissionGrants": {"v2Migrated": True},
                "settings": {},
                "isWorkspaceOnly": False,
            }
            codex_to_agy[cp.id] = target_id
            agy_names[target_id] = cp.name
            agy_roots[target_id] = [os.path.normpath(r) for r in cp.roots]
            new_ap = AgyProject(
                id=target_id,
                name=cp.name,
                raw=new_json,
                is_sidecar=False,
                folder_paths=list(agy_roots[target_id]),
            )
            for r in agy_roots[target_id]:
                folder_to_agy[r] = new_ap
            created_in_order.append(target_id)
            actions.append(
                ProjectMergeAction(
                    codex_project_id=cp.id,
                    codex_project_name=cp.name,
                    target_agy_project_id=target_id,
                    target_agy_project_name=cp.name,
                    action="create",
                    added_roots=list(agy_roots[target_id]),
                    all_roots=list(agy_roots[target_id]),
                    project_json=new_json,
                )
            )

    # Order newly created projects according to Codex's project-order if available
    if global_state.project_order:
        ordered_app_ids: List[str] = []
        for lpid in global_state.project_order:
            aid = global_state.legacy_to_app_project_id.get(lpid, lpid)
            mapped = codex_to_agy.get(aid)
            if mapped in created_in_order and mapped not in ordered_app_ids:
                ordered_app_ids.append(mapped)
        for pid in created_in_order:
            if pid not in ordered_app_ids:
                ordered_app_ids.append(pid)
        created_in_order = ordered_app_ids

    return actions, created_in_order, codex_to_agy, agy_names, agy_roots


def build_migration_plan(
    *,
    codex_projects: List[CodexProject],
    codex_threads: List[CodexThread],
    global_state: CodexGlobalState,
    existing_agy_projects: Dict[str, AgyProject],
    thread_turn_counts: Dict[str, Tuple[int, int]],
    subagent_mode: Optional[str] = None,
    include_subagents: Optional[bool] = None,
    include_guardian: bool = False,
    include_exec: bool = False,
    include_empty: bool = False,
    only_thread_ids: Optional[List[str]] = None,
    project_filter: Optional[str] = None,
    limit: Optional[int] = None,
    existing_disk_cids: Optional[set[str]] = None,
    existing_any_cids: Optional[set[str]] = None,
    manifest_convos: Optional[Dict[str, Any]] = None,
    update_existing: bool = False,
    force_update: bool = False,
    hard_limit: int = AGY_HARD_CONVERSATION_LIMIT,
    reserve_slots: int = DEFAULT_RESERVE_SLOTS,
    enforce_slot_budget: bool = True,
) -> MigrationPlan:
    if subagent_mode is None:
        if include_subagents is True:
            subagent_mode = "separate"
        elif include_subagents is False:
            subagent_mode = "skip"
        else:
            subagent_mode = "inline"
    if subagent_mode not in SUBAGENT_MODES:
        raise ValueError(f"Unsupported subagent_mode: {subagent_mode!r}; expected one of {SUBAGENT_MODES}")

    disk_cids = set(existing_disk_cids or ())
    any_cids = set(existing_any_cids) if existing_any_cids is not None else set(disk_cids)
    manifest_map = manifest_convos or {}

    git_cache = GitProbeCache()
    actions, new_order_ids, codex_to_agy, agy_names, agy_roots = plan_project_merges(
        codex_projects, existing_agy_projects, global_state, git_cache
    )

    # Reverse map: normalized root path -> agy_project_id
    root_to_agy_pid: Dict[str, str] = {}
    basename_to_agy_pids: Dict[str, List[str]] = {}
    for agy_pid, roots in agy_roots.items():
        for r in roots:
            nr = os.path.normpath(r)
            root_to_agy_pid[nr] = agy_pid
            bname = os.path.basename(nr)
            if bname:
                basename_to_agy_pids.setdefault(bname, [])
                if agy_pid not in basename_to_agy_pids[bname]:
                    basename_to_agy_pids[bname].append(agy_pid)

    codex_docs_prefix = os.path.normpath(str(Path.home() / "Documents" / "Codex")) + os.sep
    threads_by_id: Dict[str, CodexThread] = {t.id: t for t in codex_threads}

    def make_workspace_for(agy_pid: str, thread: CodexThread) -> List[WorkspaceInfo]:
        if agy_pid == OUTSIDE_OF_PROJECT_ID:
            return []
        roots = agy_roots.get(agy_pid) or []
        chosen = ""
        ncwd = os.path.normpath(thread.cwd) if thread.cwd else ""
        for r in roots:
            if ncwd == r or (ncwd and ncwd.startswith(r + os.sep)):
                chosen = r
                break
        if not chosen and thread.worktree_source_cwd:
            for r in roots:
                if thread.worktree_source_cwd == r:
                    chosen = r
                    break
        if not chosen and roots:
            chosen = roots[0]
        if not chosen:
            return []

        ginfo = git_cache.probe(chosen)
        uri = path_to_file_uri(chosen)
        if ginfo.is_git:
            origin = thread.git_origin_url or ginfo.origin_url
            branch = thread.git_branch or ginfo.default_branch
            return [
                WorkspaceInfo(
                    folder_uri=uri,
                    git_root_uri=uri,
                    repo_computed_name=compute_repo_name(origin, chosen),
                    git_origin_url=origin,
                    branch_name=branch,
                )
            ]
        return [WorkspaceInfo(folder_uri=uri)]

    # Resolve top-level threads first
    resolutions: Dict[str, ThreadResolution] = {}

    def resolve_root_thread(t: CodexThread) -> ThreadResolution:
        # 1. Explicit DB project_id
        if t.db_project_id and t.db_project_id in codex_to_agy:
            apid = codex_to_agy[t.db_project_id]
            return ThreadResolution(
                project_id=apid,
                project_name=agy_names.get(apid, ""),
                workspaces=make_workspace_for(apid, t),
                reason="db.project_id",
                root_conversation_id=t.id,
            )
        # 2. Global state thread-project-assignments
        tpa_pid = global_state.thread_project_assignments.get(t.id)
        if tpa_pid and tpa_pid in codex_to_agy:
            apid = codex_to_agy[tpa_pid]
            return ThreadResolution(
                project_id=apid,
                project_name=agy_names.get(apid, ""),
                workspaces=make_workspace_for(apid, t),
                reason="global_state.tpa",
                root_conversation_id=t.id,
            )
        # 3. Explicit projectless
        ncwd = os.path.normpath(t.cwd) if t.cwd else ""
        if t.id in global_state.projectless_thread_ids or ncwd.startswith(codex_docs_prefix):
            return ThreadResolution(
                project_id=OUTSIDE_OF_PROJECT_ID,
                project_name="",
                workspaces=[],
                reason="projectless",
                root_conversation_id=t.id,
            )
        # 4. Workspace root hint from global state
        hint = global_state.workspace_root_hints.get(t.id)
        if hint:
            nhint = os.path.normpath(hint)
            if nhint in root_to_agy_pid:
                apid = root_to_agy_pid[nhint]
                return ThreadResolution(
                    project_id=apid,
                    project_name=agy_names.get(apid, ""),
                    workspaces=make_workspace_for(apid, t),
                    reason="workspace_hint",
                    root_conversation_id=t.id,
                )
        # 5. Worktree attachment sourceCwd
        if t.worktree_source_cwd and t.worktree_source_cwd in root_to_agy_pid:
            apid = root_to_agy_pid[t.worktree_source_cwd]
            return ThreadResolution(
                project_id=apid,
                project_name=agy_names.get(apid, ""),
                workspaces=make_workspace_for(apid, t),
                reason="worktree_attachment",
                root_conversation_id=t.id,
            )
        # 6. Exact cwd match to project root
        if ncwd and ncwd in root_to_agy_pid:
            apid = root_to_agy_pid[ncwd]
            return ThreadResolution(
                project_id=apid,
                project_name=agy_names.get(apid, ""),
                workspaces=make_workspace_for(apid, t),
                reason="cwd_exact",
                root_conversation_id=t.id,
            )
        # 7. Subdirectory of a project root (longest matching root wins)
        if ncwd:
            best_root = ""
            best_pid = ""
            for r, apid in root_to_agy_pid.items():
                if ncwd.startswith(r + os.sep) and len(r) > len(best_root):
                    best_root = r
                    best_pid = apid
            if best_pid:
                return ThreadResolution(
                    project_id=best_pid,
                    project_name=agy_names.get(best_pid, ""),
                    workspaces=make_workspace_for(best_pid, t),
                    reason="cwd_subdir",
                    root_conversation_id=t.id,
                )
        # 8. Codex worktree path ~/.codex/worktrees/<name>/<repo_basename>
        if "/.codex/worktrees/" in ncwd.replace("\\", "/"):
            bname = os.path.basename(ncwd)
            matches = basename_to_agy_pids.get(bname, [])
            if len(matches) == 1:
                apid = matches[0]
                return ThreadResolution(
                    project_id=apid,
                    project_name=agy_names.get(apid, ""),
                    workspaces=make_workspace_for(apid, t),
                    reason="worktree_basename",
                    root_conversation_id=t.id,
                )
        # 9. Fallback: standalone (outside-of-project)
        return ThreadResolution(
            project_id=OUTSIDE_OF_PROJECT_ID,
            project_name="",
            workspaces=[],
            reason="unmatched_standalone",
            root_conversation_id=t.id,
        )

    for t in codex_threads:
        if not t.is_subagent:
            resolutions[t.id] = resolve_root_thread(t)

    # Resolve subagent threads by walking parent chain to root
    def find_root_ancestor(tid: str) -> Optional[str]:
        seen = set()
        cur_id = tid
        while cur_id in threads_by_id and cur_id not in seen:
            seen.add(cur_id)
            ct = threads_by_id[cur_id]
            if not ct.is_subagent:
                return ct.id
            if not ct.parent_thread_id:
                return None
            cur_id = ct.parent_thread_id
        return None

    for t in codex_threads:
        if t.is_thread_spawn:
            root_id = find_root_ancestor(t.id)
            if root_id and root_id in resolutions:
                parent_res = resolutions[root_id]
                resolutions[t.id] = ThreadResolution(
                    project_id=parent_res.project_id,
                    project_name=parent_res.project_name,
                    workspaces=parent_res.workspaces,
                    reason="inherit_parent",
                    root_conversation_id=root_id,
                )
            else:
                resolutions[t.id] = resolve_root_thread(t)

    # Filter threads according to options
    skipped: Dict[str, int] = {
        "guardian": 0,
        "subagent_other": 0,
        "subagent_disabled": 0,
        "subagent_inlined": 0,
        "exec": 0,
        "empty": 0,
        "filtered_out": 0,
        "budget": 0,
    }

    only_set = set(only_thread_ids) if only_thread_ids else None
    all_subagent_threads: Dict[str, CodexThread] = {t.id: t for t in codex_threads if t.is_thread_spawn}

    def _is_empty_thread(t: CodexThread) -> bool:
        turns_cnt, items_cnt = thread_turn_counts.get(t.id, (0, 0))
        return turns_cnt == 0 and items_cnt == 0 and not t.has_user_event and not t.preview

    def _thread_action_kind(t: CodexThread) -> str:
        """Classify a thread relative to current Antigravity disk DBs and migration manifest.

        Returns one of: 'create', 'overwrite', 'skip_migrated', 'skip_native'.
        """
        if t.id in manifest_map and t.id in disk_cids:
            prev_ts = manifest_map[t.id].get("updated_at_ms", 0)
            if update_existing and (force_update or t.updated_at_ms > prev_ts):
                return "overwrite"
            return "skip_migrated"
        if t.id in any_cids and t.id not in manifest_map:
            return "skip_native"
        if t.id in disk_cids:
            return "skip_native"
        return "create"

    # Pass 1: filter candidate root threads
    candidate_roots: List[CodexThread] = []
    for t in codex_threads:
        if t.is_guardian:
            if not include_guardian:
                skipped["guardian"] += 1
                continue
        elif t.is_thread_spawn:
            continue  # handled in pass 2
        elif t.is_subagent:
            skipped["subagent_other"] += 1
            continue

        if t.source_kind == "exec" and not include_exec:
            if only_set is None or t.id not in only_set:
                skipped["exec"] += 1
                continue

        if _is_empty_thread(t) and not include_empty:
            if only_set is None or t.id not in only_set:
                skipped["empty"] += 1
                continue

        if only_set is not None and t.id not in only_set:
            skipped["filtered_out"] += 1
            continue

        if project_filter:
            res = resolutions.get(t.id)
            pf_low = project_filter.lower()
            if not res or (
                res.project_id.lower() != pf_low
                and res.project_name.lower() != pf_low
                and not (pf_low in ("standalone", "outside-of-project", "普通对话") and res.project_id == OUTSIDE_OF_PROJECT_ID)
            ):
                skipped["filtered_out"] += 1
                continue

        candidate_roots.append(t)

    # Always sort candidate roots in reverse chronological order (most recently active first)
    candidate_roots.sort(key=lambda x: (x.updated_at_ms, x.created_at_ms, x.id), reverse=True)

    # Apply explicit --limit (newest N roots)
    if limit is not None and limit >= 0:
        extra_filtered = max(0, len(candidate_roots) - limit)
        skipped["filtered_out"] += extra_filtered
        candidate_roots = candidate_roots[:limit]

    candidate_root_ids = {t.id for t in candidate_roots}

    # Pre-group valid non-empty child subagents per candidate root (depth-ordered so parent chain is intact)
    sub_candidates = [t for t in codex_threads if t.is_thread_spawn]
    sub_candidates.sort(key=lambda x: (x.nesting_depth, x.created_at_ms, x.id))
    valid_subs_by_root: Dict[str, List[CodexThread]] = {}
    explicit_only_subs: List[CodexThread] = []
    valid_tree_ids: set[str] = set(candidate_root_ids)

    for t in sub_candidates:
        res = resolutions.get(t.id)
        root_id = res.root_conversation_id if res else ""
        is_explicit_only = only_set is not None and t.id in only_set
        if not is_explicit_only:
            if root_id not in candidate_root_ids:
                skipped["filtered_out"] += 1
                continue
            if t.parent_thread_id and t.parent_thread_id not in valid_tree_ids:
                skipped["filtered_out"] += 1
                continue
        if _is_empty_thread(t) and not include_empty and not is_explicit_only:
            skipped["empty"] += 1
            continue
        valid_tree_ids.add(t.id)
        if is_explicit_only and root_id not in candidate_root_ids:
            explicit_only_subs.append(t)
        elif root_id:
            valid_subs_by_root.setdefault(root_id, []).append(t)

    # Compute Antigravity 500-conversation slot budget
    eff_reserve = max(0, reserve_slots)
    max_total_dbs = max(0, min(hard_limit, hard_limit - eff_reserve))
    existing_migrated_on_disk = sum(1 for cid in disk_cids if cid in manifest_map)
    existing_native_on_disk = len(disk_cids) - existing_migrated_on_disk
    available_new_slots = max(0, max_total_dbs - len(disk_cids))

    # Pass 2: select roots (and subagents) under the slot budget in reverse chronological order
    selected_roots: List[CodexThread] = []
    selected_subagents: List[CodexThread] = []
    inline_subagents_by_root: Dict[str, List[CodexThread]] = {}

    planned_new_dbs = 0
    planned_overwrite_dbs = 0
    trimmed_roots_by_budget = 0
    trimmed_subagents_by_budget = 0
    cutoff_updated_at_ms = 0
    budget_exceeded_explicit = False

    for r in candidate_roots:
        r_kind = _thread_action_kind(r)
        r_subs = valid_subs_by_root.get(r.id, [])

        r_new_cost = 1 if r_kind == "create" else 0
        r_overwrite_cnt = 1 if r_kind == "overwrite" else 0
        if subagent_mode == "separate":
            subs_new_cost = sum(1 for s in r_subs if _thread_action_kind(s) == "create")
            subs_overwrite_cnt = sum(1 for s in r_subs if _thread_action_kind(s) == "overwrite")
        else:
            subs_new_cost = 0
            subs_overwrite_cnt = 0

        total_new_cost = r_new_cost + subs_new_cost

        if enforce_slot_budget and only_set is None and total_new_cost > 0:
            if planned_new_dbs + total_new_cost > available_new_slots:
                trimmed_roots_by_budget += 1
                trimmed_subagents_by_budget += len(r_subs)
                skipped["budget"] += 1
                continue

        selected_roots.append(r)
        planned_new_dbs += total_new_cost
        planned_overwrite_dbs += r_overwrite_cnt + subs_overwrite_cnt
        if total_new_cost > 0:
            cutoff_updated_at_ms = r.updated_at_ms

        if subagent_mode == "separate":
            selected_subagents.extend(r_subs)
        elif subagent_mode == "inline":
            if r_subs:
                inline_subagents_by_root[r.id] = r_subs
                skipped["subagent_inlined"] += len(r_subs)
        else:  # skip
            skipped["subagent_disabled"] += len(r_subs)

    # Handle any explicitly requested --only subagent IDs whose root wasn't in --only
    if explicit_only_subs:
        if subagent_mode == "separate":
            for s in explicit_only_subs:
                sk = _thread_action_kind(s)
                if sk == "create":
                    planned_new_dbs += 1
                elif sk == "overwrite":
                    planned_overwrite_dbs += 1
                selected_subagents.append(s)
        elif subagent_mode == "skip":
            skipped["subagent_disabled"] += len(explicit_only_subs)
        else:
            skipped["subagent_inlined"] += len(explicit_only_subs)

    if enforce_slot_budget and only_set is not None and planned_new_dbs > available_new_slots:
        budget_exceeded_explicit = True

    # Keep roots newest-first; subagents by nesting_depth ascending
    selected_roots.sort(key=lambda x: (x.updated_at_ms, x.created_at_ms, x.id), reverse=True)
    selected_subagents.sort(key=lambda x: (x.nesting_depth, x.created_at_ms, x.id))

    budget = SlotBudget(
        hard_limit=hard_limit,
        reserve_slots=eff_reserve,
        max_total_dbs=max_total_dbs,
        existing_disk_dbs=len(disk_cids),
        existing_native_dbs=existing_native_on_disk,
        existing_migrated_dbs=existing_migrated_on_disk,
        available_new_slots=available_new_slots,
        planned_new_dbs=planned_new_dbs,
        planned_overwrite_dbs=planned_overwrite_dbs,
        projected_total_dbs=len(disk_cids) + planned_new_dbs,
        trimmed_roots_by_budget=trimmed_roots_by_budget,
        trimmed_subagents_by_budget=trimmed_subagents_by_budget,
        cutoff_updated_at_ms=cutoff_updated_at_ms,
        budget_exceeded_explicit=budget_exceeded_explicit,
    )

    return MigrationPlan(
        project_actions=actions,
        new_project_order_ids=new_order_ids,
        codex_to_agy_project_id=codex_to_agy,
        agy_project_names=agy_names,
        agy_project_roots=agy_roots,
        thread_resolutions=resolutions,
        selected_threads=selected_roots + selected_subagents,
        skipped_counts=skipped,
        subagent_mode=subagent_mode,
        inline_subagents_by_root=inline_subagents_by_root,
        all_subagent_threads=all_subagent_threads,
        budget=budget,
    )
