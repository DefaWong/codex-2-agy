"""Antigravity (agy) protobuf message builders matching extracted FileDescriptorProtos.

All field numbers and enum values are verified directly against Antigravity's
cortex.proto, jetski_cortex.proto, codeium_common.proto, diff_action.proto,
and tools.proto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .protobuf_wire import (
    decode_fields,
    encode_any,
    encode_duration_ms,
    encode_str_map_entries,
    encode_timestamp_ms,
    f_bool,
    f_bytes,
    f_msg,
    f_str,
    f_varint,
    join_fields,
)

# --- Enums ---

# exa.cortex_pb.CortexStepType
STEP_TYPE_USER_INPUT = 14
STEP_TYPE_PLANNER_RESPONSE = 15
STEP_TYPE_ERROR_MESSAGE = 17
STEP_TYPE_SYSTEM_MESSAGE = 101
STEP_TYPE_GENERIC = 132

# exa.cortex_pb.CortexStepStatus
STEP_STATUS_PENDING = 1
STEP_STATUS_RUNNING = 2
STEP_STATUS_DONE = 3
STEP_STATUS_CANCELED = 6
STEP_STATUS_ERROR = 7
STEP_STATUS_GENERATING = 8

# exa.cortex_pb.CortexStepSource
STEP_SOURCE_MODEL = 2
STEP_SOURCE_USER_IMPLICIT = 3
STEP_SOURCE_USER_EXPLICIT = 4
STEP_SOURCE_SYSTEM = 5

# exa.cortex_pb.CascadeRunStatus
CASCADE_RUN_STATUS_IDLE = 1
CASCADE_RUN_STATUS_RUNNING = 2

# exa.cortex_pb.CortexTrajectoryType
TRAJECTORY_TYPE_CASCADE = 4

# exa.cortex_pb.CortexTrajectorySource
TRAJECTORY_SOURCE_CASCADE_CLIENT = 1
TRAJECTORY_SOURCE_SUBAGENT = 16

# exa.diff_action_pb.UnifiedDiffLineType
DIFF_LINE_INSERT = 1
DIFF_LINE_DELETE = 2
DIFF_LINE_UNCHANGED = 3

# exa.cortex.core.tools.FileEdit.Kind
FILE_EDIT_KIND_CREATE = 1
FILE_EDIT_KIND_OVERWRITE = 2
FILE_EDIT_KIND_EDIT = 3

# exa.cortex_pb.ExecutorTerminationReason
# 1 = ERROR, 2 = USER_CANCELED (renders circle-X), 4 = NO_TOOL_CALL (normal completion, renders circle-check)
TERMINATION_REASON_ROOT_DONE = 4
TERMINATION_REASON_SUBAGENT_DONE = 4

# Special project_id for standalone (non-project) conversations in agy
OUTSIDE_OF_PROJECT_ID = "outside-of-project"


# --- Common Messages ---


@dataclass
class WorkspaceInfo:
    folder_uri: str
    git_root_uri: str = ""
    repo_computed_name: str = ""
    git_origin_url: str = ""
    branch_name: str = ""


def encode_repository(computed_name: str = "", git_origin_url: str = "") -> bytes:
    return join_fields(
        f_str(1, computed_name),
        f_str(2, git_origin_url),
    )


def encode_workspace_metadata(ws: WorkspaceInfo) -> bytes:
    repo = encode_repository(ws.repo_computed_name, ws.git_origin_url)
    return join_fields(
        f_str(1, ws.folder_uri),
        f_str(2, ws.git_root_uri),
        f_msg(3, repo),
        f_str(4, ws.branch_name),
    )


@dataclass
class MediaAttachment:
    mime_type: str
    uri: str
    display_name: str = ""
    media_id: str = ""
    description: str = ""
    inline_data: bytes = b""


def encode_media(m: MediaAttachment) -> bytes:
    """exa.codeium_common_pb.Media."""
    return join_fields(
        f_str(1, m.mime_type),
        f_bytes(2, m.inline_data),
        f_str(4, m.description),
        f_str(5, m.uri),
        f_str(8, m.media_id),
        f_str(9, m.display_name),
    )


@dataclass
class ToolCallSpec:
    id: str
    name: str
    arguments_json: str


def encode_chat_tool_call(tc: ToolCallSpec) -> bytes:
    """exa.codeium_common_pb.ChatToolCall."""
    return join_fields(
        f_str(1, tc.id),
        f_str(2, tc.name),
        f_str(3, tc.arguments_json),
    )


@dataclass
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    thinking_output_tokens: int = 0
    response_output_tokens: int = 0
    message_id: str = ""
    response_id: str = ""


def encode_model_usage_stats(u: UsageStats | None) -> bytes:
    """exa.codeium_common_pb.ModelUsageStats."""
    if not u:
        return b""
    if not (u.input_tokens or u.output_tokens or u.cache_read_tokens or u.thinking_output_tokens):
        return b""
    return join_fields(
        f_varint(2, u.input_tokens),
        f_varint(3, u.output_tokens),
        f_varint(4, u.cache_write_tokens),
        f_varint(5, u.cache_read_tokens),
        f_str(7, u.message_id),
        f_varint(9, u.thinking_output_tokens),
        f_varint(10, u.response_output_tokens),
        f_str(11, u.response_id),
    )


# --- Step Metadata & Error Details ---


def encode_source_trajectory_step_info(
    trajectory_id: str,
    step_index: int,
    cascade_id: str,
    metadata_index: int = 0,
) -> bytes:
    return join_fields(
        f_str(1, trajectory_id),
        f_varint(2, step_index),
        f_varint(3, metadata_index),
        f_str(4, cascade_id),
    )


def encode_status_transitions(transitions: List[Tuple[int, int]]) -> bytes:
    """exa.cortex_pb.CortexStepInternalMetadata { repeated StatusTransition status_transitions = 1; }."""
    parts = []
    for status, ts_ms in transitions:
        tr = join_fields(
            f_varint(1, status),
            f_msg(2, encode_timestamp_ms(ts_ms)),
        )
        parts.append(f_msg(1, tr))
    return b"".join(parts)


def encode_step_metadata(
    *,
    created_at_ms: int,
    started_at_ms: int,
    completed_at_ms: int,
    source: int,
    execution_id: str,
    trajectory_id: str,
    step_index: int,
    cascade_id: str,
    content_id: str,
    status: int = STEP_STATUS_DONE,
    tool_call: ToolCallSpec | None = None,
    usage: UsageStats | None = None,
    tool_source: int = 0,
) -> bytes:
    """exa.cortex_pb.CortexStepMetadata."""
    transitions = [(status, completed_at_ms or created_at_ms)]
    return join_fields(
        f_msg(1, encode_timestamp_ms(created_at_ms)),
        f_varint(3, source),
        f_msg(4, encode_chat_tool_call(tool_call)) if tool_call else b"",
        f_msg(6, encode_timestamp_ms(started_at_ms or created_at_ms)),
        f_msg(7, encode_timestamp_ms(completed_at_ms or created_at_ms)),
        f_msg(8, encode_timestamp_ms(completed_at_ms or created_at_ms)),
        f_msg(9, encode_model_usage_stats(usage)),
        f_str(12, execution_id),
        f_msg(
            20,
            encode_source_trajectory_step_info(trajectory_id, step_index, cascade_id),
        ),
        f_msg(26, encode_status_transitions(transitions)),
        f_msg(32, encode_timestamp_ms(started_at_ms or created_at_ms)),
        f_str(35, content_id),
        f_varint(38, tool_source),
    )


def encode_error_details(short_error: str, full_error: str = "") -> bytes:
    """exa.cortex_pb.CortexErrorDetails."""
    full = full_error or short_error
    return join_fields(
        f_str(1, short_error),
        f_str(2, short_error),
        f_str(3, full),
    )


# --- Step Body Builders ---


@dataclass
class FileScopeRef:
    absolute_path: str
    absolute_uri: str
    num_bytes: int = 0


def encode_path_scope_item(ref: FileScopeRef) -> bytes:
    """exa.codeium_common_pb.PathScopeItem."""
    return join_fields(
        f_str(1, ref.absolute_path),
        f_varint(4, ref.num_bytes),
        f_str(5, ref.absolute_uri),
    )


def encode_text_or_scope_file(ref: FileScopeRef) -> bytes:
    """exa.codeium_common_pb.TextOrScopeItem { ContextScopeItem item = 2 { PathScopeItem file = 1 } }."""
    scope_item = f_msg(1, encode_path_scope_item(ref))
    return f_msg(2, scope_item)


def encode_user_input(
    text: str,
    *,
    media: List[MediaAttachment] | None = None,
    file_refs: List[FileScopeRef] | None = None,
    client_message_id: str = "",
) -> bytes:
    """exa.cortex_pb.CortexStepUserInput."""
    parts = [f_str(2, text)]
    if file_refs:
        for ref in file_refs:
            parts.append(f_msg(3, encode_text_or_scope_file(ref)))
    if text:
        parts.append(f_msg(3, f_str(1, text), omit_empty=False))
    if media:
        for m in media:
            parts.append(f_msg(9, encode_media(m)))
    if client_message_id:
        parts.append(f_str(19, client_message_id))
    return b"".join(parts)


def encode_planner_response(
    *,
    response: str = "",
    thinking: str = "",
    message_id: str = "",
    tool_calls: List[ToolCallSpec] | None = None,
    thinking_duration_ms: int = 0,
    media: List[MediaAttachment] | None = None,
) -> bytes:
    """exa.cortex_pb.CortexStepPlannerResponse."""
    parts = [
        f_str(1, response),
        f_str(3, thinking),
        f_str(6, message_id),
    ]
    if tool_calls:
        for tc in tool_calls:
            parts.append(f_msg(7, encode_chat_tool_call(tc)))
    if response:
        parts.append(f_str(8, response))
    if thinking_duration_ms > 0:
        parts.append(f_msg(11, encode_duration_ms(thinking_duration_ms)))
    if media:
        for m in media:
            parts.append(f_msg(20, encode_media(m)))
    return b"".join(parts)


def encode_agent_message(
    *,
    msg_id: str,
    sender: str,
    recipient: str,
    content: str,
    timestamp_ms: int,
    priority: int = 2,
) -> bytes:
    """exa.cortex_pb.AgentMessage."""
    return join_fields(
        f_str(1, msg_id),
        f_str(2, recipient),
        f_str(3, sender),
        f_str(4, content),
        f_varint(5, priority),
        f_msg(6, encode_timestamp_ms(timestamp_ms)),
    )


def encode_system_message(
    message: str,
    *,
    event_type: str = "",
    agent_message: bytes = b"",
    render_info: bytes = b"",
    media: List[MediaAttachment] | None = None,
) -> bytes:
    """exa.cortex_pb.CortexStepSystemMessage."""
    parts = [
        f_str(1, message),
        f_msg(2, render_info),
        f_str(3, event_type),
        f_msg(4, agent_message),
    ]
    if media:
        for m in media:
            parts.append(f_msg(6, encode_media(m)))
    return b"".join(parts)


# --- StepRenderInfo & Tool Payloads (CortexStepGeneric) ---


def encode_diff_stats(additions: int, deletions: int) -> bytes:
    """exa.cortex_pb.DiffStats."""
    return join_fields(
        f_varint(1, additions),
        f_varint(2, deletions),
    )


def encode_title_part_text(text: str) -> bytes:
    return f_str(1, text)


def encode_title_part_file(
    uri: str,
    *,
    start_line: int = 0,
    end_line: int = 0,
    total_lines: int = 0,
    total_bytes: int = 0,
) -> bytes:
    fref = join_fields(
        f_str(1, uri),
        f_varint(2, start_line),
        f_varint(3, end_line),
        f_varint(4, total_lines),
        f_varint(5, total_bytes),
    )
    return f_msg(2, fref)


def encode_title_part_badge(icon: str, label: str = "", tooltip: str = "") -> bytes:
    badge = join_fields(f_str(1, icon), f_str(2, label), f_str(3, tooltip))
    return f_msg(3, badge)


def encode_title_part_file_diff(uri: str, additions: int, deletions: int) -> bytes:
    fdiff = join_fields(
        f_str(1, uri),
        f_msg(2, encode_diff_stats(additions, deletions)),
    )
    return f_msg(4, fdiff)


def encode_agent_ref(
    conversation_id: str,
    title: str = "",
    subtitle: str = "",
    log_uri: str = "",
    workspace_uris: List[str] | None = None,
) -> bytes:
    parts = [
        f_str(1, conversation_id),
        f_str(2, title),
        f_str(3, subtitle),
        f_str(4, log_uri),
    ]
    if workspace_uris:
        for u in workspace_uris:
            parts.append(f_str(5, u))
    return b"".join(parts)


def encode_title_part_agent(
    conversation_id: str,
    title: str = "",
    subtitle: str = "",
    log_uri: str = "",
    workspace_uris: List[str] | None = None,
) -> bytes:
    return f_msg(5, encode_agent_ref(conversation_id, title, subtitle, log_uri, workspace_uris))


def encode_content_block_markdown(md: str) -> bytes:
    return f_str(1, md)


def encode_content_block_media(uri: str, mime_type: str = "", alt: str = "") -> bytes:
    mref = join_fields(f_str(1, uri), f_str(2, mime_type), f_str(3, alt))
    return f_msg(2, mref)


def encode_content_block_terminal(
    command: str,
    cwd: str = "",
    exit_code: int = 0,
    output: str = "",
) -> bytes:
    tb = join_fields(
        f_str(1, command),
        f_str(2, cwd),
        f_varint(3, exit_code, omit_zero=False),
        f_str(4, output),
    )
    return f_msg(3, tb)


def encode_content_block_agent(
    conversation_id: str,
    title: str = "",
    subtitle: str = "",
    log_uri: str = "",
    workspace_uris: List[str] | None = None,
) -> bytes:
    return f_msg(4, encode_agent_ref(conversation_id, title, subtitle, log_uri, workspace_uris))


def encode_step_render_info(
    *,
    title: str = "",
    title_prefix: str = "",
    markdown: str = "",
    hidden: bool = False,
    icon: str = "",
    aggregation_key: str = "",
    title_parts: List[bytes] | None = None,
    content_blocks: List[bytes] | None = None,
    show_on_error: bool = False,
) -> bytes:
    """exa.cortex_pb.StepRenderInfo."""
    parts = [
        f_str(1, title),
        f_str(2, markdown),
        f_bool(4, hidden),
        f_str(5, title_prefix),
        f_str(6, icon),
        f_str(7, aggregation_key),
    ]
    if title_parts:
        for tp in title_parts:
            parts.append(f_msg(9, tp))
    if content_blocks:
        for cb in content_blocks:
            parts.append(f_msg(10, cb))
    parts.append(f_bool(11, show_on_error))
    return b"".join(parts)


# --- Typed Tool Payloads (exa.cortex.core.tools.*) ---


def encode_tool_run_command(
    command_line: str,
    cwd: str = "",
    exit_code: int = 0,
    shell_name: str = "zsh",
) -> bytes:
    inner = join_fields(
        f_varint(1, exit_code),
        f_str(5, shell_name),
        f_str(6, command_line),
        f_str(7, cwd),
    )
    return encode_any("type.googleapis.com/exa.cortex.core.tools.RunCommand", inner)


def encode_tool_read_file(
    path: str,
    *,
    file_num_lines: int = 0,
    file_num_bytes: int = 0,
    start_line: int = 0,
    end_line: int = 0,
    line_range_bytes: int = 0,
) -> bytes:
    inner = join_fields(
        f_varint(1, file_num_lines),
        f_varint(2, file_num_bytes),
        f_str(3, path),
        f_varint(4, start_line),
        f_varint(5, end_line),
        f_varint(6, line_range_bytes),
    )
    return encode_any("type.googleapis.com/exa.cortex.core.tools.ReadFile", inner)


def encode_tool_file_edit(
    path: str,
    *,
    kind: int,
    diff_lines: List[Tuple[str, int]],
    additions: int,
    deletions: int,
    start_line: int = 1,
    end_line: int = 0,
) -> bytes:
    line_msgs = []
    for text, ltype in diff_lines:
        ln = join_fields(
            f_str(1, text),
            f_varint(2, ltype),
        )
        line_msgs.append(f_msg(3, ln))
    unified_diff = b"".join(line_msgs)
    diff_block = join_fields(
        f_varint(1, start_line) if start_line > 1 else b"",
        f_varint(2, end_line or len(diff_lines)),
        f_msg(3, unified_diff),
    )
    inner = join_fields(
        f_varint(1, kind),
        f_str(2, path),
        f_msg(3, diff_block),
        f_msg(4, encode_diff_stats(additions, deletions)),
    )
    return encode_any("type.googleapis.com/exa.cortex.core.tools.FileEdit", inner)


def encode_tool_mcp(
    server_name: str,
    tool_call: ToolCallSpec,
    result_string: str = "",
    result_uri: str = "",
) -> bytes:
    inner = join_fields(
        f_str(1, server_name),
        f_msg(2, encode_chat_tool_call(tool_call)),
        f_str(3, result_string),
        f_str(4, result_uri),
    )
    return encode_any("type.googleapis.com/exa.cortex.core.tools.McpTool", inner)


def encode_tool_search_web(
    query: str,
    results: List[Tuple[str, str, str]],
    summary: str = "",
) -> bytes:
    """results: list of (url, title, snippet)."""
    parts = [f_str(1, query)]
    for idx, (url, title, snippet) in enumerate(results):
        kb = join_fields(
            f_str(1, url or f"result-{idx}"),
            f_str(3, url),
            f_str(4, title),
            f_str(7, snippet),
        )
        parts.append(f_msg(3, kb))
    if summary:
        parts.append(f_str(4, summary))
    return encode_any("type.googleapis.com/exa.cortex.core.tools.SearchWeb", b"".join(parts))


def encode_tool_generate_image(
    prompt: str,
    image_paths: List[str],
    image_uri: str = "",
    image_num_bytes: int = 0,
    image_name: str = "",
) -> bytes:
    parts = [f_str(1, prompt)]
    for p in image_paths:
        parts.append(f_str(2, p))
    parts.append(f_str(4, image_uri))
    parts.append(f_varint(5, image_num_bytes))
    parts.append(f_str(8, image_name))
    return encode_any("type.googleapis.com/exa.cortex.core.tools.GenerateImage", b"".join(parts))


def encode_subagent_spec(
    type_name: str,
    role: str,
    initial_prompt: str = "",
    inherit: bool = True,
    workspace_uri: str = "",
) -> bytes:
    """exa.cortex_pb.SubagentSpec."""
    return join_fields(
        f_str(1, type_name),
        f_str(2, role),
        f_str(3, initial_prompt),
        f_bool(4, inherit),
        f_str(6, workspace_uri),
    )


def encode_tool_invoke_subagent(
    *,
    type_name: str,
    role: str,
    initial_prompt: str,
    conversation_id: str,
    log_uri: str = "",
    workspace_uris: List[str] | None = None,
) -> bytes:
    spec = encode_subagent_spec(
        type_name=type_name,
        role=role,
        initial_prompt=initial_prompt,
        inherit=True,
    )
    res_parts = [f_str(1, conversation_id), f_str(2, log_uri)]
    if workspace_uris:
        for u in workspace_uris:
            res_parts.append(f_str(3, u))
    inner = join_fields(
        f_msg(1, spec),
        f_msg(2, b"".join(res_parts)),
    )
    return encode_any("type.googleapis.com/exa.cortex.core.tools.InvokeSubagent", inner)


def encode_tool_send_message(recipient: str, recipient_name: str, message: str) -> bytes:
    inner = join_fields(
        f_str(1, recipient),
        f_str(2, recipient_name),
        f_str(3, message),
    )
    return encode_any("type.googleapis.com/exa.cortex.core.tools.SendMessage", inner)


def encode_tool_manage_subagents(action: str, conversation_ids: List[str]) -> bytes:
    parts = [f_str(1, action)]
    for cid in conversation_ids:
        parts.append(f_str(2, cid))
    return encode_any("type.googleapis.com/exa.cortex.core.tools.ManageSubagents", b"".join(parts))


@dataclass
class AskQuestionEntryData:
    question: str
    options: List[Tuple[str, str]] = field(default_factory=list)  # [(id, text), ...]
    is_multi_select: bool = False
    selected_option_ids: List[str] = field(default_factory=list)
    write_in_response: str = ""
    skipped: bool = False


def encode_ask_question_entry(entry: AskQuestionEntryData, *, include_answer: bool = True) -> bytes:
    """exa.cortex_pb.AskQuestionEntry."""
    parts = [f_str(1, entry.question)]
    for opt_id, opt_text in entry.options:
        opt_msg = join_fields(f_str(1, opt_id), f_str(2, opt_text))
        parts.append(f_msg(2, opt_msg))
    if entry.is_multi_select:
        parts.append(f_bool(3, True))
    if include_answer:
        for sid in entry.selected_option_ids:
            parts.append(f_str(4, sid))
        if entry.write_in_response:
            parts.append(f_str(5, entry.write_in_response))
        if entry.skipped:
            parts.append(f_bool(6, True))
    return b"".join(parts)


def encode_tool_ask_question(entries: List[AskQuestionEntryData]) -> bytes:
    """type.googleapis.com/exa.cortex.core.tools.AskQuestion."""
    inner = b"".join(f_msg(1, encode_ask_question_entry(e, include_answer=True)) for e in entries)
    return encode_any("type.googleapis.com/exa.cortex.core.tools.AskQuestion", inner)


def encode_completed_ask_question_interaction(
    entries: List[AskQuestionEntryData],
    *,
    trajectory_id: str,
    step_index: int,
) -> bytes:
    """exa.cortex_pb.CompletedInteraction (CortexStep field 147) for native ask_question UI."""
    req_spec = b"".join(f_msg(1, encode_ask_question_entry(e, include_answer=False)) for e in entries)
    requested_interaction = f_msg(22, req_spec)

    resp_aq = b"".join(f_msg(1, encode_ask_question_entry(e, include_answer=True)) for e in entries)
    user_interaction = join_fields(
        f_str(1, trajectory_id),
        f_varint(2, step_index, omit_zero=False),
        f_msg(22, resp_aq),
    )
    return join_fields(
        f_msg(1, requested_interaction),
        f_msg(2, user_interaction),
    )


def encode_generic_step(
    *,
    args: Dict[str, str],
    result_text: str,
    render_info: bytes = b"",
    payload_any: bytes = b"",
    full_output_uri: str = "",
    media: List[MediaAttachment] | None = None,
) -> bytes:
    """exa.cortex_pb.CortexStepGeneric."""
    res_parts = [
        f_str(1, result_text),
        f_msg(3, render_info),
    ]
    if media:
        for m in media:
            res_parts.append(f_msg(4, encode_media(m)))
    res_parts.append(f_msg(6, payload_any))
    res_parts.append(f_str(7, full_output_uri))
    result_msg = b"".join(res_parts)
    return join_fields(
        encode_str_map_entries(1, args.items()),
        f_msg(2, result_msg),
    )


def encode_cortex_step(
    *,
    step_type: int,
    status: int,
    metadata_bytes: bytes,
    body_field_number: int,
    body_bytes: bytes,
    completed_interaction_bytes: bytes = b"",
) -> bytes:
    """exa.cortex_pb.CortexStep."""
    return join_fields(
        f_varint(1, step_type),
        f_varint(4, status),
        f_msg(5, metadata_bytes),
        f_msg(147, completed_interaction_bytes),
        f_msg(body_field_number, body_bytes),
    )


# --- Trajectory Metadata, Annotations & Summary ---


def encode_trajectory_metadata(
    *,
    workspaces: List[WorkspaceInfo],
    created_at_ms: int,
    initialization_state_id: str,
    root_conversation_id: str,
    project_id: str,
    parent_conversation_id: str = "",
    nesting_depth: int = 0,
    subagent_type_name: str = "",
    subagent_role: str = "",
    subagent_prompt: str = "",
) -> bytes:
    """exa.cortex_pb.CortexTrajectoryMetadata."""
    parts = []
    for ws in workspaces:
        parts.append(f_msg(1, encode_workspace_metadata(ws)))
    parts.append(f_msg(2, encode_timestamp_ms(created_at_ms)))
    parts.append(f_str(3, initialization_state_id))
    if subagent_type_name:
        # field 4: agent_script { name = 1 } (observed in native agy subagents)
        parts.append(f_msg(4, f_str(1, subagent_type_name)))
    if parent_conversation_id:
        parts.append(f_str(5, parent_conversation_id))
    parts.append(f_str(6, root_conversation_id))
    for ws in workspaces:
        parts.append(f_str(7, ws.folder_uri))
    if subagent_type_name:
        parts.append(
            f_msg(
                8,
                encode_subagent_spec(
                    type_name=subagent_type_name,
                    role=subagent_role or subagent_type_name,
                    initial_prompt=subagent_prompt,
                    inherit=True,
                ),
            )
        )
    if nesting_depth > 0:
        parts.append(f_varint(17, nesting_depth))
    parts.append(f_str(18, project_id))
    return b"".join(parts)


def encode_conversation_annotations(
    *,
    title: str = "",
    last_user_view_time_ms: int = 0,
    archived: bool = False,
    pinned: bool = False,
    tags: List[str] | None = None,
) -> bytes:
    """exa.jetski_cortex_pb.ConversationAnnotations."""
    parts = [f_str(1, title)]
    if tags:
        for t in tags:
            parts.append(f_str(3, t))
    parts.append(f_bool(4, archived))
    parts.append(f_msg(7, encode_timestamp_ms(last_user_view_time_ms)))
    parts.append(f_bool(12, pinned))
    return b"".join(parts)


def encode_annotations_pbtxt(
    *,
    title: str = "",
    last_user_view_time_ms: int = 0,
    archived: bool = False,
    pinned: bool = False,
) -> str:
    """Encode ~/.gemini/antigravity/annotations/<cid>.pbtxt."""
    tokens: List[str] = []
    if title:
        escaped = (
            title.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "")
        )
        tokens.append(f'title:"{escaped}"')
    if archived:
        tokens.append("archived:true")
    if last_user_view_time_ms > 0:
        sec = last_user_view_time_ms // 1000
        nanos = (last_user_view_time_ms % 1000) * 1_000_000
        tokens.append(f"last_user_view_time:{{seconds:{sec} nanos:{nanos}}}")
    if pinned:
        tokens.append("pinned:true")
    return " ".join(tokens)


def encode_cascade_trajectory_summary(
    *,
    summary: str,
    step_count: int,
    last_modified_ms: int,
    trajectory_id: str,
    created_ms: int,
    workspaces: List[WorkspaceInfo],
    last_user_input_ms: int,
    last_user_input_step_index: int,
    annotations_bytes: bytes,
    trajectory_metadata_bytes: bytes,
    is_subagent: bool = False,
) -> bytes:
    """exa.jetski_cortex_pb.CascadeTrajectorySummary."""
    parts = [
        f_str(1, summary),
        f_varint(2, step_count),
        f_msg(3, encode_timestamp_ms(last_modified_ms)),
        f_str(4, trajectory_id),
        f_varint(5, CASCADE_RUN_STATUS_IDLE),
        f_msg(7, encode_timestamp_ms(created_ms)),
    ]
    for ws in workspaces:
        parts.append(f_msg(9, encode_workspace_metadata(ws)))
    if last_user_input_ms > 0:
        parts.append(f_msg(10, encode_timestamp_ms(last_user_input_ms)))
    if annotations_bytes:
        parts.append(f_msg(15, annotations_bytes))
    if last_user_input_step_index > 0 and not is_subagent:
        parts.append(f_varint(16, last_user_input_step_index))
    parts.append(f_msg(17, trajectory_metadata_bytes))
    parts.append(f_varint(22, TRAJECTORY_TYPE_CASCADE))
    if is_subagent:
        parts.append(f_bool(23, False, omit_false=False))
    parts.append(
        f_varint(
            28,
            TERMINATION_REASON_SUBAGENT_DONE if is_subagent else TERMINATION_REASON_ROOT_DONE,
        )
    )
    return b"".join(parts)


# --- agyhub_summaries_proto.pb (CascadeTrajectorySummaries) ---


def decode_agyhub_summaries(raw: bytes) -> Dict[str, bytes]:
    """Decode agyhub_summaries_proto.pb -> {conversation_id: raw_summary_bytes}."""
    result: Dict[str, bytes] = {}
    if not raw:
        return result
    for fnum, wtype, val in decode_fields(raw):
        if fnum == 1 and isinstance(val, bytes):
            cid = ""
            summary_bytes = b""
            for sf, swt, sv in decode_fields(val):
                if sf == 1 and isinstance(sv, bytes):
                    cid = sv.decode("utf-8", "replace")
                elif sf == 2 and isinstance(sv, bytes):
                    summary_bytes = sv
            if cid:
                result[cid] = summary_bytes
    return result


def encode_agyhub_summaries(entries: Dict[str, bytes]) -> bytes:
    """Encode {conversation_id: raw_summary_bytes} -> CascadeTrajectorySummaries protobuf."""
    parts = []
    for cid, raw_summary in entries.items():
        entry = join_fields(
            f_str(1, cid, omit_empty=False),
            f_bytes(2, raw_summary, omit_empty=False),
        )
        parts.append(f_msg(1, entry, omit_empty=False))
    return b"".join(parts)
