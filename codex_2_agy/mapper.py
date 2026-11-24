"""Lossless Codex thread -> Antigravity MappedConversation mapper.

Maps all 16 Codex item types into native Antigravity CortexStep records with:
- Strict PLANNER_RESPONSE.tool_calls <-> GENERIC step pairing (for seamless continuation)
- Native UI cards (TerminalBlock, FileEdit + UnifiedDiff, SearchWeb, McpTool,
  ReadFile, GenerateImage, InvokeSubagent + AgentRef, SendMessage)
- Recovered base64 images from .jsonl.zst rollouts saved to brain/<cid>/.user_uploaded/
- Per-turn token usage stats from rollout token_usage_record
- Full untruncated outputs spilled to .system_generated/steps/<idx>/output.txt when large
- Native transcript.jsonl / transcript_full.jsonl entries
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agy_proto import (
    DIFF_LINE_DELETE,
    DIFF_LINE_INSERT,
    DIFF_LINE_UNCHANGED,
    FILE_EDIT_KIND_CREATE,
    FILE_EDIT_KIND_EDIT,
    OUTSIDE_OF_PROJECT_ID,
    STEP_SOURCE_MODEL,
    STEP_SOURCE_SYSTEM,
    STEP_SOURCE_USER_EXPLICIT,
    STEP_STATUS_DONE,
    STEP_STATUS_ERROR,
    STEP_TYPE_GENERIC,
    STEP_TYPE_PLANNER_RESPONSE,
    STEP_TYPE_SYSTEM_MESSAGE,
    STEP_TYPE_USER_INPUT,
    AskQuestionEntryData,
    FileScopeRef,
    MediaAttachment,
    ToolCallSpec,
    UsageStats,
    WorkspaceInfo,
    encode_agent_message,
    encode_completed_ask_question_interaction,
    encode_content_block_agent,
    encode_content_block_markdown,
    encode_content_block_media,
    encode_content_block_terminal,
    encode_cortex_step,
    encode_error_details,
    encode_generic_step,
    encode_planner_response,
    encode_step_metadata,
    encode_step_render_info,
    encode_system_message,
    encode_title_part_agent,
    encode_title_part_file,
    encode_title_part_file_diff,
    encode_title_part_text,
    encode_tool_ask_question,
    encode_tool_file_edit,
    encode_tool_generate_image,
    encode_tool_invoke_subagent,
    encode_tool_manage_subagents,
    encode_tool_mcp,
    encode_tool_read_file,
    encode_tool_run_command,
    encode_tool_search_web,
    encode_tool_send_message,
    encode_user_input,
)
from .agy_store import (
    AgyPaths,
    EncodedStep,
    ExtractedAsset,
    MappedConversation,
    file_uri_to_path,
    path_to_file_uri,
)
from .codex_reader import (
    DATA_URI_RE,
    CodexItem,
    CodexThread,
    RolloutImage,
    RolloutUsage,
    ThreadHistoryData,
)

# Fixed UUID namespace for deterministic, idempotent trajectory/init IDs
CODEX_2_AGY_NS = uuid.UUID("7f3b5c42-9c9f-4e1a-9a3b-6c4e2d8f1b0a")

INLINE_OUTPUT_LIMIT = 32_768
TERMINAL_PREVIEW_LIMIT = 8_000
MAX_DIFF_LINES_IN_PROTO = 4_000

MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/svg+xml": ".svg",
}


def deterministic_uuid(kind: str, seed: str) -> str:
    return str(uuid.uuid5(CODEX_2_AGY_NS, f"{kind}:{seed}"))


def deterministic_hex32(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def ms_to_iso_z(ms: int) -> str:
    if ms <= 0:
        return "1970-01-01T00:00:00Z"
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ms_to_local_iso(ms: int) -> str:
    if ms <= 0:
        return "1970-01-01T00:00:00+00:00"
    return datetime.fromtimestamp(ms / 1000.0).astimezone().isoformat(timespec="seconds")


QUESTION_REPLY_RE = re.compile(
    r"<send_user_message_question_reply>\s*(.*?)\s*</send_user_message_question_reply>",
    re.DOTALL,
)
BROWSER_CONTEXT_RE = re.compile(
    r"<in-app-browser-context\b[^>]*>.*?</in-app-browser-context>\s*",
    re.DOTALL,
)
FILES_PREAMBLE_RE = re.compile(
    r"^\s*#\s*Files\s+(?:mentioned|pasted)\s+by\s+the\s+user:\s*\n(.*?)\n##\s*My\s+request(?:\s+for\s+Codex)?:\s*\n?(.*)$",
    re.DOTALL,
)
FILES_PREAMBLE_ONLY_RE = re.compile(
    r"^\s*#\s*Files\s+(?:mentioned|pasted)\s+by\s+the\s+user:\s*\n(.*)$",
    re.DOTALL,
)
FILE_ENTRY_RE = re.compile(r"^##\s+(.+?):\s*(/\S.*?)\s*$", re.MULTILINE)


def _format_question_reply_match(m: re.Match[str]) -> str:
    raw = m.group(1).strip()
    try:
        arr = json.loads(raw)
    except Exception:
        return m.group(0)
    if not isinstance(arr, list) or not arr:
        return m.group(0)
    blocks: List[str] = []
    multi = len(arr) > 1
    for i, item in enumerate(arr, start=1):
        if not isinstance(item, dict):
            continue
        q = str(item.get("question") or "").strip()
        ans = item.get("answer")
        if isinstance(ans, list):
            a = ", ".join(str(x) for x in ans).strip()
        else:
            a = str(ans or "").strip()
        prefix_q = f"Q{i}:" if multi else "Q:"
        prefix_a = f"A{i}:" if multi else "A:"
        if q and a:
            blocks.append(f"{prefix_q} {q}\n{prefix_a} {a}")
        elif a:
            blocks.append(a)
        elif q:
            blocks.append(f"{prefix_q} {q}")
    return "\n\n".join(blocks) if blocks else m.group(0)


def parse_codex_question_replies(
    raw_text: str,
    question_specs_by_call_id: Dict[str, List[Dict[str, Any]]] | None = None,
) -> Tuple[List[AskQuestionEntryData], str, str]:
    """Extract <send_user_message_question_reply> blocks into native AskQuestionEntryData list.

    Returns (entries, primary_call_id, remaining_text).
    """
    specs_map = question_specs_by_call_id or {}
    entries: List[AskQuestionEntryData] = []
    primary_call_id = ""

    def _consume(m: re.Match[str]) -> str:
        nonlocal primary_call_id
        raw = m.group(1).strip()
        try:
            arr = json.loads(raw)
        except Exception:
            return m.group(0)
        if not isinstance(arr, list) or not arr:
            return m.group(0)
        parsed_any = False
        for item in arr:
            if not isinstance(item, dict):
                continue
            q_text = str(item.get("question") or "").strip()
            ans_raw = item.get("answer")
            if isinstance(ans_raw, list):
                ans_list = [str(x).strip() for x in ans_raw if str(x).strip()]
                ans_str = ", ".join(ans_list)
            else:
                ans_str = str(ans_raw or "").strip()
                ans_list = [ans_str] if ans_str else []

            # Parse questionItemId: e.g. '["request_user_input_async","call_abc",0]'
            call_id = ""
            q_idx = 0
            qid_raw = item.get("questionItemId")
            if isinstance(qid_raw, str):
                try:
                    qid_arr = json.loads(qid_raw)
                    if isinstance(qid_arr, list) and len(qid_arr) >= 2:
                        call_id = str(qid_arr[1] or "")
                        if len(qid_arr) >= 3 and isinstance(qid_arr[2], int):
                            q_idx = qid_arr[2]
                except Exception:
                    pass
            if call_id and not primary_call_id:
                primary_call_id = call_id

            # Recover original multiple-choice options from rollout if available
            opt_texts: List[str] = []
            is_multi = False
            spec_list = specs_map.get(call_id) or []
            if 0 <= q_idx < len(spec_list):
                spec = spec_list[q_idx]
                is_multi = bool(spec.get("is_multi_select") or spec.get("multiSelect") or spec.get("allowMultiple"))
                for opt in spec.get("options") or []:
                    if isinstance(opt, str) and opt.strip():
                        opt_texts.append(opt.strip())
                    elif isinstance(opt, dict):
                        lbl = str(opt.get("label") or opt.get("text") or opt.get("title") or "").strip()
                        if lbl:
                            opt_texts.append(lbl)

            options_pairs = [(str(idx), txt) for idx, txt in enumerate(opt_texts, start=1)]
            selected_ids: List[str] = []
            write_in = ""
            if options_pairs and ans_list:
                opt_by_text = {txt: oid for oid, txt in options_pairs}
                matched_all = True
                for a_val in ans_list:
                    if a_val in opt_by_text:
                        selected_ids.append(opt_by_text[a_val])
                    else:
                        matched_all = False
                        break
                if not matched_all:
                    selected_ids = []
                    write_in = ans_str
            else:
                write_in = ans_str

            entries.append(
                AskQuestionEntryData(
                    question=q_text,
                    options=options_pairs,
                    is_multi_select=is_multi,
                    selected_option_ids=selected_ids,
                    write_in_response=write_in,
                    skipped=(not selected_ids and not write_in),
                )
            )
            parsed_any = True
        return "" if parsed_any else m.group(0)

    remaining = QUESTION_REPLY_RE.sub(_consume, raw_text).strip()
    return entries, primary_call_id, remaining


def format_codex_user_text(
    raw_text: str,
    *,
    attached_image_paths: set[str] | None = None,
    codex_home: Path | None = None,
    attach_file_media_fn=None,
    out_file_refs: List[FileScopeRef] | None = None,
) -> str:
    """Clean Codex App internal XML/preamble wrappers into readable, lossless Antigravity user text.

    - Already-attached images have their redundant temp-path preamble stripped.
    - Pasted files in ~/.codex/attachments/ are attached via `attach_file_media_fn` as native
      Antigravity Media chips (never exposing ~/.gemini/antigravity/brain/... paths in the text!).
    - External files mentioned by the user are attached as native PathScopeItem chips (`out_file_refs`)
      referencing their original external paths.
    """
    s = (raw_text or "").strip()
    if not s:
        return ""

    # 1. Remove ambient in-app-browser-context boilerplate (not written by the user)
    s = BROWSER_CONTEXT_RE.sub("", s).strip()

    # 2. Format <send_user_message_question_reply> JSON into clean plain Q&A (fallback path)
    if "<send_user_message_question_reply>" in s:
        s = QUESTION_REPLY_RE.sub(_format_question_reply_match, s).strip()

    # 3. Unwrap '# Files mentioned/pasted by the user: ... [## My request[ for Codex]:]'
    m = FILES_PREAMBLE_RE.match(s)
    m_only = None if m else FILES_PREAMBLE_ONLY_RE.match(s)
    if m or m_only:
        header_block = m.group(1) if m else m_only.group(1)  # type: ignore[union-attr]
        body = m.group(2).strip() if m else ""
        attached_norm = {os.path.normpath(p) for p in (attached_image_paths or set()) if p}
        att_root = (
            os.path.normpath(str(codex_home / "attachments")) + os.sep
            if codex_home is not None
            else ""
        )
        kept_refs: List[str] = []

        # Split header_block by '## ' entries so we can inspect each entry's attributes
        sections = re.split(r"(?m)(?=^##\s+)", header_block)
        for sec in sections:
            sec_s = sec.strip()
            if not sec_s.startswith("## "):
                continue
            em = FILE_ENTRY_RE.match(sec_s)
            if not em:
                continue
            label = em.group(1).strip().strip('"')
            fpath = em.group(2).strip()
            npath = os.path.normpath(fpath)
            ext = os.path.splitext(npath)[1].lower()
            is_img_ext = ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
            already_attached_img = (
                "Image attachment: true" in sec_s
                or npath in attached_norm
                or (
                    is_img_ext
                    and (
                        "codex-clipboard-" in os.path.basename(npath)
                        or "/com.wiheads.paste/" in npath
                        or bool(attached_norm)
                    )
                    and not (att_root and npath.startswith(att_root) and os.path.isfile(npath))
                )
            )
            if already_attached_img:
                # Already attached natively as a MediaAttachment thumbnail above the user bubble
                continue

            # File stored inside ~/.codex/attachments/ (pasted text or persisted image attachment)
            if att_root and npath.startswith(att_root) and os.path.isfile(npath):
                if attach_file_media_fn is not None:
                    if attach_file_media_fn(Path(npath), label):
                        continue
                # Fallback if no media callback provided (e.g. unit test)
                kept_refs.append(f"📎 {label}")
                continue

            # External file referenced by the user -> native PathScopeItem chip with original external path!
            if out_file_refs is not None:
                fsize = os.path.getsize(npath) if os.path.isfile(npath) else 0
                out_file_refs.append(
                    FileScopeRef(
                        absolute_path=npath,
                        absolute_uri=path_to_file_uri(npath),
                        num_bytes=fsize,
                    )
                )
                continue

            kept_refs.append(f"📎 {label} ({fpath})")

        if kept_refs and body:
            s = "\n".join(kept_refs) + "\n\n" + body
        elif body:
            s = body
        elif kept_refs:
            s = "\n".join(kept_refs)
        else:
            s = ""

    return s.strip()


def clean_codex_title(raw: str) -> str:
    """Clean Codex's '# Files mentioned by the user:' preamble or question-reply XML if it leaked into title/preview."""
    s = (raw or "").strip()
    s = BROWSER_CONTEXT_RE.sub("", s).strip()
    if "<send_user_message_question_reply>" in s:
        s = QUESTION_REPLY_RE.sub(_format_question_reply_match, s).strip()
    for marker in ("## My request for Codex:", "## My request:"):
        if marker in s:
            after = s.split(marker, 1)[1].strip()
            if after:
                s = after
                break
    # Collapse newlines and trim
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 100:
        s = s[:97].rstrip() + "..."
    return s


# --- Codex Assistant Directive & Reasoning Normalization ---

_CODEX_DIRECTIVE_START_RE = re.compile(
    r"::(code-comment|codex-file-citation|git-stage|git-commit|git-create-branch|codex-realtime-inline)\{"
)
_HTML_COMMENT_SEP_RE = re.compile(r"<!--\s*-->")
_PRIORITY_PREFIX_RE = re.compile(r"^\[?P[0-9]+\]?\b", re.IGNORECASE)


def _find_matching_brace(text: str, open_idx: int) -> int:
    """Find matching '}' for '{' at open_idx, respecting single/double quoted strings and escapes."""
    n = len(text)
    i = open_idx + 1
    in_quote = ""
    while i < n:
        ch = text[i]
        if in_quote:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == in_quote:
                in_quote = ""
        else:
            if ch in ('"', "'"):
                in_quote = ch
            elif ch == "}":
                return i
        i += 1
    return -1


def _parse_directive_attrs(raw_attrs: str) -> Dict[str, str]:
    """Parse key=value / key=\"...\" attributes inside a Codex ::directive{...}."""
    attrs: Dict[str, str] = {}
    n = len(raw_attrs)
    i = 0
    while i < n:
        while i < n and raw_attrs[i].isspace():
            i += 1
        if i >= n:
            break
        k_start = i
        while i < n and (raw_attrs[i].isalnum() or raw_attrs[i] in ("_", "-")):
            i += 1
        key = raw_attrs[k_start:i]
        if not key:
            i += 1
            continue
        while i < n and raw_attrs[i].isspace():
            i += 1
        if i >= n or raw_attrs[i] != "=":
            attrs[key] = ""
            continue
        i += 1  # skip '='
        while i < n and raw_attrs[i].isspace():
            i += 1
        if i >= n:
            attrs[key] = ""
            break
        if raw_attrs[i] in ('"', "'"):
            q = raw_attrs[i]
            i += 1
            buf: List[str] = []
            while i < n:
                ch = raw_attrs[i]
                if ch == "\\" and i + 1 < n:
                    nxt = raw_attrs[i + 1]
                    if nxt == "n":
                        buf.append("\n")
                    elif nxt == "t":
                        buf.append("\t")
                    elif nxt == "r":
                        pass
                    else:
                        buf.append(nxt)
                    i += 2
                    continue
                if ch == q:
                    i += 1
                    break
                buf.append(ch)
                i += 1
            attrs[key] = "".join(buf)
        else:
            v_start = i
            while i < n and not raw_attrs[i].isspace():
                i += 1
            attrs[key] = raw_attrs[v_start:i]
    return attrs


def _render_codex_directive(name: str, attrs: Dict[str, str]) -> Tuple[str, bool]:
    """Render a Codex App directive into standard Markdown.

    Returns (replacement_text, is_block_element).
    """
    if name in ("git-stage", "git-commit", "git-create-branch", "codex-realtime-inline"):
        # Pure Codex Desktop interactive UI action button placeholders
        return "", False

    if name == "code-comment":
        title = (attrs.get("title") or "").strip()
        body = (attrs.get("body") or "").strip()
        fpath = (attrs.get("file") or attrs.get("path") or "").strip()
        start = (attrs.get("start") or attrs.get("start_line") or "").strip()
        end = (attrs.get("end") or attrs.get("end_line") or "").strip()
        priority = (attrs.get("priority") or "").strip()

        if priority and not _PRIORITY_PREFIX_RE.match(title):
            p_tag = priority.upper() if priority.upper().startswith("P") else f"P{priority}"
            title = f"[{p_tag}] {title}".strip()
        if not title:
            title = "Code Review Comment"

        loc_link = ""
        if fpath:
            bname = os.path.basename(fpath.rstrip("/\\")) or fpath
            frag = ""
            if start and end and start != end:
                loc_label = f"{bname}:{start}-{end}"
                frag = f"#L{start}-L{end}"
            elif start:
                loc_label = f"{bname}:{start}"
                frag = f"#L{start}"
            else:
                loc_label = bname
            loc_link = f" — [{loc_label}]({path_to_file_uri(fpath)}{frag})"

        lines = [f"> **💬 {title}**{loc_link}"]
        if body:
            lines.append(">")
            for bline in body.splitlines():
                lines.append(f"> {bline}" if bline.strip() else ">")
        return "\n".join(lines), True

    if name == "codex-file-citation":
        fpath = (attrs.get("path") or attrs.get("file") or "").strip()
        if not fpath:
            return "", False
        label = (attrs.get("title") or os.path.basename(fpath.rstrip("/\\")) or fpath).strip()
        slide = (attrs.get("slide_number") or "").strip()
        page = (attrs.get("page_number") or "").strip()
        start = (attrs.get("start_line") or attrs.get("line") or "").strip()
        end = (attrs.get("end_line") or "").strip()
        frag = ""
        if slide:
            label = f"{label} (Slide {slide})"
        elif page:
            label = f"{label} (Page {page})"
        elif start and end and start != end:
            label = f"{label}:{start}-{end}"
            frag = f"#L{start}-L{end}"
        elif start:
            label = f"{label}:{start}"
            frag = f"#L{start}"
        return f"- 📎 [{label}]({path_to_file_uri(fpath)}{frag})", True

    return "", False


def _replace_directives_in_segment(seg: str) -> str:
    """Replace Codex ::directive{...} occurrences outside inline backtick spans."""
    if "::" not in seg:
        return seg
    out: List[str] = []
    n = len(seg)
    pos = 0
    while pos < n:
        m = _CODEX_DIRECTIVE_START_RE.search(seg, pos)
        if not m:
            out.append(seg[pos:])
            break
        start_idx = m.start()
        prefix = seg[pos:start_idx]
        # Skip if inside an inline backtick code span on the same line
        last_nl = seg.rfind("\n", 0, start_idx)
        line_before = seg[last_nl + 1 : start_idx]
        if line_before.count("`") % 2 == 1:
            out.append(seg[pos : m.end()])
            pos = m.end()
            continue

        open_brace = m.end() - 1
        close_brace = _find_matching_brace(seg, open_brace)
        if close_brace == -1:
            out.append(seg[pos : m.end()])
            pos = m.end()
            continue

        dname = m.group(1)
        attrs = _parse_directive_attrs(seg[open_brace + 1 : close_brace])
        rendered, is_block = _render_codex_directive(dname, attrs)

        if not rendered:
            # Strip pure UI action directive along with any standalone blank space around it
            out.append(prefix.rstrip(" \t"))
            pos = close_brace + 1
            while pos < n and seg[pos] in (" ", "\t"):
                pos += 1
            if pos < n and seg[pos] == "\n" and (not out or out[-1].endswith("\n") or out[-1] == ""):
                pos += 1
            continue

        if is_block:
            trimmed_prefix = prefix.rstrip(" \t")
            out.append(trimmed_prefix)
            prev_text = "".join(out)
            if prev_text and not prev_text.endswith("\n\n"):
                out.append("\n" if prev_text.endswith("\n") else "\n\n")
            out.append(rendered)
            pos = close_brace + 1
            # Consume trailing horizontal spaces before next directive/newline
            while pos < n and seg[pos] in (" ", "\t"):
                pos += 1
            if pos < n and seg[pos] != "\n":
                out.append("\n\n")
        else:
            out.append(prefix)
            out.append(rendered)
            pos = close_brace + 1

    return "".join(out)


def format_codex_assistant_text(raw: str) -> str:
    """Convert Codex App proprietary ::directive{...} syntax into standard Antigravity Markdown.

    Preserves fenced code blocks (``` / ~~~) and inline backtick code spans untouched.
    """
    if not raw or "::" not in raw:
        return (raw or "").strip()

    lines = raw.splitlines(keepends=True)
    result_parts: List[str] = []
    plain_buf: List[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0

    def flush_plain() -> None:
        if plain_buf:
            result_parts.append(_replace_directives_in_segment("".join(plain_buf)))
            plain_buf.clear()

    for line in lines:
        stripped = line.lstrip(" \t")
        if not in_fence:
            fm = re.match(r"^(`{3,}|~{3,})", stripped)
            if fm:
                flush_plain()
                in_fence = True
                fence_char = fm.group(1)[0]
                fence_len = len(fm.group(1))
                result_parts.append(line)
                continue
            plain_buf.append(line)
        else:
            result_parts.append(line)
            if re.match(rf"^{re.escape(fence_char)}{{{fence_len},}}[ \t]*$", stripped.rstrip("\r\n")):
                in_fence = False
    flush_plain()

    cleaned = "".join(result_parts)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_reasoning_parts(d: Dict[str, Any]) -> List[str]:
    """Extract, clean literal '<!-- -->' separators, and deduplicate Codex reasoning summary/content blocks."""
    raw_entries: List[str] = []
    for key in ("summary", "content"):
        val = d.get(key)
        if isinstance(val, list):
            for entry in val:
                txt = entry.get("text", "") if isinstance(entry, dict) else (entry if isinstance(entry, str) else "")
                if txt:
                    raw_entries.append(txt)
        elif isinstance(val, str) and val:
            raw_entries.append(val)
    return _dedupe_thinking_blocks(raw_entries)


def _dedupe_thinking_blocks(blocks: List[str]) -> List[str]:
    """Clean '<!-- -->' separators and deduplicate repeated reasoning paragraphs while preserving order."""
    out: List[str] = []
    seen: set[str] = set()
    for raw in blocks:
        cleaned = _HTML_COMMENT_SEP_RE.sub("", raw or "")
        for para in re.split(r"\n{2,}", cleaned):
            p = para.strip()
            if not p or p in seen:
                continue
            seen.add(p)
            out.append(p)
    return out


def _normalize_codex_items(items: List[CodexItem]) -> List[CodexItem]:
    """Fold late-flushed trailing same-turn reasoning items back before the preceding agentMessage.

    Codex's streaming logger occasionally persists a turn's final `reasoning` summary 80-400ms
    *after* `agentMessage(phase='final_answer')` in the same turn_id. Leaving it in place creates
    an orphan `Thought Process` block dangling below the final answer in Antigravity.
    """
    n = len(items)
    if n < 2:
        return list(items)
    out: List[CodexItem] = []
    i = 0
    while i < n:
        cur = items[i]
        if cur.item_type == "agentMessage":
            # Look ahead for contiguous same-turn reasoning items at the end of this turn
            j = i + 1
            while (
                j < n
                and items[j].item_type == "reasoning"
                and items[j].turn_id == cur.turn_id
            ):
                j += 1
            if j > i + 1:
                next_item = items[j] if j < n else None
                ends_turn = (
                    next_item is None
                    or next_item.turn_id != cur.turn_id
                    or next_item.item_type == "userMessage"
                )
                if ends_turn:
                    # Place late-flushed reasoning right before the agentMessage it belongs to
                    out.extend(items[i + 1 : j])
                    out.append(cur)
                    i = j
                    continue
        out.append(cur)
        i += 1
    return out


def pick_best_title(thread: CodexThread) -> str:
    if thread.is_thread_spawn:
        return ""
    for cand in (thread.name, thread.session_index_name, thread.title, thread.preview, thread.first_user_message):
        if cand and cand.strip():
            cleaned = clean_codex_title(cand)
            if cleaned:
                return cleaned
    return "Untitled Session"


def parse_unified_diff(diff_text: str, kind_type: str) -> Tuple[List[Tuple[str, int]], int, int]:
    """Convert Codex fileChange diff text into Antigravity UnifiedDiffLine tuples + (additions, deletions)."""
    raw_lines = diff_text.splitlines()
    lines: List[Tuple[str, int]] = []
    additions = 0
    deletions = 0

    if kind_type == "add":
        for ln in raw_lines:
            lines.append((ln, DIFF_LINE_INSERT))
            additions += 1
        return lines, additions, deletions

    if kind_type == "delete":
        for ln in raw_lines:
            lines.append((ln, DIFF_LINE_DELETE))
            deletions += 1
        return lines, additions, deletions

    # Standard unified diff (update)
    for ln in raw_lines:
        if ln.startswith("@@") and "@@" in ln[2:]:
            continue
        if ln.startswith("+++ ") or ln.startswith("--- "):
            continue
        if ln.startswith("\\ No newline at end of file"):
            continue
        if ln.startswith("+"):
            lines.append((ln[1:], DIFF_LINE_INSERT))
            additions += 1
        elif ln.startswith("-"):
            lines.append((ln[1:], DIFF_LINE_DELETE))
            deletions += 1
        elif ln.startswith(" "):
            lines.append((ln[1:], DIFF_LINE_UNCHANGED))
        else:
            lines.append((ln, DIFF_LINE_UNCHANGED))
    return lines, additions, deletions


def short_command_summary(cmd: str) -> str:
    s = (cmd or "").strip()
    # Strip /bin/zsh -lc "..." or /bin/bash -lc '...' wrapper from Codex commandExecution
    m = re.match(r'^/(?:usr/)?bin/(?:zsh|bash|sh)\s+-[a-z]*c\s+(["\'])(.*)\1$', s, re.DOTALL)
    if m:
        s = m.group(2).strip()
    first_line = s.splitlines()[0].strip() if s else "command"
    if len(first_line) > 80:
        first_line = first_line[:77] + "..."
    return first_line or "command"


def sanitize_filename(name: str, fallback: str = "asset") -> str:
    base = os.path.basename(name or "").strip()
    base = re.sub(r"[^a-zA-Z0-9._\-\u4e00-\u9fff]+", "_", base)
    return base[:80] or fallback


@dataclass
class MappedTool:
    call_id: str
    tool_name: str
    args_dict: Dict[str, str]
    arguments_json: str
    result_text: str
    full_output_text: Optional[str]  # non-None when spilled to steps/<idx>/output.txt
    render_info: bytes
    payload_any: bytes
    created_at_ms: int
    started_at_ms: int
    completed_at_ms: int
    is_error: bool = False
    error_message: str = ""
    media: List[MediaAttachment] = field(default_factory=list)
    ask_question_entries: Optional[List[AskQuestionEntryData]] = None


@dataclass
class InlineSubagentSummary:
    """Flattened representation of a Codex thread_spawn subagent for 0-slot inline embedding."""

    thread_id: str
    role: str
    agent_type: str
    prompt: str
    final_answer: str
    status: str
    step_summary_lines: List[str]
    markdown_transcript: str
    asset_rel_path: str
    created_at_ms: int = 0
    updated_at_ms: int = 0
    parent_thread_id: str = ""
    nesting_depth: int = 1


def subagent_display_role(thread: CodexThread) -> str:
    """Pick clean display role for a subagent thread (prefer semantic agent_path leaf)."""
    leaf = os.path.basename(thread.agent_path.rstrip("/")) if thread.agent_path else ""
    if leaf and leaf != "root":
        return leaf
    if thread.agent_role:
        return thread.agent_role
    if thread.agent_nickname:
        return thread.agent_nickname
    return "Subagent"


def _safe_filename_slug(name: str, fallback: str = "subagent") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._-")
    return (slug[:48] or fallback)


def build_inline_subagent_summary(
    thread: CodexThread,
    history: ThreadHistoryData,
) -> InlineSubagentSummary:
    """Build a concise inline summary + full lossless Markdown archive for a Codex subagent thread."""
    role = subagent_display_role(thread)
    agent_type = thread.agent_role or "self"
    short_id = thread.id[:8] if thread.id else "unknown"
    asset_rel_path = f"subagents/{_safe_filename_slug(role)}_{short_id}.md"

    prompt = ""
    final_answer = ""
    last_agent_msg = ""
    had_error = False
    summary_lines: List[str] = []

    cmd_count = 0
    edit_count = 0
    tool_count = 0

    md_sections: List[str] = [
        f"# Subagent: {role}",
        "",
        f"- **Thread ID**: `{thread.id}`",
        f"- **Parent ID**: `{thread.parent_thread_id or '-'}`",
        f"- **Depth**: `{thread.nesting_depth}`",
        f"- **Agent Type**: `{agent_type}`",
        f"- **Model**: `{thread.model or '-'}`",
        f"- **Created**: `{ms_to_local_iso(thread.created_at_ms)}`",
        f"- **Updated**: `{ms_to_local_iso(thread.updated_at_ms)}`",
        "",
    ]

    for idx, item in enumerate(_normalize_codex_items(history.items), start=1):
        d = item.data
        itype = item.item_type

        if itype == "userMessage":
            raw_parts: List[str] = []
            for part in d.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = str(part.get("text") or "").strip()
                    if t:
                        raw_parts.append(t)
            cleaned = format_codex_user_text("\n".join(raw_parts)).strip()
            if cleaned:
                if not prompt:
                    prompt = cleaned
                md_sections.extend([f"## [{idx}] Task / Input", "", cleaned, ""])

        elif itype == "reasoning":
            rparts = clean_reasoning_parts(d)
            if rparts:
                md_sections.extend([f"## [{idx}] Reasoning", "", "\n\n".join(rparts), ""])

        elif itype == "agentMessage":
            txt = format_codex_assistant_text(str(d.get("text") or ""))
            phase = str(d.get("phase") or "")
            if txt:
                last_agent_msg = txt
                if phase == "final_answer":
                    final_answer = txt
                heading = f"## [{idx}] Final Answer" if phase == "final_answer" else f"## [{idx}] Assistant"
                md_sections.extend([heading, "", txt, ""])

        elif itype == "commandExecution":
            cmd_count += 1
            cmd = str(d.get("command") or "").strip()
            cwd = str(d.get("cwd") or "")
            exit_code = d.get("exitCode")
            status = str(d.get("status") or "")
            out = str(d.get("aggregatedOutput") or "")
            if status in ("failed", "error") or (isinstance(exit_code, int) and exit_code != 0):
                had_error = True
            if len(summary_lines) < 12:
                short_cmd = cmd.splitlines()[0][:100] if cmd else "(command)"
                ec_tag = f" (exit {exit_code})" if exit_code is not None else ""
                summary_lines.append(f"- Run `{short_cmd}`{ec_tag}")
            md_sections.append(f"## [{idx}] Command (`exit={exit_code}`)")
            if cwd:
                md_sections.append(f"- CWD: `{cwd}`")
            md_sections.extend(["", "```bash", cmd, "```", ""])
            if out.strip():
                md_sections.extend(["```text", out[:40_000], "```", ""])

        elif itype == "fileChange":
            changes = d.get("changes") or []
            edit_count += len(changes)
            for ch in changes:
                if not isinstance(ch, dict):
                    continue
                fpath = str(ch.get("path") or "")
                kind_obj = ch.get("kind")
                kind = str(kind_obj.get("type") if isinstance(kind_obj, dict) else (kind_obj or "update"))
                diff_text = str(ch.get("diff") or "")
                if len(summary_lines) < 12 and fpath:
                    summary_lines.append(f"- Edit `{os.path.basename(fpath)}` ({kind})")
                md_sections.extend([f"## [{idx}] File {kind}: `{fpath}`", "", "```diff", diff_text[:40_000], "```", ""])

        elif itype == "webSearch":
            tool_count += 1
            q = str(d.get("query") or "")
            if len(summary_lines) < 12 and q:
                summary_lines.append(f"- Web search: `{q[:80]}`")
            md_sections.extend([f"## [{idx}] Web Search: `{q}`", ""])
            for res in (d.get("results") or [])[:10]:
                if isinstance(res, dict):
                    md_sections.append(f"- [{res.get('title') or res.get('url')}]({res.get('url') or ''})")
            md_sections.append("")

        elif itype in ("mcpToolCall", "dynamicToolCall", "collabAgentToolCall", "imageView", "imageGeneration"):
            tool_count += 1
            tlabel = str(d.get("tool") or d.get("name") or itype)
            if len(summary_lines) < 12:
                summary_lines.append(f"- Tool `{tlabel}`")
            raw_dump = json.dumps(d, ensure_ascii=False, indent=2)[:20_000]
            md_sections.extend([f"## [{idx}] Tool `{tlabel}` ({itype})", "", "```json", raw_dump, "```", ""])

        elif itype == "plan":
            ptxt = str(d.get("text") or "").strip()
            if ptxt:
                md_sections.extend([f"## [{idx}] Plan", "", ptxt, ""])

    if not prompt:
        prompt = format_codex_user_text(thread.first_user_message or thread.preview or "").strip()
    if not final_answer:
        final_answer = last_agent_msg

    # Add stats line if there were more actions than shown
    total_actions = cmd_count + edit_count + tool_count
    if total_actions > len(summary_lines):
        summary_lines.append(f"- ... 共 {cmd_count} 次命令、{edit_count} 处文件改动、{tool_count} 次其他工具调用")

    status = "error" if (had_error and not final_answer) else "completed"

    return InlineSubagentSummary(
        thread_id=thread.id,
        role=role,
        agent_type=agent_type,
        prompt=prompt,
        final_answer=final_answer,
        status=status,
        step_summary_lines=summary_lines,
        markdown_transcript="\n".join(md_sections).rstrip() + "\n",
        asset_rel_path=asset_rel_path,
        created_at_ms=thread.created_at_ms,
        updated_at_ms=thread.updated_at_ms,
        parent_thread_id=thread.parent_thread_id or "",
        nesting_depth=thread.nesting_depth,
    )


def map_codex_thread(
    thread: CodexThread,
    history: ThreadHistoryData,
    *,
    project_id: str,
    project_name: str,
    workspaces: List[WorkspaceInfo],
    agy_paths: AgyPaths,
    codex_home: Path,
    root_conversation_id: str = "",
    subagent_roles: Optional[Dict[str, str]] = None,
    subagent_mode: str = "inline",
    separate_subagent_ids: Optional[set[str]] = None,
    inline_subagents: Optional[Dict[str, InlineSubagentSummary]] = None,
) -> MappedConversation:
    """Convert a CodexThread + ThreadHistoryData into a lossless MappedConversation."""
    cid = thread.id
    trajectory_id = deterministic_uuid("trajectory", cid)
    init_state_id = deterministic_uuid("init_state", cid)
    is_subagent = thread.is_thread_spawn
    subagent_roles = subagent_roles or {}
    inline_subagents = inline_subagents or {}
    attached_inline_subagent_uris: Dict[str, str] = {}

    brain_dir = agy_paths.brain_dir / cid
    ws_uris = [w.folder_uri for w in workspaces] if project_id != OUTSIDE_OF_PROJECT_ID else []

    steps: List[EncodedStep] = []
    assets: List[ExtractedAsset] = []
    asset_seq = 0
    plan_seq = 0
    user_msg_seq = 0
    seen_collab_ids: set[str] = set()
    consumed_rollout_img_ids: set[int] = set()

    # Pre-index usages by turn_id so we can attach token usage to PLANNER_RESPONSE steps
    usages_by_turn: Dict[str, List[RolloutUsage]] = {}
    for u in history.augmentation.usages:
        usages_by_turn.setdefault(u.turn_id, []).append(u)

    def add_asset(rel_path: str, data: bytes) -> str:
        assets.append(ExtractedAsset(rel_path=rel_path, data=data))
        return path_to_file_uri(str(brain_dir / rel_path))

    def ensure_inline_subagent_asset(child_id: str) -> Optional[str]:
        if child_id in attached_inline_subagent_uris:
            return attached_inline_subagent_uris[child_id]
        info = inline_subagents.get(child_id)
        if not info:
            return None
        uri = add_asset(info.asset_rel_path, info.markdown_transcript.encode("utf-8"))
        attached_inline_subagent_uris[child_id] = uri
        return uri

    def is_separate_subagent(child_id: str) -> bool:
        if subagent_mode != "separate":
            return False
        if separate_subagent_ids is None:
            return True
        return child_id in separate_subagent_ids

    def next_step_idx() -> int:
        return len(steps)

    def make_exec_id(turn_id: str) -> str:
        return deterministic_uuid("exec", f"{cid}:{turn_id}")

    # Pending assistant batch (for strict PLANNER_RESPONSE -> GENERIC pairing)
    pending_turn_id = "turn-0"
    pending_thinking: List[str] = []
    pending_text: List[str] = []
    pending_tools: List[MappedTool] = []
    pending_start_ms = 0
    pending_end_ms = 0
    pending_think_dur_ms = 0

    def pop_usage_for_turn(turn_id: str) -> Optional[UsageStats]:
        ulist = usages_by_turn.get(turn_id)
        if not ulist:
            return None
        u = ulist.pop(0)
        return UsageStats(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_tokens=u.cached_input_tokens,
            cache_write_tokens=u.cache_write_tokens,
            thinking_output_tokens=u.reasoning_output_tokens,
            response_output_tokens=max(0, u.output_tokens - u.reasoning_output_tokens),
            message_id=u.response_id or "",
            response_id=u.response_id,
        )

    def flush_assistant_batch() -> None:
        nonlocal pending_thinking, pending_text, pending_tools, pending_start_ms, pending_end_ms, pending_think_dur_ms
        if not pending_thinking and not pending_text and not pending_tools:
            return

        thinking_str = "\n\n".join(_dedupe_thinking_blocks(pending_thinking))
        response_str = "\n\n".join(t for t in pending_text if t.strip())
        ts_start = pending_start_ms or thread.created_at_ms
        ts_end = max(ts_start, pending_end_ms or ts_start)

        idx = next_step_idx()
        msg_id = f"bot-{deterministic_uuid('msg', f'{cid}:{idx}')}"
        usage = pop_usage_for_turn(pending_turn_id)
        if usage and not usage.message_id:
            usage.message_id = msg_id

        # Ensure unique call_ids within the same PLANNER_RESPONSE batch
        batch_call_ids: set[str] = set()
        for t_idx, tool in enumerate(pending_tools):
            base_cid = tool.call_id or f"call_{idx}_{t_idx}"
            cand_cid = base_cid
            suffix = 1
            while cand_cid in batch_call_ids:
                cand_cid = f"{base_cid}_{suffix}"
                suffix += 1
            tool.call_id = cand_cid
            batch_call_ids.add(cand_cid)

        tc_specs = [
            ToolCallSpec(id=t.call_id, name=t.tool_name, arguments_json=t.arguments_json)
            for t in pending_tools
        ]
        meta_bytes = encode_step_metadata(
            created_at_ms=ts_start,
            started_at_ms=ts_start,
            completed_at_ms=ts_end,
            source=STEP_SOURCE_MODEL,
            execution_id=make_exec_id(pending_turn_id),
            trajectory_id=trajectory_id,
            step_index=idx,
            cascade_id=cid,
            content_id=deterministic_hex32(f"step:{cid}:{idx}"),
            status=STEP_STATUS_DONE,
            usage=usage,
        )
        body_bytes = encode_planner_response(
            response=response_str,
            thinking=thinking_str,
            message_id=msg_id,
            tool_calls=tc_specs,
            thinking_duration_ms=pending_think_dur_ms,
        )
        payload_bytes = encode_cortex_step(
            step_type=STEP_TYPE_PLANNER_RESPONSE,
            status=STEP_STATUS_DONE,
            metadata_bytes=meta_bytes,
            body_field_number=20,
            body_bytes=body_bytes,
        )
        model_name = history.augmentation.turn_models.get(pending_turn_id) or thread.model or ""
        t_entry: Dict[str, Any] = {
            "step_index": idx,
            "source": "MODEL",
            "type": "PLANNER_RESPONSE",
            "status": "DONE",
            "created_at": ms_to_iso_z(ts_start),
            "model": model_name,
        }
        if usage:
            t_entry["input_tokens"] = usage.input_tokens
            t_entry["cache_read_tokens"] = usage.cache_read_tokens
            t_entry["output_tokens"] = usage.output_tokens
        if thinking_str:
            t_entry["thinking"] = thinking_str
        if response_str:
            t_entry["content"] = response_str
        if tc_specs:
            t_entry["tool_calls"] = [
                {"name": tc.name, "args": _safe_json_loads(tc.arguments_json)} for tc in tc_specs
            ]

        steps.append(
            EncodedStep(
                idx=idx,
                step_type=STEP_TYPE_PLANNER_RESPONSE,
                status=STEP_STATUS_DONE,
                created_at_ms=ts_start,
                metadata_bytes=meta_bytes,
                payload_bytes=payload_bytes,
                fts_agent_text=response_str,
                transcript_entry=t_entry,
            )
        )

        # Emit each paired GENERIC step immediately after PLANNER_RESPONSE
        for tool in pending_tools:
            g_idx = next_step_idx()
            g_status = STEP_STATUS_ERROR if tool.is_error else STEP_STATUS_DONE
            g_tc = ToolCallSpec(id=tool.call_id, name=tool.tool_name, arguments_json=tool.arguments_json)
            full_uri = ""
            large_out = None
            if tool.full_output_text is not None:
                out_path = brain_dir / ".system_generated" / "steps" / str(g_idx) / "output.txt"
                full_uri = path_to_file_uri(str(out_path))
                large_out = (g_idx, tool.full_output_text)

            g_meta = encode_step_metadata(
                created_at_ms=tool.created_at_ms or ts_end,
                started_at_ms=tool.started_at_ms or tool.created_at_ms or ts_end,
                completed_at_ms=tool.completed_at_ms or tool.created_at_ms or ts_end,
                source=STEP_SOURCE_MODEL,
                execution_id=make_exec_id(pending_turn_id),
                trajectory_id=trajectory_id,
                step_index=g_idx,
                cascade_id=cid,
                content_id=deterministic_hex32(f"step:{cid}:{g_idx}"),
                status=g_status,
                tool_call=g_tc,
                tool_source=1,
            )
            g_body = encode_generic_step(
                args=tool.args_dict,
                result_text=tool.result_text,
                render_info=tool.render_info,
                payload_any=tool.payload_any,
                full_output_uri=full_uri,
                media=tool.media or None,
            )
            comp_inter = (
                encode_completed_ask_question_interaction(
                    tool.ask_question_entries,
                    trajectory_id=trajectory_id,
                    step_index=g_idx,
                )
                if tool.ask_question_entries
                else b""
            )
            g_payload = encode_cortex_step(
                step_type=STEP_TYPE_GENERIC,
                status=g_status,
                metadata_bytes=g_meta,
                body_field_number=140,
                body_bytes=g_body,
                completed_interaction_bytes=comp_inter,
            )
            err_bytes = encode_error_details(tool.error_message, tool.result_text) if tool.is_error else None
            g_entry: Dict[str, Any] = {
                "step_index": g_idx,
                "source": "MODEL",
                "type": "GENERIC",
                "status": "ERROR" if tool.is_error else "DONE",
                "created_at": ms_to_iso_z(tool.created_at_ms or ts_end),
                "content": tool.result_text,
            }
            steps.append(
                EncodedStep(
                    idx=g_idx,
                    step_type=STEP_TYPE_GENERIC,
                    status=g_status,
                    created_at_ms=tool.created_at_ms or ts_end,
                    metadata_bytes=g_meta,
                    payload_bytes=g_payload,
                    error_details_bytes=err_bytes,
                    transcript_entry=g_entry,
                    large_output_file=large_out,
                )
            )

        pending_thinking = []
        pending_text = []
        pending_tools = []
        pending_start_ms = 0
        pending_end_ms = 0
        pending_think_dur_ms = 0

    def emit_system_message(
        text: str,
        *,
        ts_ms: int,
        turn_id: str,
        event_type: str = "",
        sender: str = "",
        media: List[MediaAttachment] | None = None,
    ) -> None:
        idx = next_step_idx()
        meta_bytes = encode_step_metadata(
            created_at_ms=ts_ms,
            started_at_ms=ts_ms,
            completed_at_ms=ts_ms,
            source=STEP_SOURCE_SYSTEM,
            execution_id=make_exec_id(turn_id),
            trajectory_id=trajectory_id,
            step_index=idx,
            cascade_id=cid,
            content_id=deterministic_hex32(f"step:{cid}:{idx}"),
            status=STEP_STATUS_DONE,
        )
        agent_msg_pb = b""
        display_msg = text
        if sender:
            display_msg = (
                f"[Message] timestamp={ms_to_iso_z(ts_ms)} sender={sender} "
                f"priority=MESSAGE_PRIORITY_HIGH content={text}"
            )
            agent_msg_pb = encode_agent_message(
                msg_id=deterministic_uuid("amsg", f"{cid}:{idx}"),
                sender=sender,
                recipient=cid,
                content=text,
                timestamp_ms=ts_ms,
            )
        body_bytes = encode_system_message(
            display_msg,
            event_type=event_type,
            agent_message=agent_msg_pb,
            media=media,
        )
        payload_bytes = encode_cortex_step(
            step_type=STEP_TYPE_SYSTEM_MESSAGE,
            status=STEP_STATUS_DONE,
            metadata_bytes=meta_bytes,
            body_field_number=114,
            body_bytes=body_bytes,
        )
        t_entry: Dict[str, Any] = {
            "step_index": idx,
            "source": "SYSTEM",
            "type": "SYSTEM_MESSAGE",
            "status": "DONE",
            "created_at": ms_to_iso_z(ts_ms),
            "content": display_msg,
        }
        if media:
            t_entry["media"] = [{"mime_type": m.mime_type, "uri": m.uri} for m in media]
        steps.append(
            EncodedStep(
                idx=idx,
                step_type=STEP_TYPE_SYSTEM_MESSAGE,
                status=STEP_STATUS_DONE,
                created_at_ms=ts_ms,
                metadata_bytes=meta_bytes,
                payload_bytes=payload_bytes,
                transcript_entry=t_entry,
            )
        )

    # Pre-collect all explicit localImage paths in history.items so fallback never steals them
    referenced_img_paths: set[str] = set()
    for it in history.items:
        if it.item_type == "userMessage":
            for p in it.data.get("content") or []:
                if isinstance(p, dict) and p.get("type") == "localImage" and p.get("path"):
                    referenced_img_paths.add(str(p["path"]))

    def extract_user_media(item: CodexItem, u_idx: int) -> List[MediaAttachment]:
        nonlocal asset_seq
        media_list: List[MediaAttachment] = []
        turn_imgs = history.augmentation.images_by_turn_id.get(item.turn_id, [])
        content = item.data.get("content") or []

        def save_img(mime: str, raw_bytes: bytes, hint_name: str) -> MediaAttachment:
            nonlocal asset_seq
            asset_seq += 1
            ext = MIME_TO_EXT.get(mime.lower(), ".png")
            raw_base = os.path.basename(hint_name or "")
            if not raw_base or raw_base.startswith("codex-clipboard-"):
                short_h = hashlib.sha256(raw_bytes).hexdigest()[:8]
                base = f"media_{asset_seq:03d}_{short_h}{ext}"
                rel = f".user_uploaded/{base}"
            else:
                base = sanitize_filename(raw_base, f"image_{asset_seq:03d}")
                if not base.lower().endswith(ext):
                    base = f"{Path(base).stem}{ext}"
                rel = f".user_uploaded/{asset_seq:03d}_{base}"
            uri = add_asset(rel, raw_bytes)
            return MediaAttachment(
                mime_type=mime,
                uri=uri,
                display_name=base,
                media_id=deterministic_uuid("media", f"{cid}:{asset_seq}"),
            )

        has_explicit_image_parts = False
        for part in content:
            if not isinstance(part, dict):
                continue
            pt = part.get("type")
            if pt == "localImage":
                has_explicit_image_parts = True
                img_path = str(part.get("path") or "")
                # 1. Prefer original full-resolution file on disk if it still exists
                #    (Codex downscales inline base64 images in rollouts, whereas files on disk
                #    such as ~/.codex/attachments/ or local folders keep full Retina resolution!)
                if img_path and os.path.isfile(img_path):
                    try:
                        raw_bytes = Path(img_path).read_bytes()
                        if raw_bytes:
                            mime = mimetypes.guess_type(img_path)[0] or "image/png"
                            # Mark any corresponding rollout copy as consumed so fallback never duplicates it
                            for cand in turn_imgs:
                                if cand.path_hint == img_path:
                                    consumed_rollout_img_ids.add(id(cand))
                            r_hit = history.augmentation.images_by_path.get(img_path)
                            if r_hit is not None:
                                consumed_rollout_img_ids.add(id(r_hit))
                            media_list.append(save_img(mime, raw_bytes, img_path))
                            continue
                    except Exception:
                        pass
                # 2. Fallback to rollout base64 matching path_hint within THIS turn (recovers expired /var/folders temp images!)
                matched_turn_path = None
                if img_path:
                    for cand in turn_imgs:
                        if id(cand) not in consumed_rollout_img_ids and cand.path_hint == img_path:
                            consumed_rollout_img_ids.add(id(cand))
                            matched_turn_path = cand
                            break
                if matched_turn_path is not None:
                    media_list.append(save_img(matched_turn_path.mime_type, matched_turn_path.data, img_path))
                    continue
                # 3. Try global exact path match from rollout base64
                rimg: Optional[RolloutImage] = history.augmentation.images_by_path.get(img_path)
                if rimg is not None:
                    consumed_rollout_img_ids.add(id(rimg))
                    media_list.append(save_img(rimg.mime_type, rimg.data, img_path))
                    continue
                # 4. Try sequential unconsumed match from this turn's rollout images
                matched_seq = None
                for cand in turn_imgs:
                    if id(cand) not in consumed_rollout_img_ids:
                        consumed_rollout_img_ids.add(id(cand))
                        matched_seq = cand
                        break
                if matched_seq is not None:
                    media_list.append(save_img(matched_seq.mime_type, matched_seq.data, img_path or matched_seq.path_hint))
                    continue
            elif pt == "image":
                has_explicit_image_parts = True
                url = str(part.get("url") or part.get("image_url") or "")
                dm = DATA_URI_RE.match(url)
                if dm:
                    try:
                        raw_bytes = base64.b64decode(dm.group(2).strip())
                        if raw_bytes:
                            media_list.append(save_img(dm.group(1).strip(), raw_bytes, "inline_image"))
                    except Exception:
                        pass

        # Pure-rollout fallback (or older formats without localImage parts):
        # ONLY attach unconsumed images belonging to THIS exact turn_id whose path_hint
        # is not already claimed by an explicit localImage part elsewhere.
        if not has_explicit_image_parts and not media_list and turn_imgs:
            for cand in turn_imgs:
                if id(cand) not in consumed_rollout_img_ids and cand.path_hint not in referenced_img_paths:
                    consumed_rollout_img_ids.add(id(cand))
                    media_list.append(save_img(cand.mime_type, cand.data, cand.path_hint or "image"))

        return media_list

    # First pass: collect collabAgentToolCall ids so subAgentActivity doesn't duplicate them
    for it in history.items:
        if it.item_type == "collabAgentToolCall" and it.item_id:
            seen_collab_ids.add(it.item_id)

    last_user_input_ms = 0
    last_user_input_step_index = -1
    first_subagent_prompt = ""

    # Main item iteration (normalize late-flushed trailing reasoning before final_answer)
    for item in _normalize_codex_items(history.items):
        itype = item.item_type
        d = item.data
        ts_c = item.created_at_ms or thread.created_at_ms
        ts_s = item.started_at_ms or ts_c
        ts_e = max(ts_s, item.completed_at_ms or ts_c)

        if itype == "userMessage":
            flush_assistant_batch()
            content_parts = d.get("content") or []
            texts: List[str] = []
            attached_img_paths: set[str] = set()
            for p in content_parts:
                if not isinstance(p, dict):
                    continue
                pt = p.get("type")
                if pt == "text" and p.get("text"):
                    texts.append(str(p["text"]))
                elif pt == "localImage" and p.get("path"):
                    attached_img_paths.add(str(p["path"]))
                elif pt == "skill":
                    sname = p.get("name") or p.get("path") or "skill"
                    texts.append(f"[Skill: {sname}]")
            raw_user_text = "\n".join(texts).strip()
            media = extract_user_media(item, user_msg_seq)
            user_msg_seq += 1

            # Native Antigravity ask_question UI card for Codex <send_user_message_question_reply>
            if not is_subagent and "<send_user_message_question_reply>" in raw_user_text:
                q_entries, q_call_id, rem_text = parse_codex_question_replies(
                    raw_user_text,
                    history.augmentation.question_specs_by_call_id,
                )
                if q_entries:
                    q_payload_json = [
                        {
                            "is_multi_select": e.is_multi_select,
                            "options": [txt for _oid, txt in e.options],
                            "question": e.question,
                        }
                        for e in q_entries
                    ]
                    q_args_dict = {"questions": json.dumps(q_payload_json, ensure_ascii=False)}
                    q_args_json = json.dumps({"questions": q_payload_json}, ensure_ascii=False)
                    ans_lines = []
                    for qi, e in enumerate(q_entries, start=1):
                        if e.selected_option_ids:
                            opt_map = dict(e.options)
                            chosen = ", ".join(opt_map.get(oid, oid) for oid in e.selected_option_ids)
                            ans_lines.append(f"A{qi}: {chosen}")
                        elif e.write_in_response:
                            ans_lines.append(f"A{qi}: {e.write_in_response}")
                        else:
                            ans_lines.append(f"A{qi}: (skipped)")
                    q_res_text = (
                        f"Created At: {ms_to_local_iso(ts_s)}\n"
                        f"Completed At: {ms_to_local_iso(ts_e)}\n"
                        + "\n".join(ans_lines)
                    )
                    pending_turn_id = item.turn_id
                    pending_start_ms = ts_s
                    pending_end_ms = ts_e
                    pending_tools.append(
                        MappedTool(
                            call_id=q_call_id or item.item_id or f"askq_{item.rollout_ordinal}",
                            tool_name="ask_question",
                            args_dict=q_args_dict,
                            arguments_json=q_args_json,
                            result_text=q_res_text,
                            full_output_text=None,
                            render_info=b"",
                            payload_any=encode_tool_ask_question(q_entries),
                            created_at_ms=ts_c,
                            started_at_ms=ts_s,
                            completed_at_ms=ts_e,
                            ask_question_entries=q_entries,
                        )
                    )
                    flush_assistant_batch()
                    raw_user_text = rem_text
                    if not raw_user_text and not media:
                        continue

            file_refs: List[FileScopeRef] = []

            def attach_codex_attachment(src_path: Path, label: str) -> bool:
                nonlocal asset_seq
                try:
                    raw_b = src_path.read_bytes()
                except Exception:
                    return False
                if not raw_b:
                    return False
                asset_seq += 1
                mime = mimetypes.guess_type(str(src_path))[0] or "text/plain"
                raw_name = src_path.name
                if raw_name.startswith("codex-clipboard-"):
                    ext = src_path.suffix or MIME_TO_EXT.get(mime.lower(), ".png")
                    short_h = hashlib.sha256(raw_b).hexdigest()[:8]
                    base = f"media_{asset_seq:03d}_{short_h}{ext}"
                    rel = f".user_uploaded/{base}"
                else:
                    base = sanitize_filename(raw_name, f"attachment_{asset_seq:03d}.txt")
                    rel = f".user_uploaded/{asset_seq:03d}_{base}"
                uri = add_asset(rel, raw_b)
                disp = (label[:60].strip() if label and raw_name == "pasted-text.txt" else raw_name) or base
                media.append(
                    MediaAttachment(
                        mime_type=mime,
                        uri=uri,
                        display_name=disp,
                        description=disp,
                        media_id=deterministic_uuid("media", f"{cid}:{asset_seq}"),
                    )
                )
                return True

            user_text = format_codex_user_text(
                raw_user_text,
                attached_image_paths=attached_img_paths,
                codex_home=codex_home,
                attach_file_media_fn=attach_codex_attachment,
                out_file_refs=file_refs,
            )

            # For subagent step 0, emit SYSTEM_MESSAGE matching native Antigravity subagents
            if is_subagent and len(steps) == 0:
                first_subagent_prompt = user_text
                emit_system_message(
                    user_text,
                    ts_ms=ts_c,
                    turn_id=item.turn_id,
                    sender=thread.parent_thread_id or root_conversation_id or "parent",
                    media=media or None,
                )
                continue

            idx = next_step_idx()
            meta_bytes = encode_step_metadata(
                created_at_ms=ts_c,
                started_at_ms=ts_c,
                completed_at_ms=ts_e,
                source=STEP_SOURCE_USER_EXPLICIT,
                execution_id=make_exec_id(item.turn_id),
                trajectory_id=trajectory_id,
                step_index=idx,
                cascade_id=cid,
                content_id=deterministic_hex32(f"step:{cid}:{idx}"),
                status=STEP_STATUS_DONE,
            )
            body_bytes = encode_user_input(
                user_text,
                media=media or None,
                file_refs=file_refs or None,
                client_message_id=str(d.get("clientId") or deterministic_uuid("cmsg", f"{cid}:{idx}")),
            )
            payload_bytes = encode_cortex_step(
                step_type=STEP_TYPE_USER_INPUT,
                status=STEP_STATUS_DONE,
                metadata_bytes=meta_bytes,
                body_field_number=19,
                body_bytes=body_bytes,
            )
            t_entry: Dict[str, Any] = {
                "step_index": idx,
                "source": "USER_EXPLICIT",
                "type": "USER_INPUT",
                "status": "DONE",
                "created_at": ms_to_iso_z(ts_c),
                "content": f"<USER_REQUEST>\n{user_text}\n</USER_REQUEST>",
            }
            if media:
                t_entry["media"] = [{"mime_type": m.mime_type, "uri": m.uri} for m in media]
            steps.append(
                EncodedStep(
                    idx=idx,
                    step_type=STEP_TYPE_USER_INPUT,
                    status=STEP_STATUS_DONE,
                    created_at_ms=ts_c,
                    metadata_bytes=meta_bytes,
                    payload_bytes=payload_bytes,
                    fts_user_text=user_text,
                    transcript_entry=t_entry,
                )
            )
            last_user_input_ms = ts_c
            last_user_input_step_index = idx

        elif itype == "reasoning":
            if pending_tools or pending_text:
                flush_assistant_batch()
            parts = clean_reasoning_parts(d)
            if parts:
                pending_turn_id = item.turn_id
                if not pending_start_ms:
                    pending_start_ms = ts_s
                pending_end_ms = max(pending_end_ms, ts_e)
                if ts_e > ts_s:
                    pending_think_dur_ms += ts_e - ts_s
                pending_thinking.append("\n\n".join(parts))

        elif itype == "agentMessage":
            if pending_tools:
                flush_assistant_batch()
            text = format_codex_assistant_text(str(d.get("text") or ""))
            if text:
                pending_turn_id = item.turn_id
                if not pending_start_ms:
                    pending_start_ms = ts_s
                pending_end_ms = max(pending_end_ms, ts_e)
                pending_text.append(text)
                if d.get("phase") == "final_answer":
                    flush_assistant_batch()

        elif itype == "plan":
            flush_assistant_batch()
            plan_text = format_codex_assistant_text(str(d.get("text") or ""))
            if plan_text:
                plan_seq += 1
                plan_rel = "implementation_plan.md" if plan_seq == 1 else f"implementation_plan_{plan_seq}.md"
                add_asset(plan_rel, plan_text.encode("utf-8"))
                meta_json = {
                    "artifactType": "ARTIFACT_TYPE_IMPLEMENTATION_PLAN",
                    "summary": clean_codex_title(plan_text.splitlines()[0] if plan_text else "Implementation Plan"),
                    "updatedAt": ms_to_iso_z(ts_e),
                }
                add_asset(f"{plan_rel}.metadata.json", (json.dumps(meta_json, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
                pending_turn_id = item.turn_id
                pending_start_ms = ts_s
                pending_end_ms = ts_e
                pending_text.append(plan_text)
                flush_assistant_batch()

        elif itype == "contextCompaction":
            flush_assistant_batch()
            emit_system_message(
                "Context compacted.",
                ts_ms=ts_c,
                turn_id=item.turn_id,
                event_type="context_compaction",
            )

        elif itype == "functionCallOutput":
            flush_assistant_batch()
            fname = str(d.get("name") or "function_call_output")
            ns = str(d.get("namespace") or "")
            out_str = str(d.get("output") or "")
            label = f"{ns}/{fname}" if ns else fname
            emit_system_message(
                f"[{label}]\n{out_str}",
                ts_ms=ts_c,
                turn_id=item.turn_id,
                event_type=label,
            )

        elif itype == "commandExecution":
            pending_turn_id = item.turn_id
            if not pending_start_ms:
                pending_start_ms = ts_s
            pending_end_ms = max(pending_end_ms, ts_e)
            pending_tools.append(_map_command_execution(item, ts_s, ts_e))

        elif itype == "fileChange":
            pending_turn_id = item.turn_id
            if not pending_start_ms:
                pending_start_ms = ts_s
            pending_end_ms = max(pending_end_ms, ts_e)
            pending_tools.extend(_map_file_change(item, ts_s, ts_e))

        elif itype == "mcpToolCall":
            pending_turn_id = item.turn_id
            if not pending_start_ms:
                pending_start_ms = ts_s
            pending_end_ms = max(pending_end_ms, ts_e)
            pending_tools.append(_map_mcp_tool_call(item, ts_s, ts_e))

        elif itype == "webSearch":
            pending_turn_id = item.turn_id
            if not pending_start_ms:
                pending_start_ms = ts_s
            pending_end_ms = max(pending_end_ms, ts_e)
            pending_tools.append(_map_web_search(item, ts_s, ts_e))

        elif itype == "imageView":
            pending_turn_id = item.turn_id
            if not pending_start_ms:
                pending_start_ms = ts_s
            pending_end_ms = max(pending_end_ms, ts_e)
            pending_tools.append(_map_image_view(item, ts_s, ts_e))

        elif itype == "imageGeneration":
            pending_turn_id = item.turn_id
            if not pending_start_ms:
                pending_start_ms = ts_s
            pending_end_ms = max(pending_end_ms, ts_e)
            pending_tools.append(
                _map_image_generation(item, ts_s, ts_e, cid, codex_home, add_asset)
            )

        elif itype == "dynamicToolCall":
            pending_turn_id = item.turn_id
            if not pending_start_ms:
                pending_start_ms = ts_s
            pending_end_ms = max(pending_end_ms, ts_e)
            pending_tools.append(_map_dynamic_tool_call(item, ts_s, ts_e))

        elif itype == "collabAgentToolCall":
            pending_turn_id = item.turn_id
            if not pending_start_ms:
                pending_start_ms = ts_s
            pending_end_ms = max(pending_end_ms, ts_e)
            pending_tools.append(
                _map_collab_agent_tool(
                    item,
                    ts_s,
                    ts_e,
                    agy_paths,
                    ws_uris,
                    subagent_roles,
                    is_separate_fn=is_separate_subagent,
                    inline_subagents=inline_subagents,
                    ensure_inline_asset_fn=ensure_inline_subagent_asset,
                )
            )

        elif itype == "subAgentActivity":
            # Skip duplicates already represented by collabAgentToolCall or empty "interacted" heartbeats
            if item.item_id in seen_collab_ids:
                continue
            kind = str(d.get("kind") or "")
            if kind == "interacted":
                continue
            child_tid = str(d.get("agentThreadId") or "")
            apath = str(d.get("agentPath") or "")
            if not child_tid:
                continue
            pending_turn_id = item.turn_id
            if not pending_start_ms:
                pending_start_ms = ts_s
            pending_end_ms = max(pending_end_ms, ts_e)
            pending_tools.append(
                _map_subagent_activity(
                    item,
                    ts_s,
                    ts_e,
                    agy_paths,
                    ws_uris,
                    subagent_roles,
                    is_separate_fn=is_separate_subagent,
                    inline_subagents=inline_subagents,
                    ensure_inline_asset_fn=ensure_inline_subagent_asset,
                )
            )

        elif itype == "sleep":
            pending_turn_id = item.turn_id
            if not pending_start_ms:
                pending_start_ms = ts_s
            pending_end_ms = max(pending_end_ms, ts_e)
            dur = int(d.get("durationMs") or 0)
            args = {"DurationSeconds": str(max(1, dur // 1000))}
            res_text = f"Created At: {ms_to_local_iso(ts_s)}\nCompleted At: {ms_to_local_iso(ts_e)}\nSlept {dur}ms."
            render = encode_step_render_info(
                title=f"Waited {dur / 1000:.1f}s",
                icon="schedule",
                aggregation_key="actions",
                hidden=True,
            )
            pending_tools.append(
                MappedTool(
                    call_id=item.item_id or f"sleep_{item.rollout_ordinal}",
                    tool_name="schedule",
                    args_dict=args,
                    arguments_json=json.dumps(args, ensure_ascii=False),
                    result_text=res_text,
                    full_output_text=None,
                    render_info=render,
                    payload_any=b"",
                    created_at_ms=ts_c,
                    started_at_ms=ts_s,
                    completed_at_ms=ts_e,
                )
            )

        else:
            # Unknown future Codex item type -> preserve losslessly as JSON generic tool step
            pending_turn_id = item.turn_id
            if not pending_start_ms:
                pending_start_ms = ts_s
            pending_end_ms = max(pending_end_ms, ts_e)
            raw_json = json.dumps(d, ensure_ascii=False, indent=2)
            inline_res, full_res = _split_large_output(raw_json)
            render = encode_step_render_info(
                title_prefix="Ran",
                icon="extension",
                aggregation_key="actions",
                title_parts=[encode_title_part_text(itype)],
                content_blocks=[encode_content_block_markdown(f"```json\n{inline_res[:4000]}\n```")],
            )
            pending_tools.append(
                MappedTool(
                    call_id=item.item_id or f"tool_{item.rollout_ordinal}",
                    tool_name=itype,
                    args_dict={"type": itype},
                    arguments_json=json.dumps({"type": itype}, ensure_ascii=False),
                    result_text=inline_res,
                    full_output_text=full_res,
                    render_info=render,
                    payload_any=b"",
                    created_at_ms=ts_c,
                    started_at_ms=ts_s,
                    completed_at_ms=ts_e,
                )
            )

    flush_assistant_batch()

    # Persist any remaining inline subagent transcripts (e.g. depth-2 nested subagents) into brain/<cid>/subagents/
    if not is_subagent and inline_subagents:
        for child_id in inline_subagents:
            ensure_inline_subagent_asset(child_id)

    # Timestamps
    created_ms = thread.created_at_ms or (steps[0].created_at_ms if steps else 0)
    last_mod_ms = max(thread.updated_at_ms, steps[-1].created_at_ms if steps else 0, created_ms)
    if last_user_input_ms == 0 and not is_subagent:
        last_user_input_ms = created_ms

    title = pick_best_title(thread)
    preview = "" if is_subagent else (clean_codex_title(thread.preview) or title)

    subagent_role = ""
    agent_name = ""
    if is_subagent:
        leaf = os.path.basename(thread.agent_path.rstrip("/")) if thread.agent_path else ""
        if leaf and leaf != "root":
            subagent_role = leaf
        elif thread.agent_nickname:
            subagent_role = thread.agent_nickname
        else:
            subagent_role = "Subagent"
        agent_name = thread.agent_role or "self"

    return MappedConversation(
        conversation_id=cid,
        trajectory_id=trajectory_id,
        init_state_id=init_state_id,
        title=title,
        preview=preview,
        project_id=project_id,
        project_name=project_name,
        workspaces=workspaces,
        created_at_ms=created_ms,
        last_modified_ms=last_mod_ms,
        last_user_input_ms=last_user_input_ms,
        last_user_input_step_index=last_user_input_step_index,
        steps=steps,
        assets=assets,
        archived=thread.archived,
        pinned=thread.is_pinned,
        parent_conversation_id=thread.parent_thread_id or "" if is_subagent else "",
        root_conversation_id=root_conversation_id or (thread.parent_thread_id or cid if is_subagent else cid),
        nesting_depth=thread.nesting_depth if is_subagent else 0,
        agent_name=agent_name,
        subagent_role=subagent_role,
        subagent_prompt=first_subagent_prompt,
    )


# --- Per-Tool Mappers ---


def _safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return s


def _split_large_output(text: str, limit: int = INLINE_OUTPUT_LIMIT) -> Tuple[str, Optional[str]]:
    if len(text.encode("utf-8")) <= limit:
        return text, None
    clipped = text[: limit // 2] + "\n... [full output saved to step output.txt]"
    return clipped, text


def _map_command_execution(item: CodexItem, ts_s: int, ts_e: int) -> MappedTool:
    d = item.data
    cmd = str(d.get("command") or "")
    cwd = str(d.get("cwd") or "")
    raw_exit = d.get("exitCode")
    exit_code = int(raw_exit) if raw_exit is not None else 0
    status = str(d.get("status") or "completed")
    agg_out = str(d.get("aggregatedOutput") or "")
    summary = short_command_summary(cmd)

    header = (
        f"Created At: {ms_to_local_iso(ts_s)}\n"
        f"Completed At: {ms_to_local_iso(ts_e)}\n"
        f"Command exited with code {exit_code}.\n\nCommand output:\n"
    )
    full_text = header + agg_out
    inline_res, spilled_full = _split_large_output(full_text)
    term_preview = agg_out if len(agg_out) <= TERMINAL_PREVIEW_LIMIT else (agg_out[:TERMINAL_PREVIEW_LIMIT] + "\n... [truncated]")

    args = {
        "CommandLine": cmd,
        "toolSummary": summary,
    }
    if cwd:
        args["Cwd"] = cwd

    render = encode_step_render_info(
        title_prefix="Ran",
        icon="terminal",
        aggregation_key="terminal",
        title_parts=[encode_title_part_text(summary)],
        content_blocks=[
            encode_content_block_terminal(
                command=cmd,
                cwd=cwd,
                exit_code=exit_code,
                output=term_preview,
            )
        ],
    )
    payload = encode_tool_run_command(
        command_line=cmd,
        cwd=cwd,
        exit_code=exit_code,
        shell_name="zsh",
    )
    is_err = status in ("failed", "declined", "error")
    return MappedTool(
        call_id=item.item_id or f"cmd_{item.rollout_ordinal}",
        tool_name="run_command",
        args_dict=args,
        arguments_json=json.dumps(args, ensure_ascii=False),
        result_text=inline_res,
        full_output_text=spilled_full,
        render_info=render,
        payload_any=payload,
        created_at_ms=item.created_at_ms or ts_s,
        started_at_ms=ts_s,
        completed_at_ms=ts_e,
        is_error=is_err,
        error_message=f"Command failed (exit code {exit_code})" if is_err else "",
    )


def _map_file_change(item: CodexItem, ts_s: int, ts_e: int) -> List[MappedTool]:
    d = item.data
    changes = d.get("changes") or []
    status = str(d.get("status") or "completed")
    is_err = status in ("failed", "declined", "error")
    out: List[MappedTool] = []

    if not changes:
        args = {"Changes": "[]"}
        res = f"Created At: {ms_to_local_iso(ts_s)}\nCompleted At: {ms_to_local_iso(ts_e)}\nNo file changes."
        return [
            MappedTool(
                call_id=item.item_id or f"fc_{item.rollout_ordinal}",
                tool_name="edit_file",
                args_dict=args,
                arguments_json=json.dumps(args),
                result_text=res,
                full_output_text=None,
                render_info=encode_step_render_info(title_prefix="Edited", icon="edit", aggregation_key="edits"),
                payload_any=b"",
                created_at_ms=item.created_at_ms or ts_s,
                started_at_ms=ts_s,
                completed_at_ms=ts_e,
                is_error=is_err,
                error_message="File change failed" if is_err else "",
            )
        ]

    multi = len(changes) > 1
    for i, ch in enumerate(changes):
        if not isinstance(ch, dict):
            continue
        path = str(ch.get("path") or "")
        kind_obj = ch.get("kind") or {}
        kind_type = str(kind_obj.get("type") if isinstance(kind_obj, dict) else kind_obj or "update")
        move_path = kind_obj.get("move_path") if isinstance(kind_obj, dict) else None
        target_path = str(move_path) if move_path else path
        diff_text = str(ch.get("diff") or "")

        diff_lines, additions, deletions = parse_unified_diff(diff_text, kind_type)
        proto_lines = diff_lines[:MAX_DIFF_LINES_IN_PROTO]

        if kind_type == "add":
            tool_name = "write_file"
            fe_kind = FILE_EDIT_KIND_CREATE
            prefix = "Created"
        elif kind_type == "delete":
            tool_name = "edit_file"
            fe_kind = FILE_EDIT_KIND_EDIT
            prefix = "Deleted"
        else:
            tool_name = "edit_file"
            fe_kind = FILE_EDIT_KIND_EDIT
            prefix = "Edited"

        file_uri = path_to_file_uri(target_path) if target_path else ""
        title_parts = []
        if file_uri:
            title_parts.append(encode_title_part_file(file_uri))
            if additions or deletions:
                title_parts.append(encode_title_part_file_diff(file_uri, additions, deletions))
        elif target_path:
            title_parts.append(encode_title_part_text(target_path))

        render = encode_step_render_info(
            title_prefix=prefix,
            icon="edit",
            aggregation_key="edits",
            title_parts=title_parts,
        )
        payload = encode_tool_file_edit(
            target_path,
            kind=fe_kind,
            diff_lines=proto_lines,
            additions=additions,
            deletions=deletions,
            start_line=1,
            end_line=len(proto_lines),
        )

        move_note = f" (moved from {path})" if move_path and move_path != path else ""
        summary_line = f"{prefix} {target_path}{move_note} (+{additions}/-{deletions} lines)."
        full_res_text = (
            f"Created At: {ms_to_local_iso(ts_s)}\n"
            f"Completed At: {ms_to_local_iso(ts_e)}\n"
            f"{summary_line}\n\n{diff_text}"
        )
        inline_res, spilled = _split_large_output(full_res_text)
        args = {"Path": target_path}
        if kind_type == "add":
            args["Content"] = diff_text[:INLINE_OUTPUT_LIMIT]
        else:
            args["Diff"] = diff_text[:INLINE_OUTPUT_LIMIT]

        call_id = f"{item.item_id}_{i}" if (multi and item.item_id) else (item.item_id or f"fc_{item.rollout_ordinal}_{i}")
        out.append(
            MappedTool(
                call_id=call_id,
                tool_name=tool_name,
                args_dict=args,
                arguments_json=json.dumps({"Path": target_path}, ensure_ascii=False),
                result_text=inline_res,
                full_output_text=spilled,
                render_info=render,
                payload_any=payload,
                created_at_ms=item.created_at_ms or ts_s,
                started_at_ms=ts_s,
                completed_at_ms=ts_e,
                is_error=is_err,
                error_message=f"File edit failed: {target_path}" if is_err else "",
            )
        )
    return out


def _map_mcp_tool_call(item: CodexItem, ts_s: int, ts_e: int) -> MappedTool:
    d = item.data
    server = str(d.get("server") or "mcp")
    tool = str(d.get("tool") or "tool")
    arguments = d.get("arguments")
    args_json_inner = json.dumps(arguments, ensure_ascii=False) if not isinstance(arguments, str) else arguments
    status = str(d.get("status") or "completed")
    is_err = status in ("failed", "error") or bool(d.get("error"))

    if d.get("error"):
        err_obj = d["error"]
        result_body = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)
    else:
        res_obj = d.get("result")
        if isinstance(res_obj, dict) and isinstance(res_obj.get("content"), list):
            parts = []
            for p in res_obj["content"]:
                if isinstance(p, dict) and p.get("text"):
                    parts.append(str(p["text"]))
                else:
                    parts.append(json.dumps(p, ensure_ascii=False))
            result_body = "\n".join(parts)
        elif res_obj is not None:
            result_body = json.dumps(res_obj, ensure_ascii=False) if not isinstance(res_obj, str) else res_obj
        else:
            result_body = ""

    full_text = f"Created At: {ms_to_local_iso(ts_s)}\nCompleted At: {ms_to_local_iso(ts_e)}\n{result_body}"
    inline_res, spilled = _split_large_output(full_text)

    args = {
        "ServerName": server,
        "ToolName": tool,
        "Arguments": args_json_inner,
    }
    call_id = item.item_id or f"mcp_{item.rollout_ordinal}"
    payload = encode_tool_mcp(
        server_name=server,
        tool_call=ToolCallSpec(id=call_id, name=tool, arguments_json=args_json_inner),
        result_string=inline_res[:16_000],
    )
    render = encode_step_render_info(
        title_prefix="Called",
        icon="extension",
        aggregation_key="mcp",
        title_parts=[encode_title_part_text(f"{server}/{tool}")],
    )
    return MappedTool(
        call_id=call_id,
        tool_name="call_mcp_tool",
        args_dict=args,
        arguments_json=json.dumps(args, ensure_ascii=False),
        result_text=inline_res,
        full_output_text=spilled,
        render_info=render,
        payload_any=payload,
        created_at_ms=item.created_at_ms or ts_s,
        started_at_ms=ts_s,
        completed_at_ms=ts_e,
        is_error=is_err,
        error_message=result_body[:500] if is_err else "",
    )


def _map_web_search(item: CodexItem, ts_s: int, ts_e: int) -> MappedTool:
    d = item.data
    query = str(d.get("query") or "")
    action = d.get("action") or {}
    if not query and isinstance(action, dict):
        queries = action.get("queries") or []
        query = ", ".join(str(q) for q in queries) if queries else str(action.get("query") or "")
    results_raw = d.get("results") or []
    kb_results: List[Tuple[str, str, str]] = []
    text_lines: List[str] = []
    for r in results_raw:
        if not isinstance(r, dict):
            continue
        url = str(r.get("url") or r.get("domain") or "")
        title = str(r.get("title") or url)
        snippet = str(r.get("snippet") or r.get("text") or "")
        kb_results.append((url, title, snippet))
        text_lines.append(f"- {title} ({url}): {snippet}")

    summary_text = "\n".join(text_lines) if text_lines else f"Web search: {query}"
    full_text = f"Created At: {ms_to_local_iso(ts_s)}\nCompleted At: {ms_to_local_iso(ts_e)}\n{summary_text}"
    inline_res, spilled = _split_large_output(full_text)

    args = {"query": query}
    render = encode_step_render_info(
        title_prefix="Searched",
        icon="search",
        aggregation_key="web",
        title_parts=[encode_title_part_text(f'"{query}"' if query else "web")],
    )
    payload = encode_tool_search_web(query=query, results=kb_results, summary=inline_res[:8000])
    return MappedTool(
        call_id=item.item_id or f"web_{item.rollout_ordinal}",
        tool_name="search_web",
        args_dict=args,
        arguments_json=json.dumps(args, ensure_ascii=False),
        result_text=inline_res,
        full_output_text=spilled,
        render_info=render,
        payload_any=payload,
        created_at_ms=item.created_at_ms or ts_s,
        started_at_ms=ts_s,
        completed_at_ms=ts_e,
    )


def _map_image_view(item: CodexItem, ts_s: int, ts_e: int) -> MappedTool:
    d = item.data
    path = str(d.get("path") or "")
    file_uri = path_to_file_uri(path) if path else ""
    args = {"Path": path}
    res = f"Created At: {ms_to_local_iso(ts_s)}\nCompleted At: {ms_to_local_iso(ts_e)}\nViewed image: {path}"
    blocks = []
    if path and os.path.isfile(path):
        mime = mimetypes.guess_type(path)[0] or "image/png"
        blocks.append(encode_content_block_media(file_uri, mime, os.path.basename(path)))
    render = encode_step_render_info(
        title_prefix="Viewed",
        icon="image",
        aggregation_key="files",
        title_parts=[encode_title_part_file(file_uri)] if file_uri else [encode_title_part_text(path)],
        content_blocks=blocks or None,
    )
    payload = encode_tool_read_file(path)
    return MappedTool(
        call_id=item.item_id or f"imgv_{item.rollout_ordinal}",
        tool_name="read_file",
        args_dict=args,
        arguments_json=json.dumps(args, ensure_ascii=False),
        result_text=res,
        full_output_text=None,
        render_info=render,
        payload_any=payload,
        created_at_ms=item.created_at_ms or ts_s,
        started_at_ms=ts_s,
        completed_at_ms=ts_e,
    )


def _map_image_generation(
    item: CodexItem,
    ts_s: int,
    ts_e: int,
    cid: str,
    codex_home: Path,
    add_asset_fn,
) -> MappedTool:
    d = item.data
    prompt = str(d.get("revisedPrompt") or d.get("prompt") or "")
    saved_path = str(d.get("savedPath") or "")
    status = str(d.get("status") or "completed")
    is_err = status in ("failed", "error") or bool(d.get("failure"))

    # Locate generated image bytes (savedPath or ~/.codex/generated_images/<tid>/<id>.png)
    cand_paths = []
    if saved_path:
        cand_paths.append(Path(saved_path))
    if item.item_id:
        cand_paths.append(codex_home / "generated_images" / cid / f"{item.item_id}.png")
        cand_paths.append(codex_home / "generated_images" / cid / f"{item.item_id}.jpg")

    dest_uri = ""
    dest_path_str = saved_path
    img_bytes_len = 0
    mime = "image/png"
    media_list: List[MediaAttachment] = []

    for cp in cand_paths:
        if cp.is_file():
            try:
                raw = cp.read_bytes()
                img_bytes_len = len(raw)
                mime = mimetypes.guess_type(str(cp))[0] or "image/png"
                rel = f"generated_{sanitize_filename(cp.name, 'image.png')}"
                dest_uri = add_asset_fn(rel, raw)
                dest_path_str = file_uri_to_path(dest_uri) if dest_uri.startswith("file://") else rel
                media_list.append(
                    MediaAttachment(
                        mime_type=mime,
                        uri=dest_uri,
                        display_name=cp.name,
                        description=prompt[:200],
                    )
                )
                break
            except Exception:
                pass

    args = {"Prompt": prompt}
    res_msg = f"Generated image: {dest_uri or saved_path}" if not is_err else f"Image generation failed: {d.get('failure')}"
    res = f"Created At: {ms_to_local_iso(ts_s)}\nCompleted At: {ms_to_local_iso(ts_e)}\n{res_msg}"

    blocks = [encode_content_block_media(dest_uri, mime, prompt[:120])] if dest_uri else None
    render = encode_step_render_info(
        title_prefix="Generated",
        icon="image",
        aggregation_key="images",
        title_parts=[encode_title_part_text(prompt[:80] or "image")],
        content_blocks=blocks,
    )
    payload = encode_tool_generate_image(
        prompt=prompt,
        image_paths=[dest_path_str] if dest_path_str else [],
        image_uri=dest_uri,
        image_num_bytes=img_bytes_len,
        image_name=os.path.basename(dest_path_str) if dest_path_str else "",
    )
    return MappedTool(
        call_id=item.item_id or f"imgg_{item.rollout_ordinal}",
        tool_name="generate_image",
        args_dict=args,
        arguments_json=json.dumps(args, ensure_ascii=False),
        result_text=res,
        full_output_text=None,
        render_info=render,
        payload_any=payload,
        created_at_ms=item.created_at_ms or ts_s,
        started_at_ms=ts_s,
        completed_at_ms=ts_e,
        is_error=is_err,
        error_message=str(d.get("failure") or "") if is_err else "",
        media=media_list,
    )


def _map_dynamic_tool_call(item: CodexItem, ts_s: int, ts_e: int) -> MappedTool:
    d = item.data
    tool = str(d.get("tool") or "tool")
    ns = d.get("namespace")
    raw_args = d.get("arguments")
    if isinstance(raw_args, dict):
        args_dict = {str(k): (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) for k, v in raw_args.items()}
        args_json = json.dumps(raw_args, ensure_ascii=False)
    else:
        args_dict = {"arguments": str(raw_args or "")}
        args_json = json.dumps(args_dict, ensure_ascii=False)

    content_items = d.get("contentItems") or []
    parts = []
    for ci in content_items:
        if isinstance(ci, dict) and ci.get("text"):
            parts.append(str(ci["text"]))
        else:
            parts.append(json.dumps(ci, ensure_ascii=False))
    body = "\n".join(parts)
    full_text = f"Created At: {ms_to_local_iso(ts_s)}\nCompleted At: {ms_to_local_iso(ts_e)}\n{body}"
    inline_res, spilled = _split_large_output(full_text)

    status = str(d.get("status") or "completed")
    is_err = (status in ("failed", "error")) or (d.get("success") is False)

    # Special case: Codex / imported `Read` dynamic tool -> native Antigravity `read_file`
    if tool == "Read" and isinstance(raw_args, dict) and raw_args.get("path"):
        fpath = str(raw_args["path"])
        furi = path_to_file_uri(fpath)
        r_args = {"Path": fpath}
        render = encode_step_render_info(
            title_prefix="Analyzed",
            icon="description",
            aggregation_key="files",
            title_parts=[encode_title_part_file(furi)],
        )
        payload = encode_tool_read_file(fpath)
        return MappedTool(
            call_id=item.item_id or f"dyn_{item.rollout_ordinal}",
            tool_name="read_file",
            args_dict=r_args,
            arguments_json=json.dumps(r_args, ensure_ascii=False),
            result_text=inline_res,
            full_output_text=spilled,
            render_info=render,
            payload_any=payload,
            created_at_ms=item.created_at_ms or ts_s,
            started_at_ms=ts_s,
            completed_at_ms=ts_e,
            is_error=is_err,
            error_message=body[:500] if is_err else "",
        )

    label = f"{ns}/{tool}" if ns else tool
    render = encode_step_render_info(
        title_prefix="Ran",
        icon="build",
        aggregation_key="actions",
        title_parts=[encode_title_part_text(label)],
    )
    return MappedTool(
        call_id=item.item_id or f"dyn_{item.rollout_ordinal}",
        tool_name=tool,
        args_dict=args_dict,
        arguments_json=args_json,
        result_text=inline_res,
        full_output_text=spilled,
        render_info=render,
        payload_any=b"",
        created_at_ms=item.created_at_ms or ts_s,
        started_at_ms=ts_s,
        completed_at_ms=ts_e,
        is_error=is_err,
        error_message=body[:500] if is_err else "",
    )


def _format_inline_subagent_card(
    *,
    role: str,
    child_id: str,
    prompt: str,
    info: Optional[InlineSubagentSummary],
    archive_uri: Optional[str],
) -> Tuple[str, str]:
    """Return (markdown_card, full_tool_result_text) for an inlined subagent step."""
    md_parts: List[str] = []
    res_parts: List[str] = [f"Subagent: {role} ({child_id})"]

    eff_prompt = prompt or (info.prompt if info else "")
    if eff_prompt:
        md_parts.append(f"**任务指令**\n\n{eff_prompt[:2500]}" + ("..." if len(eff_prompt) > 2500 else ""))
        res_parts.append(f"Prompt:\n{eff_prompt}")

    if info:
        if info.step_summary_lines:
            md_parts.append("**执行摘要**\n\n" + "\n".join(info.step_summary_lines))
            res_parts.append("Actions:\n" + "\n".join(info.step_summary_lines))
        if info.final_answer:
            ans_inline = info.final_answer[:6000] + ("\n\n... *(完整输出见下方存档)*" if len(info.final_answer) > 6000 else "")
            md_parts.append(f"**最终结论**\n\n{ans_inline}")
            res_parts.append(f"Final Answer:\n{info.final_answer}")
        if archive_uri:
            md_parts.append(f"📎 [查看完整子代理记录 ({os.path.basename(info.asset_rel_path)})]({archive_uri})")
            res_parts.append(f"Transcript: {archive_uri}")

    md_card = "\n\n---\n\n".join(md_parts) if md_parts else f"Subagent `{role}` (`{child_id}`)"
    return md_card, "\n\n".join(res_parts)


def _map_collab_agent_tool(
    item: CodexItem,
    ts_s: int,
    ts_e: int,
    agy_paths: AgyPaths,
    ws_uris: List[str],
    subagent_roles: Dict[str, str],
    *,
    is_separate_fn: Optional[Any] = None,
    inline_subagents: Optional[Dict[str, InlineSubagentSummary]] = None,
    ensure_inline_asset_fn: Optional[Any] = None,
) -> MappedTool:
    d = item.data
    ctool = str(d.get("tool") or "collab")
    receivers = [str(r) for r in (d.get("receiverThreadIds") or []) if r]
    prompt = str(d.get("prompt") or "")
    agents_states = d.get("agentsStates") or {}
    status = str(d.get("status") or "completed")
    is_err = status in ("failed", "error")
    inline_map = inline_subagents or {}

    def _use_separate(child_id: str) -> bool:
        if is_separate_fn is None:
            return False
        return bool(is_separate_fn(child_id))

    def log_uri_for(child_id: str) -> str:
        return path_to_file_uri(
            str(agy_paths.brain_dir / child_id / ".system_generated" / "logs" / "transcript.jsonl")
        )

    if ctool in ("spawn_agent", "spawn") and receivers:
        child_id = receivers[0]
        info = inline_map.get(child_id)
        role = (info.role if info else "") or subagent_roles.get(child_id) or "Subagent"

        if _use_separate(child_id):
            luri = log_uri_for(child_id)
            args = {"TypeName": "self", "Role": role, "Prompt": prompt}
            res_json = json.dumps(
                {"conversationId": child_id, "logAbsoluteUri": luri},
                ensure_ascii=False,
                indent=2,
            )
            res = f"Created At: {ms_to_local_iso(ts_s)}\nCompleted At: {ms_to_local_iso(ts_e)}\nCreated the following subagent:\n{res_json}"
            render = encode_step_render_info(
                title_prefix="Started",
                icon="robot_2",
                aggregation_key="actions",
                title_parts=[encode_title_part_agent(child_id, role, log_uri=luri, workspace_uris=ws_uris)],
            )
            payload = encode_tool_invoke_subagent(
                type_name="self",
                role=role,
                initial_prompt=prompt,
                conversation_id=child_id,
                log_uri=luri,
                workspace_uris=ws_uris,
            )
            return MappedTool(
                call_id=item.item_id or f"collab_{item.rollout_ordinal}",
                tool_name="invoke_subagent",
                args_dict=args,
                arguments_json=json.dumps(args, ensure_ascii=False),
                result_text=res,
                full_output_text=None,
                render_info=render,
                payload_any=payload,
                created_at_ms=item.created_at_ms or ts_s,
                started_at_ms=ts_s,
                completed_at_ms=ts_e,
                is_error=is_err,
            )

        # Inline / non-separate mode: 0 extra .db slots, no broken AgentRef link!
        archive_uri = ensure_inline_asset_fn(child_id) if ensure_inline_asset_fn else None
        md_card, body_text = _format_inline_subagent_card(
            role=role,
            child_id=child_id,
            prompt=prompt,
            info=info,
            archive_uri=archive_uri,
        )
        full_res = f"Created At: {ms_to_local_iso(ts_s)}\nCompleted At: {ms_to_local_iso(ts_e)}\n{body_text}"
        inline_res, spilled = _split_large_output(full_res)
        args = {"TypeName": "inline", "Role": role, "Prompt": prompt}
        render = encode_step_render_info(
            title_prefix="Subagent",
            icon="robot_2",
            aggregation_key="actions",
            title_parts=[encode_title_part_text(role)],
            content_blocks=[encode_content_block_markdown(md_card)],
        )
        return MappedTool(
            call_id=item.item_id or f"collab_{item.rollout_ordinal}",
            tool_name="subagent_inline",
            args_dict=args,
            arguments_json=json.dumps(args, ensure_ascii=False),
            result_text=inline_res,
            full_output_text=spilled,
            render_info=render,
            payload_any=b"",
            created_at_ms=item.created_at_ms or ts_s,
            started_at_ms=ts_s,
            completed_at_ms=ts_e,
            is_error=is_err or (info is not None and info.status == "error"),
        )

    if ctool in ("send_input", "send_message") and receivers:
        child_id = receivers[0]
        info = inline_map.get(child_id)
        role = (info.role if info else "") or subagent_roles.get(child_id) or child_id
        args = {"Recipient": child_id, "Message": prompt}
        res = f'Created At: {ms_to_local_iso(ts_s)}\nCompleted At: {ms_to_local_iso(ts_e)}\nMessage sent to "{role}" ({child_id}).'
        render = encode_step_render_info(
            title_prefix="Messaged",
            icon="send",
            aggregation_key="actions",
            title_parts=[encode_title_part_text(role)],
            content_blocks=[encode_content_block_markdown(f"> {prompt[:4000]}")] if prompt else None,
        )
        payload = encode_tool_send_message(child_id, role, prompt) if _use_separate(child_id) else b""
        return MappedTool(
            call_id=item.item_id or f"collab_{item.rollout_ordinal}",
            tool_name="send_message",
            args_dict=args,
            arguments_json=json.dumps(args, ensure_ascii=False),
            result_text=res,
            full_output_text=None,
            render_info=render,
            payload_any=payload,
            created_at_ms=item.created_at_ms or ts_s,
            started_at_ms=ts_s,
            completed_at_ms=ts_e,
            is_error=is_err,
        )

    # wait / close_agent / other collab tool
    all_ids = list(dict.fromkeys(receivers + list(agents_states.keys())))
    state_json = json.dumps(agents_states, ensure_ascii=False, indent=2) if agents_states else status
    full_res = f"Created At: {ms_to_local_iso(ts_s)}\nCompleted At: {ms_to_local_iso(ts_e)}\n{state_json}"
    inline_res, spilled = _split_large_output(full_res)
    args = {"Action": ctool, "ConversationIds": json.dumps(all_ids)}

    all_separate = bool(all_ids) and all(_use_separate(cid_i) for cid_i in all_ids)
    if all_separate:
        blocks = [
            encode_content_block_agent(
                cid_i,
                subagent_roles.get(cid_i) or cid_i,
                subtitle=str((agents_states.get(cid_i) or {}).get("status") or ""),
                log_uri=log_uri_for(cid_i),
                workspace_uris=ws_uris,
            )
            for cid_i in all_ids
        ]
        render = encode_step_render_info(
            title_prefix="Subagents",
            icon="robot_2",
            aggregation_key="actions",
            title_parts=[encode_title_part_text(ctool)],
            content_blocks=blocks or None,
        )
        payload = encode_tool_manage_subagents(ctool, all_ids)
    else:
        md_blocks: List[str] = []
        role_labels: List[str] = []
        for cid_i in all_ids:
            info = inline_map.get(cid_i)
            rname = (info.role if info else "") or subagent_roles.get(cid_i) or cid_i[:8]
            role_labels.append(rname)
            st_obj = agents_states.get(cid_i) if isinstance(agents_states, dict) else None
            st_status = str((st_obj or {}).get("status") or (info.status if info else status))
            st_msg = str((st_obj or {}).get("message") or "").strip()
            if not st_msg and info and info.final_answer and ctool == "wait":
                st_msg = info.final_answer
            entry_lines = [f"### {rname} (`{st_status}`)"]
            if st_msg:
                entry_lines.append(st_msg[:4000] + ("..." if len(st_msg) > 4000 else ""))
            archive_uri = ensure_inline_asset_fn(cid_i) if ensure_inline_asset_fn else None
            if archive_uri and info:
                entry_lines.append(f"📎 [查看完整子代理记录 ({os.path.basename(info.asset_rel_path)})]({archive_uri})")
            md_blocks.append("\n\n".join(entry_lines))
        title_suffix = f"{ctool} ({', '.join(role_labels[:4])})" if role_labels else ctool
        render = encode_step_render_info(
            title_prefix="Subagent",
            icon="robot_2",
            aggregation_key="actions",
            title_parts=[encode_title_part_text(title_suffix)],
            content_blocks=[encode_content_block_markdown("\n\n---\n\n".join(md_blocks))] if md_blocks else None,
        )
        payload = b""

    return MappedTool(
        call_id=item.item_id or f"collab_{item.rollout_ordinal}",
        tool_name="manage_subagents" if all_separate else "subagent_wait_inline",
        args_dict=args,
        arguments_json=json.dumps(args, ensure_ascii=False),
        result_text=inline_res,
        full_output_text=spilled,
        render_info=render,
        payload_any=payload,
        created_at_ms=item.created_at_ms or ts_s,
        started_at_ms=ts_s,
        completed_at_ms=ts_e,
        is_error=is_err,
    )


def _map_subagent_activity(
    item: CodexItem,
    ts_s: int,
    ts_e: int,
    agy_paths: AgyPaths,
    ws_uris: List[str],
    subagent_roles: Dict[str, str],
    *,
    is_separate_fn: Optional[Any] = None,
    inline_subagents: Optional[Dict[str, InlineSubagentSummary]] = None,
    ensure_inline_asset_fn: Optional[Any] = None,
) -> MappedTool:
    d = item.data
    kind = str(d.get("kind") or "started")
    child_id = str(d.get("agentThreadId") or "")
    apath = str(d.get("agentPath") or "")
    inline_map = inline_subagents or {}
    info = inline_map.get(child_id)
    role = (info.role if info else "") or subagent_roles.get(child_id) or os.path.basename(apath.rstrip("/")) or "Subagent"

    use_separate = bool(is_separate_fn and is_separate_fn(child_id))
    if use_separate:
        luri = path_to_file_uri(
            str(agy_paths.brain_dir / child_id / ".system_generated" / "logs" / "transcript.jsonl")
        )
        args = {"TypeName": "self", "Role": role, "Prompt": apath}
        res = f"Created At: {ms_to_local_iso(ts_s)}\nCompleted At: {ms_to_local_iso(ts_e)}\nSubagent {kind}: {child_id} ({role})"
        prefix = "Started" if kind == "started" else ("Finished" if kind == "completed" else "Subagent")
        render = encode_step_render_info(
            title_prefix=prefix,
            icon="robot_2",
            aggregation_key="actions",
            title_parts=[encode_title_part_agent(child_id, role, log_uri=luri, workspace_uris=ws_uris)],
        )
        payload = encode_tool_invoke_subagent(
            type_name="self",
            role=role,
            initial_prompt=apath,
            conversation_id=child_id,
            log_uri=luri,
            workspace_uris=ws_uris,
        )
        return MappedTool(
            call_id=item.item_id or f"subact_{item.rollout_ordinal}",
            tool_name="invoke_subagent",
            args_dict=args,
            arguments_json=json.dumps(args, ensure_ascii=False),
            result_text=res,
            full_output_text=None,
            render_info=render,
            payload_any=payload,
            created_at_ms=item.created_at_ms or ts_s,
            started_at_ms=ts_s,
            completed_at_ms=ts_e,
            is_error=(kind in ("errored", "failed")),
        )

    archive_uri = ensure_inline_asset_fn(child_id) if ensure_inline_asset_fn else None
    md_card, body_text = _format_inline_subagent_card(
        role=role,
        child_id=child_id,
        prompt="",
        info=info,
        archive_uri=archive_uri,
    )
    full_res = f"Created At: {ms_to_local_iso(ts_s)}\nCompleted At: {ms_to_local_iso(ts_e)}\nSubagent {kind}: {body_text}"
    inline_res, spilled = _split_large_output(full_res)
    args = {"TypeName": "inline", "Role": role, "Kind": kind}
    prefix = "Subagent"
    render = encode_step_render_info(
        title_prefix=prefix,
        icon="robot_2",
        aggregation_key="actions",
        title_parts=[encode_title_part_text(f"{role} ({kind})")],
        content_blocks=[encode_content_block_markdown(md_card)],
    )
    return MappedTool(
        call_id=item.item_id or f"subact_{item.rollout_ordinal}",
        tool_name="subagent_inline",
        args_dict=args,
        arguments_json=json.dumps(args, ensure_ascii=False),
        result_text=inline_res,
        full_output_text=spilled,
        render_info=render,
        payload_any=b"",
        created_at_ms=item.created_at_ms or ts_s,
        started_at_ms=ts_s,
        completed_at_ms=ts_e,
        is_error=(kind in ("errored", "failed")),
    )
