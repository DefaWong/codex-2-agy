"""Tests for lossless Codex item mapper, image recovery, diff parsing, and subagents."""

from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

from codex_2_agy.agy_proto import (
    OUTSIDE_OF_PROJECT_ID,
    STEP_STATUS_ERROR,
    STEP_TYPE_GENERIC,
    STEP_TYPE_PLANNER_RESPONSE,
    STEP_TYPE_SYSTEM_MESSAGE,
    STEP_TYPE_USER_INPUT,
)
from codex_2_agy.agy_store import AgyPaths
from codex_2_agy.codex_reader import (
    CodexItem,
    CodexReader,
    CodexThread,
    RolloutAugmentation,
    RolloutImage,
    RolloutUsage,
    ThreadHistoryData,
)
from codex_2_agy.mapper import (
    INLINE_OUTPUT_LIMIT,
    build_inline_subagent_summary,
    clean_codex_title,
    clean_reasoning_parts,
    format_codex_assistant_text,
    format_codex_user_text,
    map_codex_thread,
    parse_unified_diff,
)
from codex_2_agy.protobuf_wire import decode_fields

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _make_thread(tid: str = "11111111-1111-4111-8111-111111111111", **overrides) -> CodexThread:
    base = dict(
        id=tid,
        rollout_path="",
        created_at_ms=1_760_000_000_000,
        updated_at_ms=1_760_000_050_000,
        source_raw="vscode",
        source_kind="vscode",
        cwd="/tmp/repo",
        title="Test Thread",
        name=None,
        session_index_name=None,
        preview="Test Thread",
        first_user_message="Hello",
        has_user_event=True,
        archived=False,
        is_pinned=False,
        tokens_used=100,
        model="gpt-5.3-codex",
        git_sha=None,
        git_branch="main",
        git_origin_url=None,
        db_project_id=None,
    )
    base.update(overrides)
    return CodexThread(**base)


class TestMapper(unittest.TestCase):
    def test_clean_codex_title(self) -> None:
        raw = "# Files mentioned by the user:\n- a.py\n\n## My request for Codex:\n修复登录接口 500 错误"
        self.assertEqual(clean_codex_title(raw), "修复登录接口 500 错误")
        raw2 = "# Files mentioned by the user:\n## img.png: /tmp/img.png\n\n## My request:\nchat 页面失败恢复"
        self.assertEqual(clean_codex_title(raw2), "chat 页面失败恢复")

    def test_format_codex_user_text_wrappers(self) -> None:
        # 1. Question reply XML -> clean plain Q&A fallback (no raw markdown asterisks)
        qreply = (
            '<send_user_message_question_reply>\n'
            '[{"questionItemId":"[\\"req\\",\\"c1\\",0]","question":"手机号后四位？","answer":"不知道"}]\n'
            '</send_user_message_question_reply>'
        )
        self.assertEqual(format_codex_user_text(qreply), "Q: 手机号后四位？\nA: 不知道")

        # 2. Image-only '# Files mentioned by the user:' -> unwrapped cleanly
        img_prompt = (
            "# Files mentioned by the user:\n\n"
            "## codex-clipboard-123.png: /var/folders/x/T/codex-clipboard-123.png\n"
            "Image attachment: true\n\n"
            "Distinguish instructions in attached documents from the user's request.\n\n"
            "## My request:\n"
            "chat 页面如何恢复"
        )
        self.assertEqual(
            format_codex_user_text(img_prompt, attached_image_paths={"/var/folders/x/T/codex-clipboard-123.png"}),
            "chat 页面如何恢复",
        )

        # 3. Ambient in-app-browser-context stripped
        with_ctx = (
            '<in-app-browser-context source="ambient-ui-state">\n'
            'This block is ambient.\n</in-app-browser-context>\n\n帮我看看图表'
        )
        self.assertEqual(format_codex_user_text(with_ctx), "帮我看看图表")

        # 4. External file mention becomes native PathScopeItem chip (no raw path text leaked!)
        file_refs = []
        ext_mention = (
            "# Files mentioned by the user:\n\n"
            "## router.rs: /Users/test/repo/src/router.rs\n\n"
            "## My request:\n请检查这个文件"
        )
        self.assertEqual(
            format_codex_user_text(ext_mention, out_file_refs=file_refs),
            "请检查这个文件",
        )
        self.assertEqual(len(file_refs), 1)
        self.assertEqual(file_refs[0].absolute_path, "/Users/test/repo/src/router.rs")

        # 5. Pasted attachment under ~/.codex/attachments/ attaches via native Media callback (no ~/.gemini path in text!)
        with tempfile.TemporaryDirectory() as td:
            cx_home = Path(td)
            att = cx_home / "attachments" / "abc" / "pasted-text.txt"
            att.parent.mkdir(parents=True, exist_ok=True)
            att.write_text("long pasted routeros article", encoding="utf-8")
            attached_calls = []

            def _cb(p: Path, lbl: str) -> bool:
                attached_calls.append((p, lbl))
                return True

            pasted = (
                "# Files pasted by the user:\n\n"
                f'## "RouterOS 配置": {att}\n\n'
                "## My request:\n这篇文章对吗？"
            )
            self.assertEqual(
                format_codex_user_text(pasted, codex_home=cx_home, attach_file_media_fn=_cb),
                "这篇文章对吗？",
            )
            self.assertEqual(attached_calls, [(att, "RouterOS 配置")])

    def test_parse_unified_diff(self) -> None:
        u_diff = "@@ -1,3 +1,3 @@\n line1\n-old\n+new1\n+new2\n line3"
        lines, adds, dels = parse_unified_diff(u_diff, "update")
        self.assertEqual(adds, 2)
        self.assertEqual(dels, 1)
        self.assertEqual(len(lines), 5)

        add_lines, a2, d2 = parse_unified_diff("a\nb\nc", "add")
        self.assertEqual((a2, d2, len(add_lines)), (3, 0, 3))

        del_lines, a3, d3 = parse_unified_diff("x\ny", "delete")
        self.assertEqual((a3, d3, len(del_lines)), (0, 2, 2))

    def test_all_item_types_and_strict_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agy_paths = AgyPaths.custom(root / "agy")
            codex_home = root / "codex"
            gen_dir = codex_home / "generated_images" / "11111111-1111-4111-8111-111111111111"
            gen_dir.mkdir(parents=True, exist_ok=True)
            (gen_dir / "img-gen-1.png").write_bytes(TINY_PNG)

            expired_clip = "/var/folders/xx/yy/T/codex-clipboard-expired.png"
            rimg = RolloutImage(mime_type="image/png", data=TINY_PNG, path_hint=expired_clip, ordinal=1)
            aug = RolloutAugmentation(
                images_by_path={expired_clip: rimg},
                images_by_user_msg_index={0: [rimg]},
                usages=[
                    RolloutUsage(1, "t1", "resp-1", 500, 200, 0, 120, 40),
                    RolloutUsage(2, "t1", "resp-2", 600, 300, 0, 80, 10),
                ],
                turn_models={"t1": "gpt-5.3-codex"},
            )
            big_output = "LINE\n" * (INLINE_OUTPUT_LIMIT // 2)
            items = [
                CodexItem(
                    "t1", "u1", "userMessage", 1, 1_760_000_001_000, 1_760_000_001_000, 1_760_000_001_000,
                    {
                        "type": "userMessage",
                        "content": [
                            {"type": "text", "text": "帮我看看这张图并改代码"},
                            {"type": "localImage", "path": expired_clip},
                            {"type": "skill", "name": "debugging"},
                        ],
                    },
                ),
                CodexItem(
                    "t1", "r1", "reasoning", 2, 1_760_000_002_000, 1_760_000_002_000, 1_760_000_003_000,
                    {"type": "reasoning", "summary": ["先检查代码"], "content": []},
                ),
                CodexItem(
                    "t1", "cmd1", "commandExecution", 3, 1_760_000_003_000, 1_760_000_003_000, 1_760_000_004_000,
                    {
                        "type": "commandExecution",
                        "command": "/bin/zsh -lc 'pytest -q'",
                        "cwd": "/tmp/repo",
                        "exitCode": 1,
                        "status": "failed",
                        "aggregatedOutput": big_output,
                    },
                ),
                CodexItem(
                    "t1", "fc1", "fileChange", 4, 1_760_000_004_000, 1_760_000_004_000, 1_760_000_005_000,
                    {
                        "type": "fileChange",
                        "status": "completed",
                        "changes": [
                            {"path": "/tmp/repo/a.py", "kind": {"type": "update"}, "diff": "@@ -1 +1 @@\n-old\n+new"},
                            {"path": "/tmp/repo/b.py", "kind": {"type": "add"}, "diff": "print(1)\nprint(2)"},
                            {"path": "/tmp/repo/c.py", "kind": {"type": "delete"}, "diff": "bye"},
                        ],
                    },
                ),
                CodexItem(
                    "t1", "mcp1", "mcpToolCall", 5, 1_760_000_005_000, 1_760_000_005_000, 1_760_000_006_000,
                    {
                        "type": "mcpToolCall",
                        "server": "github",
                        "tool": "get_issue",
                        "arguments": {"id": 42},
                        "result": {"content": [{"type": "text", "text": "issue body"}]},
                        "status": "completed",
                    },
                ),
                CodexItem(
                    "t1", "web1", "webSearch", 6, 1_760_000_006_000, 1_760_000_006_000, 1_760_000_007_000,
                    {
                        "type": "webSearch",
                        "query": "python fts5",
                        "results": [{"url": "https://sqlite.org/fts5.html", "title": "FTS5", "snippet": "full text"}],
                    },
                ),
                CodexItem(
                    "t1", "imgv1", "imageView", 7, 1_760_000_007_000, 1_760_000_007_000, 1_760_000_008_000,
                    {"type": "imageView", "path": "/tmp/repo/diagram.png"},
                ),
                CodexItem(
                    "t1", "img-gen-1", "imageGeneration", 8, 1_760_000_008_000, 1_760_000_008_000, 1_760_000_009_000,
                    {"type": "imageGeneration", "prompt": "architecture diagram", "status": "completed"},
                ),
                CodexItem(
                    "t1", "dyn1", "dynamicToolCall", 9, 1_760_000_009_000, 1_760_000_009_000, 1_760_000_010_000,
                    {
                        "type": "dynamicToolCall",
                        "tool": "Read",
                        "arguments": {"path": "/tmp/repo/a.py"},
                        "contentItems": [{"type": "text", "text": "new"}],
                        "status": "completed",
                    },
                ),
                CodexItem(
                    "t1", "col1", "collabAgentToolCall", 10, 1_760_000_010_000, 1_760_000_010_000, 1_760_000_011_000,
                    {
                        "type": "collabAgentToolCall",
                        "tool": "spawn_agent",
                        "receiverThreadIds": ["22222222-2222-4222-8222-222222222222"],
                        "prompt": "inspect tests",
                        "status": "completed",
                    },
                ),
                # Duplicate subAgentActivity sharing col1 item_id should be skipped
                CodexItem(
                    "t1", "col1", "subAgentActivity", 11, 1_760_000_011_000, 1_760_000_011_000, 1_760_000_011_000,
                    {"type": "subAgentActivity", "kind": "started", "agentThreadId": "22222222-2222-4222-8222-222222222222"},
                ),
                CodexItem(
                    "t1", "sl1", "sleep", 12, 1_760_000_012_000, 1_760_000_012_000, 1_760_000_014_000,
                    {"type": "sleep", "durationMs": 2000},
                ),
                CodexItem(
                    "t1", "plan1", "plan", 13, 1_760_000_015_000, 1_760_000_015_000, 1_760_000_015_000,
                    {"type": "plan", "text": "# Plan\n1. Step A\n2. Step B"},
                ),
                CodexItem(
                    "t1", "cc1", "contextCompaction", 14, 1_760_000_016_000, 1_760_000_016_000, 1_760_000_016_000,
                    {"type": "contextCompaction"},
                ),
                CodexItem(
                    "t1", "fco1", "functionCallOutput", 15, 1_760_000_017_000, 1_760_000_017_000, 1_760_000_017_000,
                    {"type": "functionCallOutput", "name": "hook", "output": "ok"},
                ),
                CodexItem(
                    "t1", "am1", "agentMessage", 16, 1_760_000_018_000, 1_760_000_018_000, 1_760_000_019_000,
                    {"type": "agentMessage", "text": "全部修好了！", "phase": "final_answer"},
                ),
            ]
            mapped = map_codex_thread(
                _make_thread(),
                ThreadHistoryData(turns=[], items=items, augmentation=aug),
                project_id=OUTSIDE_OF_PROJECT_ID,
                project_name="",
                workspaces=[],
                agy_paths=agy_paths,
                codex_home=codex_home,
                subagent_roles={"22222222-2222-4222-8222-222222222222": "Worker"},
            )

            # Verify recovered assets: 1 clipboard PNG + 1 generated PNG + 1 plan.md + 1 plan.md.metadata.json
            asset_paths = [a.rel_path for a in mapped.assets]
            self.assertTrue(any(p.startswith(".user_uploaded/") and p.endswith(".png") for p in asset_paths))
            self.assertTrue(any(p.startswith("generated_") and p.endswith(".png") for p in asset_paths))
            self.assertIn("implementation_plan.md", asset_paths)
            self.assertIn("implementation_plan.md.metadata.json", asset_paths)

            # Verify large command output spilled to output.txt and error status preserved
            spilled = [s for s in mapped.steps if s.large_output_file is not None]
            self.assertEqual(len(spilled), 1)
            self.assertEqual(spilled[0].status, STEP_STATUS_ERROR)
            self.assertIsNotNone(spilled[0].error_details_bytes)

            # Verify strict PLANNER_RESPONSE -> GENERIC pairing across all steps
            expected_ids = []
            for s in mapped.steps:
                pfields = decode_fields(s.payload_bytes)
                if s.step_type == STEP_TYPE_PLANNER_RESPONSE:
                    self.assertEqual(expected_ids, [])
                    for f, _wt, v in pfields:
                        if f == 20:
                            for sf, _swt, sv in decode_fields(v):
                                if sf == 7:
                                    for tf, _twt, tv in decode_fields(sv):
                                        if tf == 1:
                                            expected_ids.append(tv.decode())
                elif s.step_type == STEP_TYPE_GENERIC:
                    self.assertTrue(len(expected_ids) > 0)
                    tc_id = ""
                    for mf, _mwt, mv in decode_fields(s.metadata_bytes):
                        if mf == 4:
                            for tf, _twt, tv in decode_fields(mv):
                                if tf == 1:
                                    tc_id = tv.decode()
                    self.assertEqual(tc_id, expected_ids.pop(0))
                else:
                    self.assertEqual(expected_ids, [])
            self.assertEqual(expected_ids, [])

    def test_subagent_step_zero_is_system_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agy_paths = AgyPaths.custom(root / "agy")
            sub = _make_thread(
                tid="22222222-2222-4222-8222-222222222222",
                source_kind="subagent_spawn",
                parent_thread_id="11111111-1111-4111-8111-111111111111",
                nesting_depth=1,
                agent_nickname="Socrates",
                agent_role="explorer",
            )
            items = [
                CodexItem("t1", "u1", "userMessage", 1, 1000, 1000, 1000, {"type": "userMessage", "content": [{"type": "text", "text": "Check module X"}]}),
                CodexItem("t1", "a1", "agentMessage", 2, 2000, 2000, 2000, {"type": "agentMessage", "text": "Checked!", "phase": "final_answer"}),
            ]
            mapped = map_codex_thread(
                sub,
                ThreadHistoryData(turns=[], items=items, augmentation=RolloutAugmentation()),
                project_id=OUTSIDE_OF_PROJECT_ID,
                project_name="",
                workspaces=[],
                agy_paths=agy_paths,
                codex_home=root,
                root_conversation_id="11111111-1111-4111-8111-111111111111",
            )
            self.assertEqual(mapped.steps[0].step_type, STEP_TYPE_SYSTEM_MESSAGE)
            self.assertEqual(mapped.steps[1].step_type, STEP_TYPE_PLANNER_RESPONSE)
            self.assertEqual(mapped.title, "")
            self.assertEqual(mapped.preview, "")
            self.assertEqual(mapped.nesting_depth, 1)
            self.assertEqual(mapped.agent_name, "explorer")

    def test_rollout_only_multi_turn_fallback(self) -> None:
        """Verify rollout-only threads (no thread_history_1.sqlite rows) preserve all turns & tool calls."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rollout = root / "rollout-test.jsonl"
            lines = [
                {"timestamp": "2026-10-01T00:00:01Z", "type": "turn_context", "payload": {"turn_id": "t1", "model": "gpt-5.3-codex"}},
                {"timestamp": "2026-10-01T00:00:02Z", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<environment_context>skip</environment_context>"}]}},
                {"timestamp": "2026-10-01T00:00:03Z", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "First question"}]}},
                {"timestamp": "2026-10-01T00:00:04Z", "type": "response_item", "payload": {"type": "reasoning", "summary": [{"text": "Thinking 1"}]}},
                {"timestamp": "2026-10-01T00:00:05Z", "type": "response_item", "payload": {"type": "function_call", "call_id": "call_1", "name": "exec_command", "arguments": '{"cmd":"echo hi","workdir":"/tmp"}'}},
                {"timestamp": "2026-10-01T00:00:06Z", "type": "response_item", "payload": {"type": "function_call_output", "call_id": "call_1", "output": "hi\n"}},
                {"timestamp": "2026-10-01T00:00:07Z", "type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Answer 1"}]}},
                {"timestamp": "2026-10-01T00:00:08Z", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Second question"}]}},
                {"timestamp": "2026-10-01T00:00:09Z", "type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Answer 2"}]}},
            ]
            rollout.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

            with CodexReader(root) as reader:
                t = _make_thread(rollout_path=str(rollout))
                hist = reader.load_thread_history(t, parse_rollout=True)

            types = [it.item_type for it in hist.items]
            self.assertEqual(
                types,
                ["userMessage", "reasoning", "commandExecution", "agentMessage", "userMessage", "agentMessage"],
            )
            self.assertEqual(hist.items[2].data["aggregatedOutput"], "hi\n")

    def test_no_duplicate_images_when_rollout_has_injected_wrappers(self) -> None:
        """Regression test: injected rollout user wrappers must NOT duplicate or shift images onto next user message."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agy_paths = AgyPaths.custom(root / "agy")
            clip = "/var/folders/xx/T/codex-clipboard-unique.png"
            b64 = base64.b64encode(TINY_PNG).decode("ascii")
            rollout = root / "rollout-wrap.jsonl"
            rollout.write_text(
                "\n".join(
                    json.dumps(x)
                    for x in [
                        {"timestamp": "2026-10-01T00:00:00Z", "type": "turn_context", "payload": {"turn_id": "t1"}},
                        {"timestamp": "2026-10-01T00:00:01Z", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<external_codex_apps_open_page>{}</external_codex_apps_open_page>"}]}},
                        {"timestamp": "2026-10-01T00:00:01Z", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": f'<image path="{clip}">'}, {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}, {"type": "input_text", "text": "</image>"}, {"type": "input_text", "text": "msg 1"}]}},
                        {"timestamp": "2026-10-01T00:00:02Z", "type": "turn_context", "payload": {"turn_id": "t2"}},
                        {"timestamp": "2026-10-01T00:00:02Z", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "msg 2 text only"}]}},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with CodexReader(root) as reader:
                aug = reader._parse_rollout(rollout, need_fallback_items=False)

            items = [
                CodexItem("t1", "u1", "userMessage", 1, 1000, 1000, 1000, {"type": "userMessage", "content": [{"type": "text", "text": "msg 1"}, {"type": "localImage", "path": clip}]}),
                CodexItem("t1", "a1", "agentMessage", 2, 1500, 1500, 1500, {"type": "agentMessage", "text": "ok", "phase": "final_answer"}),
                CodexItem("t2", "u2", "userMessage", 3, 2000, 2000, 2000, {"type": "userMessage", "content": [{"type": "text", "text": "msg 2 text only"}]}),
                CodexItem("t2", "a2", "agentMessage", 4, 2500, 2500, 2500, {"type": "agentMessage", "text": "ok2", "phase": "final_answer"}),
            ]
            mapped = map_codex_thread(
                _make_thread(),
                ThreadHistoryData(turns=[], items=items, augmentation=aug),
                project_id=OUTSIDE_OF_PROJECT_ID,
                project_name="",
                workspaces=[],
                agy_paths=agy_paths,
                codex_home=root,
            )
            self.assertEqual(len(mapped.assets), 1)
            # Ensure u1 has media and u2 has NO media
            u_steps = [s for s in mapped.steps if s.step_type == STEP_TYPE_USER_INPUT]
            self.assertEqual(len(u_steps), 2)
            self.assertIn("media", u_steps[0].transcript_entry)
            self.assertNotIn("media", u_steps[1].transcript_entry)

    def test_native_ask_question_step_for_question_reply(self) -> None:
        """Codex <send_user_message_question_reply> maps to native agy ask_question + CompletedInteraction (field 147)."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agy_paths = AgyPaths.custom(root / "agy")
            rollout = root / "rollout-askq.jsonl"
            rollout.write_text(
                "\n".join(
                    json.dumps(x, ensure_ascii=False)
                    for x in [
                        {
                            "timestamp": "2026-10-01T00:00:01Z",
                            "type": "response_item",
                            "payload": {
                                "type": "function_call",
                                "call_id": "call_mc",
                                "name": "request_user_input_async",
                                "arguments": json.dumps(
                                    {"questions": [{"title": "选哪种方式？", "options": ["方案 A", "方案 B"]}]},
                                    ensure_ascii=False,
                                ),
                            },
                        },
                        {
                            "timestamp": "2026-10-01T00:00:02Z",
                            "type": "response_item",
                            "payload": {
                                "type": "function_call",
                                "call_id": "call_free",
                                "name": "request_user_input_async",
                                "arguments": json.dumps(
                                    {"questions": [{"title": "手机号后四位？"}]},
                                    ensure_ascii=False,
                                ),
                            },
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with CodexReader(root) as reader:
                aug = reader._parse_rollout(rollout, need_fallback_items=False)

            items = [
                CodexItem("t1", "u0", "userMessage", 1, 1000, 1000, 1000, {"type": "userMessage", "content": [{"type": "text", "text": "开始"}]}),
                CodexItem(
                    "t2", "u1", "userMessage", 2, 2000, 2000, 2000,
                    {
                        "type": "userMessage",
                        "content": [
                            {
                                "type": "text",
                                "text": '<send_user_message_question_reply>\n[{"questionItemId":"[\\"request_user_input_async\\",\\"call_mc\\",0]","question":"选哪种方式？","answer":"方案 B"}]\n</send_user_message_question_reply>',
                            }
                        ],
                    },
                ),
                CodexItem(
                    "t3", "u2", "userMessage", 3, 3000, 3000, 3000,
                    {
                        "type": "userMessage",
                        "content": [
                            {
                                "type": "text",
                                "text": '<send_user_message_question_reply>\n[{"questionItemId":"[\\"request_user_input_async\\",\\"call_free\\",0]","question":"手机号后四位？","answer":"不知道"}]\n</send_user_message_question_reply>',
                            }
                        ],
                    },
                ),
            ]
            mapped = map_codex_thread(
                _make_thread(),
                ThreadHistoryData(turns=[], items=items, augmentation=aug),
                project_id=OUTSIDE_OF_PROJECT_ID,
                project_name="",
                workspaces=[],
                agy_paths=agy_paths,
                codex_home=root,
            )
            # Only u0 is a USER_INPUT; u1 and u2 became native PLANNER_RESPONSE + GENERIC ask_question pairs!
            u_steps = [s for s in mapped.steps if s.step_type == STEP_TYPE_USER_INPUT]
            g_steps = [s for s in mapped.steps if s.step_type == STEP_TYPE_GENERIC]
            self.assertEqual(len(u_steps), 1)
            self.assertEqual(len(g_steps), 2)

            # Verify field 147 (CompletedInteraction) and AskQuestion payload on both generic steps
            for gs, expected_field, expected_val in (
                (g_steps[0], 4, b"2"),  # selected_option_ids = "2" (方案 B)
                (g_steps[1], 5, "不知道".encode("utf-8")),  # write_in_response = "不知道"
            ):
                top = dict((f, v) for f, _wt, v in decode_fields(gs.payload_bytes))
                self.assertIn(147, top)
                self.assertIn(140, top)
                # Inspect CompletedInteraction.response.ask_question.responses[0]
                ci = dict((f, v) for f, _wt, v in decode_fields(top[147]))
                resp = dict((f, v) for f, _wt, v in decode_fields(ci[2]))
                aq = dict((f, v) for f, _wt, v in decode_fields(resp[22]))
                entry = dict((f, v) for f, _wt, v in decode_fields(aq[1]))
                self.assertEqual(entry[expected_field], expected_val)

    def test_prefers_full_res_disk_image_over_downscaled_rollout(self) -> None:
        """When the original image file still exists on disk, prefer full-res disk bytes over downscaled rollout base64."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agy_paths = AgyPaths.custom(root / "agy")
            disk_img = root / "fullres.png"
            full_res_bytes = TINY_PNG + b"FULLRES_EXTRA_MARKER"
            disk_img.write_bytes(full_res_bytes)

            rimg = RolloutImage(mime_type="image/png", data=TINY_PNG, path_hint=str(disk_img), ordinal=1)
            aug = RolloutAugmentation(
                images_by_path={str(disk_img): rimg},
                images_by_turn_id={"t1": [rimg]},
            )
            items = [
                CodexItem(
                    "t1", "u1", "userMessage", 1, 1000, 1000, 1000,
                    {"type": "userMessage", "content": [{"type": "text", "text": "看图"}, {"type": "localImage", "path": str(disk_img)}]},
                ),
            ]
            mapped = map_codex_thread(
                _make_thread(),
                ThreadHistoryData(turns=[], items=items, augmentation=aug),
                project_id=OUTSIDE_OF_PROJECT_ID,
                project_name="",
                workspaces=[],
                agy_paths=agy_paths,
                codex_home=root,
            )
            self.assertEqual(len(mapped.assets), 1)
            self.assertEqual(mapped.assets[0].data, full_res_bytes)

    def test_inline_subagent_renders_markdown_card_and_archives_transcript(self) -> None:
        """Default inline subagent mode emits Markdown card + brain/subagents/*.md with zero dangling AgentRef."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agy_paths = AgyPaths.custom(root / "agy")
            sub_id = "22222222-2222-4222-8222-222222222222"
            sub_thread = _make_thread(
                tid=sub_id,
                source_kind="subagent_spawn",
                parent_thread_id="11111111-1111-4111-8111-111111111111",
                nesting_depth=1,
                agent_path="/root/netcheck",
                agent_role="explorer",
            )
            sub_items = [
                CodexItem("st1", "su1", "userMessage", 1, 1000, 1000, 1000, {"type": "userMessage", "content": [{"type": "text", "text": "Inspect nftables"}]}),
                CodexItem("st1", "sc1", "commandExecution", 2, 1100, 1100, 1200, {"type": "commandExecution", "command": "nft list ruleset", "exitCode": 0, "aggregatedOutput": "table inet filter {}"}),
                CodexItem("st1", "sa1", "agentMessage", 3, 1300, 1300, 1300, {"type": "agentMessage", "text": "Found missing MSS clamping.", "phase": "final_answer"}),
            ]
            summary = build_inline_subagent_summary(
                sub_thread,
                ThreadHistoryData(turns=[], items=sub_items, augmentation=RolloutAugmentation()),
            )
            self.assertEqual(summary.role, "netcheck")
            self.assertIn("Found missing MSS clamping.", summary.final_answer)
            self.assertIn("nft list ruleset", summary.markdown_transcript)

            parent_items = [
                CodexItem("t1", "u1", "userMessage", 1, 1000, 1000, 1000, {"type": "userMessage", "content": [{"type": "text", "text": "Check network"}]}),
                CodexItem(
                    "t1", "col1", "collabAgentToolCall", 2, 1005, 1005, 1350,
                    {"type": "collabAgentToolCall", "tool": "spawn_agent", "receiverThreadIds": [sub_id], "prompt": "Inspect nftables", "status": "completed"},
                ),
                CodexItem(
                    "t1", "col2", "collabAgentToolCall", 3, 1355, 1355, 1360,
                    {"type": "collabAgentToolCall", "tool": "wait", "receiverThreadIds": [sub_id], "agentsStates": {sub_id: {"status": "completed"}}, "status": "completed"},
                ),
                CodexItem("t1", "a1", "agentMessage", 4, 1400, 1400, 1400, {"type": "agentMessage", "text": "Done!", "phase": "final_answer"}),
            ]
            mapped = map_codex_thread(
                _make_thread(),
                ThreadHistoryData(turns=[], items=parent_items, augmentation=RolloutAugmentation()),
                project_id=OUTSIDE_OF_PROJECT_ID,
                project_name="",
                workspaces=[],
                agy_paths=agy_paths,
                codex_home=root,
                subagent_mode="inline",
                inline_subagents={sub_id: summary},
            )
            # Archive asset written under subagents/
            sub_assets = [a for a in mapped.assets if a.rel_path.startswith("subagents/")]
            self.assertEqual(len(sub_assets), 1)
            self.assertIn("Found missing MSS clamping.", sub_assets[0].data.decode("utf-8"))

            # Verify generic steps have NO InvokeSubagent / ManageSubagents payload_any (field 6) and NO AgentRef!
            g_steps = [s for s in mapped.steps if s.step_type == STEP_TYPE_GENERIC]
            self.assertEqual(len(g_steps), 2)
            for gs in g_steps:
                top = dict((f, v) for f, _wt, v in decode_fields(gs.payload_bytes))
                gen = dict((f, v) for f, _wt, v in decode_fields(top[140]))
                res_fields = dict((f, v) for f, _wt, v in decode_fields(gen[2]))
                self.assertNotIn(6, res_fields)  # No typed InvokeSubagent/ManageSubagents Any payload!
                self.assertIn(b"Found missing MSS clamping.", res_fields[3])

    def test_format_codex_assistant_directives(self) -> None:
        raw = (
            "::codex-realtime-inline{}\n"
            "Summary complete.\n\n"
            '::code-comment{title="[P1] Fix QUIC epoch" body="Handle {braces} and \\"quotes\\" safely." file="/work/src/flow.rs" start=749 end=754 priority=1} '
            '::code-comment{title="Check DNS" body="Require active transaction." file="/work/src/dns.rs" start=220 priority=2}\n\n'
            '::codex-file-citation{path="/work/decks/Q4 Plan.pptx" artifact_kind="presentation" slide_number="7"}\n\n'
            "Keep inline `::git-commit{cwd=\"/work\"}` and fenced code untouched:\n\n"
            "```text\n"
            '::code-comment{title="Literal example" body="keep me" file="/tmp/a.rs"}\n'
            "```\n\n"
            '::git-create-branch{cwd="/work" branch="feat"}\n'
            '::git-stage{cwd="/work"}\n'
            '::git-commit{cwd="/work"}\n'
        )
        out = format_codex_assistant_text(raw)
        # Pure UI action directives outside code stripped
        self.assertNotIn("::codex-realtime-inline", out)
        self.assertNotIn("::git-stage", out)
        self.assertNotIn("::git-create-branch", out)
        # Inline backtick and fenced code block preserved literally
        self.assertIn('`::git-commit{cwd="/work"}`', out)
        self.assertIn('::code-comment{title="Literal example"', out)
        # code-comment converted cleanly into separate blockquotes with clickable file:// links & no duplicate [P1]
        self.assertIn("> **💬 [P1] Fix QUIC epoch** — [flow.rs:749-754](file:///work/src/flow.rs#L749-L754)", out)
        self.assertNotIn("[P1] [P1]", out)
        self.assertIn('> Handle {braces} and "quotes" safely.', out)
        self.assertIn("> **💬 [P2] Check DNS** — [dns.rs:220](file:///work/src/dns.rs#L220)", out)
        # codex-file-citation converted into clickable link with URL-encoded space
        self.assertIn("- 📎 [Q4 Plan.pptx (Slide 7)](file:///work/decks/Q4%20Plan.pptx)", out)

    def test_trailing_reasoning_folded_and_deduplicated(self) -> None:
        """Late-flushed same-turn reasoning after final_answer folds above the answer with dedupe & no <!-- -->."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agy_paths = AgyPaths.custom(root / "agy")
            items = [
                CodexItem("turn-1", "u1", "userMessage", 1, 1000, 1000, 1000, {"type": "userMessage", "content": [{"type": "text", "text": "Review PR"}]}),
                CodexItem("turn-1", "r1", "reasoning", 2, 1100, 1100, 1200, {
                    "type": "reasoning",
                    "summary": [
                        "**Checking tests**\n\n<!-- -->",
                        "**Checking tests**\n\n<!-- -->",
                    ],
                }),
                CodexItem("turn-1", "a1", "agentMessage", 3, 1300, 1300, 1300, {"type": "agentMessage", "text": "Tests passed, reviewing edge cases.", "phase": "commentary"}),
                CodexItem("turn-1", "r2", "reasoning", 4, 1400, 1400, 1500, {
                    "type": "reasoning",
                    "summary": [
                        "**Planning final answer**\n\n<!-- -->",
                        "**Checking QUIC edge case**",
                    ],
                }),
                CodexItem("turn-1", "a2", "agentMessage", 5, 1600, 1600, 1600, {
                    "type": "agentMessage",
                    "text": 'All done.\n\n::git-commit{cwd="/work"}',
                    "phase": "final_answer",
                }),
                # Late-flushed duplicate/extended reasoning 200ms after final_answer in the SAME turn_id!
                CodexItem("turn-1", "r3", "reasoning", 6, 1800, 1800, 1800, {
                    "type": "reasoning",
                    "summary": [
                        "**Planning final answer**\n\n<!-- -->",
                        "**Validating inline review comments**",
                    ],
                }),
            ]
            mapped = map_codex_thread(
                _make_thread(),
                ThreadHistoryData(turns=[], items=items, augmentation=RolloutAugmentation()),
                project_id=OUTSIDE_OF_PROJECT_ID,
                project_name="",
                workspaces=[],
                agy_paths=agy_paths,
                codex_home=root,
            )
            planner_steps = [s for s in mapped.steps if s.step_type == STEP_TYPE_PLANNER_RESPONSE]
            # Must be exactly 2 PLANNER_RESPONSE steps (commentary + final_answer), NO trailing orphan thinking step!
            self.assertEqual(len(planner_steps), 2)

            def _extract_pr(step) -> tuple[str, str]:
                top = dict((f, v) for f, _wt, v in decode_fields(step.payload_bytes))
                body = dict((f, v) for f, _wt, v in decode_fields(top[20]))
                resp = body.get(1, b"").decode("utf-8") if isinstance(body.get(1), bytes) else ""
                think = body.get(3, b"").decode("utf-8") if isinstance(body.get(3), bytes) else ""
                return resp, think

            resp1, think1 = _extract_pr(planner_steps[0])
            resp2, think2 = _extract_pr(planner_steps[1])
            self.assertEqual(resp1, "Tests passed, reviewing edge cases.")
            self.assertEqual(think1, "**Checking tests**")  # Deduplicated & stripped <!-- -->
            self.assertEqual(resp2, "All done.")  # ::git-commit stripped
            self.assertNotIn("<!--", think2)
            # Folded r2 + r3 merged cleanly above final_answer with "**Planning final answer**" appearing once
            self.assertEqual(think2.count("**Planning final answer**"), 1)
            self.assertIn("**Checking QUIC edge case**", think2)
            self.assertIn("**Validating inline review comments**", think2)


if __name__ == "__main__":
    unittest.main()
