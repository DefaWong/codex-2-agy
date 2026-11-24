"""Tests for protobuf_wire and agy_proto encoders."""

from __future__ import annotations

import shutil
import subprocess
import unittest

from codex_2_agy.agy_proto import (
    DIFF_LINE_DELETE,
    DIFF_LINE_INSERT,
    DIFF_LINE_UNCHANGED,
    FILE_EDIT_KIND_EDIT,
    STEP_SOURCE_MODEL,
    STEP_SOURCE_USER_EXPLICIT,
    STEP_STATUS_DONE,
    STEP_TYPE_GENERIC,
    STEP_TYPE_PLANNER_RESPONSE,
    STEP_TYPE_USER_INPUT,
    AskQuestionEntryData,
    MediaAttachment,
    ToolCallSpec,
    UsageStats,
    WorkspaceInfo,
    decode_agyhub_summaries,
    encode_agyhub_summaries,
    encode_annotations_pbtxt,
    encode_cascade_trajectory_summary,
    encode_completed_ask_question_interaction,
    encode_content_block_terminal,
    encode_conversation_annotations,
    encode_cortex_step,
    encode_generic_step,
    encode_planner_response,
    encode_step_metadata,
    encode_step_render_info,
    encode_title_part_text,
    encode_tool_ask_question,
    encode_tool_file_edit,
    encode_tool_generate_image,
    encode_tool_invoke_subagent,
    encode_tool_mcp,
    encode_tool_run_command,
    encode_tool_search_web,
    encode_trajectory_metadata,
    encode_user_input,
)
from codex_2_agy.protobuf_wire import (
    decode_fields,
    encode_any,
    encode_duration_ms,
    encode_str_map_entries,
    encode_timestamp_ms,
    encode_varint,
    f_bool,
    f_bytes,
    f_msg,
    f_str,
    f_varint,
    join_fields,
)


class TestProtobufWire(unittest.TestCase):
    def test_varint_and_negative_int32(self) -> None:
        self.assertEqual(encode_varint(0), b"\x00")
        self.assertEqual(encode_varint(150), b"\x96\x01")
        # Negative varint (two's complement 64-bit)
        neg = encode_varint(-1)
        self.assertEqual(len(neg), 10)
        fields = decode_fields(f_varint(1, -1))
        self.assertEqual(fields[0][0], 1)
        self.assertEqual(fields[0][2], (1 << 64) - 1)

    def test_timestamp_and_duration(self) -> None:
        ts = encode_timestamp_ms(1_760_000_000_250)
        f = dict((num, val) for num, _wt, val in decode_fields(ts))
        self.assertEqual(f[1], 1_760_000_000)
        self.assertEqual(f[2], 250_000_000)

        dur = encode_duration_ms(12_345)
        df = dict((num, val) for num, _wt, val in decode_fields(dur))
        self.assertEqual(df[1], 12)
        self.assertEqual(df[2], 345_000_000)

    def test_any_and_map_entries(self) -> None:
        inner = f_str(1, "hello")
        any_b = encode_any("type.googleapis.com/test.Msg", inner)
        af = dict((num, val) for num, _wt, val in decode_fields(any_b))
        self.assertEqual(af[1], b"type.googleapis.com/test.Msg")
        self.assertEqual(af[2], inner)

        m = encode_str_map_entries(1, [("k1", "v1"), ("k2", "中文")])
        entries = decode_fields(m)
        self.assertEqual(len(entries), 2)
        decoded_map = {}
        for fnum, _wt, val in entries:
            self.assertEqual(fnum, 1)
            kv = dict((sf, sv.decode("utf-8")) for sf, _swt, sv in decode_fields(val))
            decoded_map[kv[1]] = kv[2]
        self.assertEqual(decoded_map, {"k1": "v1", "k2": "中文"})

    def test_agyhub_summaries_roundtrip(self) -> None:
        ws = [
            WorkspaceInfo(
                folder_uri="file:///Users/test/repo",
                git_root_uri="file:///Users/test/repo",
                repo_computed_name="org/repo",
                git_origin_url="git@github.com:org/repo.git",
                branch_name="main",
            )
        ]
        tm = encode_trajectory_metadata(
            workspaces=ws,
            created_at_ms=1_760_000_000_000,
            initialization_state_id="init-1",
            root_conversation_id="cid-1",
            project_id="proj-1",
        )
        ann = encode_conversation_annotations(title="测试标题", last_user_view_time_ms=1_760_000_001_000)
        summary = encode_cascade_trajectory_summary(
            summary="测试标题",
            step_count=5,
            last_modified_ms=1_760_000_001_000,
            trajectory_id="traj-1",
            created_ms=1_760_000_000_000,
            workspaces=ws,
            last_user_input_ms=1_760_000_000_500,
            last_user_input_step_index=3,
            annotations_bytes=ann,
            trajectory_metadata_bytes=tm,
            is_subagent=False,
        )
        hub_bytes = encode_agyhub_summaries({"cid-1": summary, "cid-2": summary})
        decoded = decode_agyhub_summaries(hub_bytes)
        self.assertEqual(decoded["cid-1"], summary)
        self.assertEqual(decoded["cid-2"], summary)

        # Verify subagent summary uses termination reason 4 (NO_TOOL_CALL / completed), NOT 2 (USER_CANCELED)
        sub_summary = encode_cascade_trajectory_summary(
            summary="",
            step_count=3,
            last_modified_ms=1_760_000_001_000,
            trajectory_id="traj-sub",
            created_ms=1_760_000_000_000,
            workspaces=ws,
            last_user_input_ms=0,
            last_user_input_step_index=-1,
            annotations_bytes=b"",
            trajectory_metadata_bytes=tm,
            is_subagent=True,
        )
        sub_fields = dict((f, v) for f, _wt, v in decode_fields(sub_summary))
        self.assertEqual(sub_fields[28], 4)

    def test_protoc_decode_raw_compatibility(self) -> None:
        """Cross-check complex encoded messages with real `protoc --decode_raw` if available."""
        protoc = shutil.which("protoc")
        if not protoc:
            self.skipTest("protoc not installed")

        meta = encode_step_metadata(
            created_at_ms=1_760_000_000_000,
            started_at_ms=1_760_000_000_100,
            completed_at_ms=1_760_000_001_000,
            source=STEP_SOURCE_MODEL,
            execution_id="exec-1",
            trajectory_id="traj-1",
            step_index=1,
            cascade_id="cid-1",
            content_id="abcdef123456",
            usage=UsageStats(input_tokens=100, output_tokens=50, cache_read_tokens=40, thinking_output_tokens=20),
        )
        user_step = encode_cortex_step(
            step_type=STEP_TYPE_USER_INPUT,
            status=STEP_STATUS_DONE,
            metadata_bytes=meta,
            body_field_number=19,
            body_bytes=encode_user_input(
                "你好",
                media=[MediaAttachment(mime_type="image/png", uri="file:///tmp/a.png", display_name="a.png")],
            ),
        )
        planner_step = encode_cortex_step(
            step_type=STEP_TYPE_PLANNER_RESPONSE,
            status=STEP_STATUS_DONE,
            metadata_bytes=meta,
            body_field_number=20,
            body_bytes=encode_planner_response(
                response="回答内容",
                thinking="思考过程",
                message_id="m1",
                tool_calls=[ToolCallSpec(id="tc-1", name="run_command", arguments_json='{"CommandLine":"ls"}')],
                thinking_duration_ms=500,
            ),
        )
        generic_steps = [
            encode_cortex_step(
                step_type=STEP_TYPE_GENERIC,
                status=STEP_STATUS_DONE,
                metadata_bytes=meta,
                body_field_number=140,
                body_bytes=encode_generic_step(
                    args={"CommandLine": "ls"},
                    result_text="ok",
                    render_info=encode_step_render_info(
                        title_prefix="Ran",
                        icon="terminal",
                        title_parts=[encode_title_part_text("ls")],
                        content_blocks=[encode_content_block_terminal("ls", "/tmp", 0, "a\nb")],
                    ),
                    payload_any=encode_tool_run_command("ls", "/tmp", 0),
                ),
            ),
            encode_tool_file_edit(
                "/tmp/f.py",
                kind=FILE_EDIT_KIND_EDIT,
                diff_lines=[("old", DIFF_LINE_DELETE), ("new", DIFF_LINE_INSERT), ("ctx", DIFF_LINE_UNCHANGED)],
                additions=1,
                deletions=1,
            ),
            encode_tool_mcp("srv", ToolCallSpec("c1", "t1", "{}"), "res"),
            encode_tool_search_web("query", [("https://example.com", "Ex", "snip")], "sum"),
            encode_tool_generate_image("cat", ["/tmp/cat.png"], "file:///tmp/cat.png", 123, "cat.png"),
            encode_tool_invoke_subagent(
                type_name="self", role="Worker", initial_prompt="go", conversation_id="sub-1"
            ),
            encode_cortex_step(
                step_type=STEP_TYPE_GENERIC,
                status=STEP_STATUS_DONE,
                metadata_bytes=meta,
                body_field_number=140,
                body_bytes=encode_generic_step(
                    args={"questions": "[]"},
                    result_text="A1: yes",
                    payload_any=encode_tool_ask_question(
                        [AskQuestionEntryData(question="Ok?", options=[("1", "yes")], selected_option_ids=["1"])]
                    ),
                ),
                completed_interaction_bytes=encode_completed_ask_question_interaction(
                    [AskQuestionEntryData(question="Ok?", options=[("1", "yes")], selected_option_ids=["1"])],
                    trajectory_id="traj-1",
                    step_index=2,
                ),
            ),
        ]

        for idx, blob in enumerate([user_step, planner_step] + generic_steps):
            proc = subprocess.run(
                [protoc, "--decode_raw"],
                input=blob,
                capture_output=True,
                timeout=5,
            )
            self.assertEqual(proc.returncode, 0, f"protoc failed on blob #{idx}: {proc.stderr.decode()}")
            self.assertTrue(len(proc.stdout) > 0)

    def test_annotations_pbtxt_escaping(self) -> None:
        txt = encode_annotations_pbtxt(title='Fix "quote" \\ slash\nline2', last_user_view_time_ms=1_760_000_000_500, pinned=True)
        self.assertIn('title:"Fix \\"quote\\" \\\\ slash\\nline2"', txt)
        self.assertIn("seconds:1760000000 nanos:500000000", txt)
        self.assertIn("pinned:true", txt)


if __name__ == "__main__":
    unittest.main()
