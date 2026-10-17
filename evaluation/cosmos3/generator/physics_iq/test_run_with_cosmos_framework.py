# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Offline regression tests for the physics_iq notebook staging helpers.

The tests extract the real function definitions from
run_with_cosmos_framework.ipynb (no logic is duplicated here) and drive them
with fake ffmpeg/ffprobe pairs, so no GPU, network, or multimedia stack is
needed.

Run:  python3 test_run_with_cosmos_framework.py
"""

import json
import os
import subprocess
import types
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent / "run_with_cosmos_framework.ipynb"


def load_staging_namespace() -> dict:
    """Exec the notebook cell that defines the staging helpers and return its namespace."""
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if "def stage_i2v_last120(" in source:
            func_source = source[source.index("def _ffmpeg("):]
            namespace = {
                "Path": Path,
                "os": os,
                "FFMPEG": "/usr/bin/true",
                "shutil": types.SimpleNamespace(which=lambda _p: "/usr/bin/true"),
                "subprocess": subprocess,
            }
            exec(compile(func_source, str(NOTEBOOK), "exec"), namespace)
            return namespace
    raise AssertionError("staging helper cell not found in notebook")


class FakeMedia:
    """Stand-in for ffmpeg/ffprobe: the 'frame count' is embedded in the file bytes."""

    def __init__(self, namespace: dict):
        self.calls: list[list[str]] = []
        self._namespace = namespace
        namespace["_ffmpeg"] = self._ffmpeg
        namespace["_ffprobe_frames"] = self._ffprobe_frames

    @staticmethod
    def frames_from_args(args: list[str]) -> int:
        return int(args[args.index("-frames:v") + 1])

    def _ffmpeg(self, args: list[str]) -> None:
        self.calls.append([str(a) for a in args])
        out_path = Path(self.calls[-1][-1])
        out_path.write_bytes(f"FRAMES={self.frames_from_args(self.calls[-1])}".encode())

    @staticmethod
    def _ffprobe_frames(path) -> int:
        data = Path(path).read_bytes()
        return int(data.removeprefix(b"FRAMES=").strip())

    @staticmethod
    def write_source(path: Path, frames: int) -> Path:
        path.write_bytes(f"FRAMES={frames}".encode())
        return path


class StagingHelperTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.ns = load_staging_namespace()
        self.media = FakeMedia(self.ns)

    def last_output_arg(self) -> Path:
        return Path(self.media.calls[-1][-1])

    def test_stage_i2v_last120_writes_through_temp_file(self):
        src = self.media.write_source(self.tmp / "raw.mp4", 121)
        dst = self.tmp / "staged" / "case.mp4"
        self.ns["stage_i2v_last120"](src, dst)
        self.assertEqual(self.media._ffprobe_frames(dst), 120)
        # the producer must write to a temp path, never directly to the final one
        self.assertTrue(self.last_output_arg().name.startswith(".case.mp4."))
        self.assertTrue(self.last_output_arg().name.endswith(".tmp"))
        self.assertNotEqual(self.last_output_arg(), dst)

    def test_stage_i2v_last120_reuses_valid_artifact_without_rerun(self):
        src = self.media.write_source(self.tmp / "raw.mp4", 121)
        dst = self.tmp / "staged" / "case.mp4"
        self.ns["stage_i2v_last120"](src, dst)
        calls_after_first = len(self.media.calls)
        self.ns["stage_i2v_last120"](src, dst)
        self.assertEqual(len(self.media.calls), calls_after_first)
        self.assertEqual(self.media._ffprobe_frames(dst), 120)

    def test_stage_i2v_last120_restages_corrupted_artifact(self):
        src = self.media.write_source(self.tmp / "raw.mp4", 121)
        dst = self.tmp / "staged" / "case.mp4"
        self.ns["stage_i2v_last120"](src, dst)
        # simulate a run killed mid-encode: the artifact is short
        dst.write_bytes(b"FRAMES=100")
        self.ns["stage_i2v_last120"](src, dst)
        self.assertEqual(self.media._ffprobe_frames(dst), 120)

    def test_stage_v2v_omit73_frames_and_reuse(self):
        src = self.media.write_source(self.tmp / "raw.mp4", 193)
        dst = self.tmp / "staged" / "case.mp4"
        self.ns["stage_v2v_omit73"](src, dst)
        self.assertEqual(self.media._ffprobe_frames(dst), 120)
        calls_after_first = len(self.media.calls)
        self.ns["stage_v2v_omit73"](src, dst)
        self.assertEqual(len(self.media.calls), calls_after_first)

    def test_stage_v2v_omit73_restages_corrupted_artifact(self):
        src = self.media.write_source(self.tmp / "raw.mp4", 193)
        dst = self.tmp / "staged" / "case.mp4"
        self.ns["stage_v2v_omit73"](src, dst)
        dst.write_bytes(b"FRAMES=64")
        self.ns["stage_v2v_omit73"](src, dst)
        self.assertEqual(self.media._ffprobe_frames(dst), 120)

    def test_prepare_conditioning_pads_72_to_73_and_reuses(self):
        src = self.media.write_source(self.tmp / "cond72.mp4", 72)
        dst = self.tmp / "cond73.mp4"
        self.ns["prepare_v2v_conditioning"](src, dst)
        self.assertEqual(self.media._ffprobe_frames(dst), 73)
        calls_after_first = len(self.media.calls)
        self.ns["prepare_v2v_conditioning"](src, dst)
        self.assertEqual(len(self.media.calls), calls_after_first)

    def test_prepare_conditioning_restages_corrupted_artifact(self):
        src = self.media.write_source(self.tmp / "cond72.mp4", 72)
        dst = self.tmp / "cond73.mp4"
        self.ns["prepare_v2v_conditioning"](src, dst)
        dst.write_bytes(b"FRAMES=59")
        self.ns["prepare_v2v_conditioning"](src, dst)
        self.assertEqual(self.media._ffprobe_frames(dst), 73)

    def test_prepare_conditioning_rejects_unexpected_source(self):
        src = self.media.write_source(self.tmp / "cond71.mp4", 71)
        with self.assertRaises(AssertionError):
            self.ns["prepare_v2v_conditioning"](src, self.tmp / "cond73.mp4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
