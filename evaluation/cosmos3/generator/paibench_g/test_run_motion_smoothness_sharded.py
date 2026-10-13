"""Regression tests for run_motion_smoothness_sharded.py evaluator identity.

Covers issue #361: resume must not mix scores from different evaluator
versions. These tests run without GPU, AMT-S, or the scorer checkout.
"""

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import run_motion_smoothness_sharded as runner


def make_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"], check=True
    )
    (path / "a.txt").write_text("hello\n")
    subprocess.run(["git", "-C", str(path), "add", "a.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "init"], check=True)


class HashHelpersTest(unittest.TestCase):
    def test_sha256_deterministic(self):
        self.assertEqual(runner._sha256(b"x"), hashlib.sha256(b"x").hexdigest())

    def test_hash_file_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(runner.hash_file(Path(tmp) / "nope.txt"))

    def test_hash_file_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.txt"
            p.write_text("abc\n")
            self.assertEqual(
                runner.hash_file(p), hashlib.sha256(b"abc\n").hexdigest()
            )

    def test_hash_patches_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            patches = Path(tmp) / "patches"
            patches.mkdir()
            (patches / "b.patch").write_text("B")
            (patches / "a.patch").write_text("A")
            result = runner.hash_patches(patches_dir=patches)
            # sorted by name: a.patch, b.patch
            expected = hashlib.sha256(b"A" + b"B").hexdigest()
            self.assertEqual(result, {"sha256": expected, "status": "ok"})

    def test_hash_patches_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.hash_patches(patches_dir=Path(tmp) / "nope")
            self.assertEqual(result["status"], "missing")
            self.assertIsNone(result["sha256"])

    def test_git_head_and_status_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "scorer"
            make_git_repo(repo)
            head = runner.git_head(repo)
            self.assertIsNotNone(head)
            self.assertEqual(len(head), 40)
            before = runner.git_status_hash(repo)
            self.assertIsNotNone(before)
            # uncommitted change alters status hash, not HEAD
            (repo / "a.txt").write_text("changed\n")
            self.assertEqual(runner.git_head(repo), head)
            after = runner.git_status_hash(repo)
            self.assertNotEqual(before, after)
