"""Regression tests for run_motion_smoothness_sharded.py evaluator identity.

Covers issue #361: resume must not mix scores from different evaluator
versions. These tests run without GPU, AMT-S, or the scorer checkout.
"""

import hashlib
import json
import shutil
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
            # sorted by name: a.patch, b.patch; digest includes name+len+content
            digest = hashlib.sha256()
            digest.update(b"a.patch")
            digest.update(b"1")
            digest.update(b"A")
            digest.update(b"b.patch")
            digest.update(b"1")
            digest.update(b"B")
            expected = digest.hexdigest()
            self.assertEqual(result, {"sha256": expected, "status": "ok"})

    def test_hash_patches_name_len_prevent_collision(self):
        # {a:"AB", b:"C"} and {a:"A", b:"BC"} must NOT hash the same
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "p1"
            p2 = Path(tmp) / "p2"
            p1.mkdir()
            p2.mkdir()
            (p1 / "a.patch").write_text("AB")
            (p1 / "b.patch").write_text("C")
            (p2 / "a.patch").write_text("A")
            (p2 / "b.patch").write_text("BC")
            self.assertNotEqual(
                runner.hash_patches(patches_dir=p1),
                runner.hash_patches(patches_dir=p2),
            )

    def test_hash_patches_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.hash_patches(patches_dir=Path(tmp) / "nope")
            self.assertEqual(result["status"], "missing")
            self.assertIsNone(result["sha256"])

    @unittest.skipUnless(shutil.which("git"), "git is required")
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


@unittest.skipUnless(shutil.which("git"), "git is required")
class ManifestTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.scorer = self.tmp / "scorer"
        make_git_repo(self.scorer)
        (self.scorer / "scorer_environment.freeze.txt").write_text("torch==2.5\n")
        # isolated patch dir copy for the test, not the real repo
        self.patches = self.tmp / "patches"
        self.patches.mkdir()
        (self.patches / "p.patch").write_text("patch-v1\n")
        self.videos = [self.tmp / "v0.mp4", self.tmp / "v1.mp4"]
        for v in self.videos:
            v.write_bytes(b"fake-mp4")
        # file_identity() stats config/checkpoint, so they must exist
        (self.tmp / "cfg.yaml").write_text("cfg\n")
        (self.tmp / "ckpt.pt").write_bytes(b"ckpt")

    def _manifest(self):
        return runner.build_manifest(
            task="t2v",
            model_name="model",
            videos=self.videos,
            evaluator_root=self.scorer,
            config=self.tmp / "cfg.yaml",
            checkpoint=self.tmp / "ckpt.pt",
            patches_dir=self.patches,
        )

    def test_manifest_has_evaluator_identity(self):
        manifest = self._manifest()
        self.assertEqual(manifest["runner_version"], 3)
        ident = manifest["evaluator_identity"]
        self.assertEqual(ident["scorer_source"]["status"], "ok")
        self.assertEqual(ident["scorer_patches"]["status"], "ok")
        self.assertEqual(ident["scorer_environment"]["status"], "ok")

    def test_prepare_manifest_resume_accepts_same_identity(self):
        out = self.tmp / "out"
        out.mkdir()
        result = self.tmp / "result.json"
        manifest = self._manifest()
        runner.prepare_manifest(out, result, manifest)
        # same identity, same videos -> no error (idempotent resume)
        runner.prepare_manifest(out, result, self._manifest())

    def test_prepare_manifest_rejects_changed_patch(self):
        out = self.tmp / "out"
        out.mkdir()
        result = self.tmp / "result.json"
        runner.prepare_manifest(out, result, self._manifest())
        # change a patch -> identity mismatch -> hard fail
        (self.patches / "p.patch").write_text("patch-v2\n")
        with self.assertRaisesRegex(RuntimeError, "evaluator identity mismatch"):
            runner.prepare_manifest(out, result, self._manifest())

    def test_prepare_manifest_rejects_changed_source(self):
        out = self.tmp / "out"
        out.mkdir()
        result = self.tmp / "result.json"
        runner.prepare_manifest(out, result, self._manifest())
        # new commit on the scorer -> identity mismatch
        (self.scorer / "a.txt").write_text("new\n")
        subprocess.run(["git", "-C", str(self.scorer), "add", "a.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(self.scorer), "commit", "-qm", "bump"], check=True
        )
        with self.assertRaisesRegex(RuntimeError, "evaluator identity mismatch"):
            runner.prepare_manifest(out, result, self._manifest())

    def test_prepare_manifest_rejects_legacy_manifest(self):
        out = self.tmp / "out"
        out.mkdir()
        result = self.tmp / "result.json"
        # a pre-#361 manifest has runner_version 2 and no evaluator_identity key
        legacy = {"runner_version": 2, "task": "t2v"}
        (out / "manifest.json").write_text(json.dumps(legacy))
        with self.assertRaisesRegex(RuntimeError, "legacy manifest without evaluator identity"):
            runner.prepare_manifest(out, result, self._manifest())



class DegradationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        # NOT a git repo, no freeze file
        self.scorer = self.tmp / "plain_scorer"
        self.scorer.mkdir()
        self.videos = [self.tmp / "v0.mp4"]
        self.videos[0].write_bytes(b"fake-mp4")
        # isolated patch dir copy for the test, not the real repo
        self.patches = self.tmp / "patches"
        self.patches.mkdir()
        # file_identity() stats config/checkpoint, so they must exist
        (self.tmp / "cfg.yaml").write_text("cfg\n")
        (self.tmp / "ckpt.pt").write_bytes(b"ckpt")

    def test_identity_unavailable_without_git(self):
        ident = runner.build_evaluator_identity(self.scorer)
        self.assertEqual(ident["scorer_source"]["status"], "unavailable")
        self.assertEqual(ident["scorer_environment"]["status"], "missing")

    def test_prepare_manifest_tolerates_unavailable_identity(self):
        out = self.tmp / "out"
        out.mkdir()
        result = self.tmp / "result.json"
        manifest = runner.build_manifest(
            task="t2v",
            model_name="model",
            videos=self.videos,
            evaluator_root=self.scorer,
            config=self.tmp / "cfg.yaml",
            checkpoint=self.tmp / "ckpt.pt",
            patches_dir=self.patches,
        )
        # must NOT raise: soft degradation keeps the old baseline behavior
        runner.prepare_manifest(out, result, manifest)
        self.assertTrue((out / "manifest.json").exists())
