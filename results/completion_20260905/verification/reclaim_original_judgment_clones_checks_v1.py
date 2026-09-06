"""Small real-filesystem checks for the one-off original-judgment clone tool."""
import fcntl
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("reclaim_original_judgment_clones_v1.py")

class CloneChecks(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), "the authorized clone implementation is not present yet")
        spec = importlib.util.spec_from_file_location("reclaim_original_clones", SCRIPT)
        self.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.tool)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.original = "results/completion_20260905/judgments_v3/fixture/vendor_M.jsonl.partial"
        self.frozen = "artifacts/completion_20260905/judgment_evidence/fixture/evidence.jsonl"
        self.planned = self.original.removesuffix(".partial")
        self.lock = self.planned + ".lock"
        self.raw = b"synthetic judgment bytes\n"
        for name, raw in [(self.original, self.raw), (self.frozen, self.raw), (self.lock, b"")]:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        (self.root / self.original).chmod(0o640)
        self.job = {"job_id": "fixture", "planned_output": self.planned, "lock": self.lock,
                    "terminal_status": "pending", "candidates": [{"role": "evidence", "original": self.original,
                    "frozen": self.frozen, "sha256": hashlib.sha256(self.raw).hexdigest(), "size_bytes": len(self.raw)}]}

    def test_clone_preserves_bytes_mode_and_independent_writes(self):
        old_inode = (self.root / self.original).stat().st_ino
        rows = self.tool.clone_job(self.root, self.job, set())
        original, frozen = self.root / self.original, self.root / self.frozen
        self.assertEqual(original.read_bytes(), self.raw)
        self.assertEqual(original.stat().st_mode & 0o7777, 0o640)
        self.assertNotEqual(original.stat().st_ino, old_inode)
        self.assertNotEqual(original.stat().st_ino, frozen.stat().st_ino)
        self.assertEqual(original.stat().st_nlink, 1)
        self.assertEqual(rows[0]["verified_sha256"], hashlib.sha256(self.raw).hexdigest())
        original.write_bytes(b"changed original\n")
        self.assertEqual(frozen.read_bytes(), self.raw)
        frozen.write_bytes(b"changed frozen fixture\n")
        self.assertEqual(original.read_bytes(), b"changed original\n")

    def test_conflicting_original_fails_without_replacement(self):
        path = self.root / self.original
        path.write_bytes(b"local conflict\n")
        before = path.stat().st_ino
        with self.assertRaisesRegex(ValueError, "hash|size"):
            self.tool.clone_job(self.root, self.job, set())
        self.assertEqual(path.read_bytes(), b"local conflict\n")
        self.assertEqual(path.stat().st_ino, before)

    def test_active_lock_fails_without_replacement(self):
        path = self.root / self.original
        before = path.stat().st_ino
        with (self.root / self.lock).open("r+") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "active"):
                self.tool.clone_job(self.root, self.job, set())
        self.assertEqual(path.stat().st_ino, before)
        self.assertEqual(path.read_bytes(), self.raw)

    def test_selected_destination_fails_without_replacement(self):
        path = self.root / self.original
        before = path.stat().st_ino
        with self.assertRaisesRegex(ValueError, "selected"):
            self.tool.clone_job(self.root, self.job, {self.original})
        self.assertEqual(path.stat().st_ino, before)

    def test_symlink_fails_without_replacement(self):
        path = self.root / self.original
        path.unlink()
        path.symlink_to(self.root / self.frozen)
        with self.assertRaises((ValueError, OSError)):
            self.tool.clone_job(self.root, self.job, set())
        self.assertTrue(path.is_symlink())
        self.assertEqual((self.root / self.frozen).read_bytes(), self.raw)

    def test_nonterminal_job_fails_without_replacement(self):
        self.job["terminal_status"] = "running"
        with self.assertRaisesRegex(ValueError, "terminal"):
            self.tool.clone_job(self.root, self.job, set())
        self.assertEqual((self.root / self.original).read_bytes(), self.raw)

if __name__ == "__main__":
    unittest.main()
