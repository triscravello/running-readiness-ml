"""Synthetic tests for tests/frozen_isolated.py (L8-3DI).

Every file used here is created by the test in a temporary directory. No project dataset, report,
inventory or model is opened. This module does not import the frozen-file guard.

Access monitor: wraps ``os.open`` (comparing the target's device/inode, honouring ``dir_fd`` and
``O_NOFOLLOW``) and uses an audit hook for ``open`` with a path (which covers builtin ``open``,
``io.open`` and ``pathlib``) and for process creation. It raises BEFORE the real open.
It does NOT cover native code, reads through descriptors that were already open (or mmaps of them),
kernel copy calls, or a reference to the original ``os.open`` saved before patching.
"""

from __future__ import annotations

import builtins
import contextlib
import copy
import errno
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tests import frozen_isolated as fi

SECRET = b"CANARY-SECRET-CONTENT\n"
TRAIN = "data/synthetic/CANARY-train.csv"
HOLDOUT = "data/synthetic/CANARY-holdout.csv"
PENDING_REPORT = "reports/CANARY-pending.json"
MODEL = "artifacts/model.bin"
MANIFEST = "manifest.json"
MODEL_BYTES = bytes(range(256)) * 700  # larger than one read chunk
MANIFEST_BYTES = b'{"synthetic": true}\n'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------- access monitor
class ExcludedAccess(BaseException):
    """BaseException, so that no ``except Exception`` in the code under test can swallow it."""


_ACTIVE: list = []
_PROCESS_EVENTS = ("subprocess.Popen", "os.system", "os.posix_spawn", "os.exec", "os.fork", "os.forkpty",
                   "os.spawn")


def _identity(status):
    return status.st_dev, status.st_ino


def _audit(event, args):
    if not _ACTIVE:
        return
    monitor = _ACTIVE[-1]
    if monitor.inside_wrapper:
        return
    if event in _PROCESS_EVENTS:
        monitor.reject(event)
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        monitor.check(args[0], None, True)


sys.addaudithook(_audit)


class AccessMonitor:
    def __init__(self, paths):
        self.protected = {_identity(os.stat(p)) for p in paths if os.path.lexists(p) and not os.path.islink(p)}
        self.attempts: list = []
        self.inside_wrapper = False

    def reject(self, what):
        self.attempts.append(str(what))
        raise ExcludedAccess(str(what))

    def check(self, path, dir_fd, follow):
        previous, self.inside_wrapper = self.inside_wrapper, True
        try:
            status = os.stat(path, dir_fd=dir_fd, follow_symlinks=follow)
        except (OSError, ValueError):
            status = None
        finally:
            self.inside_wrapper = previous
        if status is not None and _identity(status) in self.protected:
            self.reject(path)

    def __enter__(self):
        real = os.open

        def guarded(path, flags, mode=0o777, *, dir_fd=None):
            self.check(path, dir_fd, not flags & os.O_NOFOLLOW)
            self.inside_wrapper = True
            try:
                return real(path, flags, mode, dir_fd=dir_fd)
            finally:
                self.inside_wrapper = False

        self.patch = mock.patch.object(os, "open", guarded)
        self.patch.start()
        _ACTIVE.append(self)
        return self

    def __exit__(self, *exc):
        _ACTIVE.pop()
        self.patch.stop()
        return False


# ------------------------------------------------------------------------ synthetic tree
class Tree:
    def __init__(self, case: unittest.TestCase):
        self.base = os.path.realpath(tempfile.mkdtemp(prefix="frozen-isolated-"))
        case.addCleanup(shutil.rmtree, self.base, True)
        self.root = os.path.join(self.base, "stage")
        os.mkdir(self.root)
        self.write(MODEL, MODEL_BYTES)
        self.write(MANIFEST, MANIFEST_BYTES)
        for name in (TRAIN, HOLDOUT, PENDING_REPORT):
            self.write(name, SECRET)
        self.inventory = {
            "title": "SYNTHETIC inventory", "inventory_version": "1.0.0",
            "excluded_holdout": {"path": HOLDOUT, "hashed_here": False},
            "files": {"tracked": {MODEL: sha(MODEL_BYTES), MANIFEST: sha(MANIFEST_BYTES)},
                      "local_only": {TRAIN: sha(SECRET), PENDING_REPORT: sha(SECRET)}},
        }
        self.entries = [
            self.row(MODEL, fi.APPROVED), self.row(MANIFEST, fi.APPROVED),
            self.row(TRAIN, fi.EXCLUDED_DATASET), self.row(PENDING_REPORT, fi.PENDING),
        ]
        self.outside = [self.row(HOLDOUT, fi.EXCLUDED_DATASET)]
        self.document_overrides: dict = {}

    @staticmethod
    def row(path, kind, reason="synthetic reason"):
        return {"path": path, "class": kind, "reason": reason}

    def path(self, name):
        return os.path.join(self.root, name)

    def write(self, name, data):
        target = self.path(name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as handle:
            handle.write(data)

    def canaries(self):
        return [self.path(n) for n in (TRAIN, HOLDOUT, PENDING_REPORT)]

    def all_artifacts(self):
        return self.canaries() + [self.path(MODEL), self.path(MANIFEST)]

    def classification(self, inventory_pin):
        document = {"classification_version": "1", "review_status": "REVIEWED", "inventory_sha256": inventory_pin,
                    "entries": self.entries, "excluded_outside_inventory": self.outside}
        document.update(self.document_overrides)
        return document

    def arguments(self, inventory_text=None, classification_text=None):
        inventory = (inventory_text if inventory_text is not None else json.dumps(self.inventory)).encode()
        self.write("inventory.json", inventory)
        classification = (classification_text if classification_text is not None
                          else json.dumps(self.classification(sha(inventory)))).encode()
        self.write("classification.json", classification)
        return [self.root, "inventory.json", sha(inventory), "classification.json", sha(classification)]


class Base(unittest.TestCase):
    def setUp(self):
        self.tree = Tree(self)

    def run_verify(self, arguments=None, protect=None):
        arguments = arguments or self.tree.arguments()
        with AccessMonitor(self.tree.canaries() if protect is None else protect) as monitor:
            report = fi.verify(*arguments)
        self.assertEqual(monitor.attempts, [])
        self.assert_invariant_wording(report)
        return report

    def assert_invariant_wording(self, report):
        self.assertEqual(report["dataset_content_integrity"], "NOT VERIFIED")
        self.assertIs(report["equivalent_to_full_guard"], False)
        for key in ("historical_holdout_cross_checks", "historical_repository_addition_checks",
                    "staging_exclusivity_check", "metadata_checks"):
            self.assertEqual(report[key], "NOT PERFORMED")

    def assert_contract_failure(self, reason, arguments=None):
        """Fails with the given reason before ANY artifact (approved or excluded) is opened."""
        arguments = arguments or self.tree.arguments()
        report = self.run_verify(arguments, protect=self.tree.all_artifacts())
        self.assertEqual((report["status"], report["failure_reasons"], report["results"]), ("FAIL", [reason], []))

    def outcomes(self, report):
        return {r["path"]: r["outcome"] for r in report["results"]}

    def assert_artifact_outcome(self, outcome, report=None, path=MODEL):
        report = report or self.run_verify()
        self.assertEqual(self.outcomes(report)[path], outcome, report)
        self.assertEqual((report["status"], report["failure_reasons"]), ("FAIL", ["mandatory_artifact_not_verified"]))


# ------------------------------------------------------------------- section 9, items 1-14
class ContentTests(Base):
    def test_01_correct_approved_artifacts(self):
        report = self.run_verify()
        self.assertEqual((report["status"], report["failure_reasons"]), ("PASS", []))
        self.assertEqual(self.outcomes(report), {
            MODEL: "VERIFIED", MANIFEST: "VERIFIED", TRAIN: "EXCLUDED_DATASET",
            PENDING_REPORT: "EXCLUDED_PENDING_CLASSIFICATION"})
        self.assertEqual(report["counts"], {"VERIFIED": 2, "EXCLUDED_DATASET": 1,
                                            "EXCLUDED_PENDING_CLASSIFICATION": 1})

    def test_02_incorrect_artifact_hash(self):
        self.tree.write(MODEL, MODEL_BYTES + b"x")
        self.assert_artifact_outcome("MISMATCH")

    def test_03_missing_mandatory_artifact(self):
        os.unlink(self.tree.path(MODEL))
        self.assert_artifact_outcome("MISSING_MANDATORY")

    def test_03b_missing_parent_directory(self):
        shutil.rmtree(self.tree.path("artifacts"))
        self.assert_artifact_outcome("MISSING_MANDATORY")

    def test_04_malformed_expected_hash(self):
        for bad in ("A" * 64, "a" * 63, "a" * 65, "g" * 64, "", None, 7):
            with self.subTest(bad=bad):
                self.tree.inventory["files"]["tracked"][MODEL] = bad
                self.assert_contract_failure("malformed_expected_hash")

    def test_05_unexpected_inventory_entry(self):
        self.tree.inventory["files"]["tracked"]["artifacts/extra.bin"] = "0" * 64
        self.assert_contract_failure("missing_classification")

    def test_05b_classification_for_an_entry_not_in_the_inventory(self):
        self.tree.entries.append(self.tree.row("artifacts/extra.bin", fi.APPROVED))
        self.assert_contract_failure("classification_for_unknown_entry")

    def test_06_duplicate_json_keys(self):
        text = json.dumps(self.tree.inventory)
        duplicated = text.replace('"tracked": {', '"tracked": {"%s": "%s", ' % (MODEL, "0" * 64), 1)
        self.assert_contract_failure("duplicate_json_key", self.tree.arguments(inventory_text=duplicated))
        self.assert_contract_failure("duplicate_json_key", self.tree.arguments(
            inventory_text='{"inventory_version":"1.0.0",' + text[1:]))

    def test_06b_entry_in_both_inventory_groups(self):
        self.tree.inventory["files"]["local_only"][MODEL] = sha(MODEL_BYTES)
        self.assert_contract_failure("inventory_duplicate_entry")

    def test_06c_duplicate_key_in_classification(self):
        pin = self.tree.arguments()[2]
        text = json.dumps(self.tree.classification(pin)).replace('"class": ', '"class": "x", "class": ', 1)
        self.assert_contract_failure("duplicate_json_key", self.tree.arguments(classification_text=text))

    def test_07_duplicate_classification(self):
        self.tree.entries.append(self.tree.row(MODEL, fi.APPROVED))
        self.assert_contract_failure("duplicate_classification")

    def test_08_missing_classification(self):
        del self.tree.entries[0]
        self.assert_contract_failure("missing_classification")

    def test_09_contradictory_classification(self):
        self.tree.entries.append(self.tree.row(TRAIN, fi.APPROVED))
        self.assert_contract_failure("contradictory_classification")
        self.tree.entries.pop()
        self.tree.outside.append(self.tree.row(PENDING_REPORT, fi.PENDING))
        self.assert_contract_failure("contradictory_classification")

    def test_10_unknown_classification(self):
        for bad in ("APPROVED", "excluded_dataset", "", None):
            with self.subTest(bad=bad):
                self.tree.entries[0] = self.tree.row(MODEL, bad)
                self.assert_contract_failure("unknown_classification")

    def test_11_excluded_dataset_present_is_never_opened(self):
        self.assertTrue(os.path.exists(self.tree.path(TRAIN)))
        report = self.run_verify()
        self.assertEqual(report["status"], "PASS")
        self.assertNotIn(SECRET.decode().strip(), json.dumps(report))

    def test_11b_excluded_paths_reach_no_filesystem_call(self):
        arguments = self.tree.arguments()
        seen = []

        def recorder(real):
            def call(path, *args, **kwargs):
                seen.append(os.fsdecode(path) if isinstance(path, (str, bytes)) else path)
                return real(path, *args, **kwargs)
            return call

        with mock.patch.object(os, "stat", recorder(os.stat)), mock.patch.object(os, "open", recorder(os.open)):
            self.assertEqual(fi.verify(*arguments)["status"], "PASS")
        names = {str(p) for p in seen}
        self.assertFalse([n for n in names if "CANARY" in n or n in ("data", "synthetic", "reports")], names)

    def test_12_excluded_files_absent(self):
        shutil.rmtree(self.tree.path("data"))
        shutil.rmtree(self.tree.path("reports"))
        self.assertEqual(self.run_verify(protect=[])["status"], "PASS")

    def test_13_pending_classification_report(self):
        report = self.run_verify()
        self.assertEqual(self.outcomes(report)[PENDING_REPORT], "EXCLUDED_PENDING_CLASSIFICATION")
        listed = {e["path"]: (e["classification"], e["reason"]) for e in report["excluded_paths"]}
        self.assertEqual(listed, {
            TRAIN: ("EXCLUDED_DATASET", "synthetic reason"), HOLDOUT: ("EXCLUDED_DATASET", "synthetic reason"),
            PENDING_REPORT: ("EXCLUDED_PENDING_CLASSIFICATION", "synthetic reason")})

    def test_14_metadata_helper_is_separate_and_opens_no_final_file(self):
        arguments = self.tree.arguments()
        with mock.patch.object(fi, "describe_metadata", side_effect=AssertionError("must not run")):
            self.assertEqual(fi.verify(*arguments)["metadata_checks"], "NOT PERFORMED")
        with AccessMonitor(self.tree.canaries()) as monitor:
            found = fi.describe_metadata(self.tree.root, TRAIN)
            absent = fi.describe_metadata(self.tree.root, "data/synthetic/none.csv")
        self.assertEqual(monitor.attempts, [])
        self.assertEqual(found, {"path": TRAIN, "present": True, "type": "regular", "size": len(SECRET),
                                 "link_count": 1, "content_integrity": "NOT VERIFIED"})
        self.assertEqual((absent["present"], absent["content_integrity"]), (False, "NOT VERIFIED"))
        os.symlink(self.tree.path(TRAIN), self.tree.path("data/synthetic/link.csv"))
        with AccessMonitor(self.tree.canaries()) as monitor:
            self.assertEqual(fi.describe_metadata(self.tree.root, "data/synthetic/link.csv")["type"], "symlink")
        self.assertEqual(monitor.attempts, [])


# ------------------------------------------------------------------------ classification rules
class ClassificationRuleTests(Base):
    def test_permission_comes_only_from_an_approved_row(self):
        self.tree.entries[0] = self.tree.row(MODEL, fi.PENDING)  # tracked, harmless name: still not read
        report = self.run_verify(protect=self.tree.canaries() + [self.tree.path(MODEL)])
        self.assertEqual((report["status"], self.outcomes(report)[MODEL]),
                         ("PASS", "EXCLUDED_PENDING_CLASSIFICATION"))

    def test_approved_path_under_data_is_rejected(self):
        self.tree.entries[2] = self.tree.row(TRAIN, fi.APPROVED)
        self.assert_contract_failure("approved_path_under_data")

    def test_proposed_classification_cannot_be_executed(self):
        for status in ("PROPOSED_NOT_APPROVED", "", None):
            self.tree.document_overrides = {"review_status": status}
            self.assert_contract_failure("classification_not_reviewed")

    def test_pins(self):
        arguments = self.tree.arguments()
        for index, bad, reason in ((2, "0" * 64, "control_document_pin_mismatch"),
                                   (4, "0" * 64, "control_document_pin_mismatch"),
                                   (2, "xyz", "control_document_pin_malformed")):
            changed = list(arguments)
            changed[index] = bad
            self.assert_contract_failure(reason, changed)
        self.tree.document_overrides = {"inventory_sha256": "0" * 64}
        self.assert_contract_failure("classification_for_other_inventory")

    def test_control_document_arguments_cannot_name_other_files(self):
        arguments = self.tree.arguments()
        for index in (1, 3):
            for name in (TRAIN, PENDING_REPORT, "../inventory.json", "/etc/hosts", "inventory.txt", "", None):
                with self.subTest(index=index, name=name):
                    changed = list(arguments)
                    changed[index] = name
                    self.assert_contract_failure("control_document_path_not_allowed", changed)

    def test_control_document_that_is_a_link_is_not_read(self):
        arguments = self.tree.arguments()
        os.unlink(self.tree.path("classification.json"))
        os.symlink(self.tree.path(PENDING_REPORT), self.tree.path("classification.json"))
        self.assert_contract_failure("control_document_unreadable", arguments)

    def test_holdout_must_be_explicitly_excluded_and_outside_the_inventory(self):
        self.tree.outside = []
        self.assert_contract_failure("holdout_not_explicitly_excluded")
        self.tree.outside = [self.tree.row(HOLDOUT, fi.PENDING)]
        self.assert_contract_failure("holdout_not_explicitly_excluded")
        self.tree.outside = [self.tree.row(HOLDOUT, fi.APPROVED)]
        self.assert_contract_failure("outside_inventory_must_be_excluded")
        self.tree.inventory["files"]["tracked"][HOLDOUT] = "0" * 64
        self.assert_contract_failure("holdout_in_inventory")

    def test_structure_validation(self):
        original = copy.deepcopy(self.tree.inventory)
        for change in (lambda i: i.pop("files"), lambda i: i.update(inventory_version="2"),
                       lambda i: i["files"].pop("local_only"), lambda i: i["files"].update(other={}),
                       lambda i: i["files"].update(tracked=[]), lambda i: i["excluded_holdout"].update(hashed_here=True)):
            self.tree.inventory = copy.deepcopy(original)
            change(self.tree.inventory)
            self.assert_contract_failure("inventory_structure")
        self.tree.inventory = original
        self.assert_contract_failure("malformed_json", self.tree.arguments(inventory_text="[1]"))
        self.assert_contract_failure("malformed_json", self.tree.arguments(inventory_text='{"a": NaN}'))
        self.assert_contract_failure("malformed_json", self.tree.arguments(inventory_text="{"))
        self.tree.entries[0] = {"path": MODEL, "class": fi.APPROVED}
        self.assert_contract_failure("classification_structure")
        self.tree.entries[0] = self.tree.row(MODEL, fi.APPROVED, reason=" ")
        self.assert_contract_failure("classification_structure")


# ------------------------------------------------------------------ section 9, items 15-22
class PathSafetyTests(Base):
    def rejected_spelling(self, spelling):
        original = copy.deepcopy((self.tree.inventory, self.tree.entries))
        tracked = self.tree.inventory["files"]["tracked"]
        tracked[spelling] = tracked.pop(MODEL)
        self.tree.entries[0] = self.tree.row(spelling, fi.APPROVED)
        self.assert_contract_failure("noncanonical_path")
        self.tree.inventory, self.tree.entries = original

    def test_15_absolute_paths(self):
        for spelling in ("/etc/hosts", self.tree.path(TRAIN), "/artifacts/model.bin"):
            self.rejected_spelling(spelling)

    def test_16_parent_traversal(self):
        for spelling in ("../model.bin", "artifacts/../data/synthetic/CANARY-train.csv", "artifacts/..", ".."):
            self.rejected_spelling(spelling)

    def test_17_noncanonical_spellings(self):
        for spelling in ("./artifacts/model.bin", "artifacts/./model.bin", "artifacts//model.bin",
                         "artifacts/model.bin/", "artifacts\\model.bin", "artifacts/model.bin.", "artifacts/model.bin ",
                         "artifacts/modél.bin", "artifacts/ｍodel.bin", "artifacts/model\x00.bin",
                         ".hidden/model.bin", "a" * 256):
            with self.subTest(spelling=spelling):
                self.rejected_spelling(spelling)

    def test_18_case_collisions_in_the_documents(self):
        self.tree.inventory["files"]["tracked"]["Artifacts/MODEL.bin"] = sha(MODEL_BYTES)
        self.tree.entries.append(self.tree.row("Artifacts/MODEL.bin", fi.APPROVED))
        self.assert_contract_failure("case_collision_or_alias")

    def test_18b_case_alias_of_an_excluded_path_or_directory(self):
        for alias in ("Reports/canary-pending.json", "Reports/other.json"):
            tree = self.tree = Tree(self)
            tree.inventory["files"]["tracked"][alias] = sha(SECRET)
            tree.entries.append(tree.row(alias, fi.APPROVED))
            self.assert_contract_failure("case_collision_or_alias")

    def test_18c_case_alias_on_disk_for_file_and_directory(self):
        os.rename(self.tree.path(MODEL), self.tree.path("artifacts/Model.bin"))
        self.assert_artifact_outcome("UNSAFE_PATH")
        self.tree = Tree(self)
        os.rename(self.tree.path("artifacts"), self.tree.path("ARTIFACTS"))
        self.assert_artifact_outcome("UNSAFE_PATH")

    def test_19_file_symlink(self):
        for target in (self.tree.path(TRAIN), self.tree.path(MANIFEST), "/nonexistent"):
            with self.subTest(target=os.path.basename(target)):
                os.unlink(self.tree.path(MODEL))
                os.symlink(target, self.tree.path(MODEL))
                self.assert_artifact_outcome("UNSAFE_PATH")

    def test_20_parent_directory_symlink(self):
        os.rename(self.tree.path("artifacts"), self.tree.path("elsewhere"))
        os.symlink("elsewhere", self.tree.path("artifacts"))
        self.assert_artifact_outcome("UNSAFE_PATH", self.run_verify(
            protect=self.tree.canaries() + [self.tree.path("elsewhere/model.bin")]))

    def test_21_hardlink_alias_is_rejected_without_being_opened(self):
        os.unlink(self.tree.path(MODEL))
        os.link(self.tree.path(TRAIN), self.tree.path(MODEL))
        self.assert_artifact_outcome("UNSAFE_PATH")  # the monitor protects the shared inode
        self.tree = Tree(self)
        os.link(self.tree.path(MODEL), self.tree.path("second-name.bin"))  # even a harmless second name
        self.assert_artifact_outcome("UNSAFE_PATH")

    def test_22_directory_and_special_files(self):
        for make in (os.mkdir, os.mkfifo):
            with self.subTest(kind=make.__name__):
                self.tree = Tree(self)
                os.unlink(self.tree.path(MODEL))
                make(self.tree.path(MODEL))
                self.assert_artifact_outcome("UNSAFE_PATH")
        self.tree = Tree(self)
        shutil.rmtree(self.tree.path("artifacts"))
        self.tree.write("artifacts", b"a file where a directory is expected")
        self.assert_artifact_outcome("UNSAFE_PATH")


# ------------------------------------------------------------- item 23: replacement, by seam
class ReplacementTests(Base):
    def at(self, stage, action):
        done = []

        def checkpoint(current, path):
            if current == stage and path == MODEL and not done:
                done.append(True)
                action()

        return mock.patch.object(fi, "_checkpoint", checkpoint), done

    def run_with(self, stage, action, **kwargs):
        arguments = self.tree.arguments()
        patch, done = self.at(stage, action)
        with patch:
            report = self.run_verify(arguments, **kwargs)
        self.assertEqual(done, [True], "the seam ran")
        return report

    def swap(self, make):
        os.unlink(self.tree.path(MODEL))
        make(self.tree.path(MODEL))

    def test_23a_swapped_for_a_symlink_before_open(self):
        report = self.run_with("after_lstat", lambda: self.swap(lambda p: os.symlink(self.tree.path(TRAIN), p)))
        self.assert_artifact_outcome("UNSAFE_PATH", report)

    def test_23b_swapped_for_another_regular_file_before_open(self):
        report = self.run_with("after_lstat", lambda: os.replace(self.copy_of_model(), self.tree.path(MODEL)))
        self.assert_artifact_outcome("UNSAFE_PATH", report)

    def copy_of_model(self):
        self.tree.write("replacement.bin", MODEL_BYTES)  # same bytes: only identity differs
        return self.tree.path("replacement.bin")

    def test_23c_replaced_after_open(self):
        report = self.run_with("after_open", lambda: os.replace(self.copy_of_model(), self.tree.path(MODEL)))
        self.assert_artifact_outcome("UNSAFE_PATH", report)

    def test_23d_modified_in_place_during_and_after_the_read(self):
        def append():
            with open(self.tree.path(MODEL), "ab") as handle:
                handle.write(b"appended")

        def truncate():
            os.truncate(self.tree.path(MODEL), 10)

        for stage, action in (("after_open", append), ("after_open", truncate), ("after_read", append)):
            with self.subTest(stage=stage, action=action.__name__):
                self.tree = Tree(self)
                self.assert_artifact_outcome("UNSAFE_PATH", self.run_with(stage, action))

    def test_23e_removed_after_read(self):
        report = self.run_with("after_read", lambda: os.unlink(self.tree.path(MODEL)))
        self.assert_artifact_outcome("UNSAFE_PATH", report)

    def test_23f_parent_swapped_for_a_symlink_hashes_only_the_validated_handle(self):
        def swap_parent():
            os.rename(self.tree.path("artifacts"), self.tree.path("moved"))
            os.mkdir(self.tree.path("decoy"))
            os.link(self.tree.path(TRAIN), self.tree.path("decoy/model.bin"))
            os.symlink("decoy", self.tree.path("artifacts"))

        report = self.run_with("after_lstat", swap_parent)  # monitor protects the canary inode
        self.assertEqual(self.outcomes(report)[MODEL], "VERIFIED")

    def test_23g_known_limit_hardlink_swap_is_opened_but_never_read(self):
        """Documented limit: in this window the other file is opened; nothing is read from it."""
        arguments = self.tree.arguments()
        canary = _identity(os.stat(self.tree.path(TRAIN)))
        read_from_canary, opened_canary = [], []
        real_read, real_fstat = os.read, os.fstat

        def watching_read(descriptor, count):
            if _identity(real_fstat(descriptor)) == canary:
                read_from_canary.append(count)
            return real_read(descriptor, count)

        def watching_fstat(descriptor):
            status = real_fstat(descriptor)
            if _identity(status) == canary:
                opened_canary.append(True)
            return status

        patch, done = self.at("after_lstat", lambda: self.swap(lambda p: os.link(self.tree.path(TRAIN), p)))
        with patch, mock.patch.object(os, "read", watching_read), mock.patch.object(os, "fstat", watching_fstat):
            report = fi.verify(*arguments)
        self.assertEqual(self.outcomes(report)[MODEL], "UNSAFE_PATH")
        self.assertEqual((done, bool(opened_canary), read_from_canary), ([True], True, []))
        self.tree = Tree(self)  # and the monitor does report that open
        patch, _ = self.at("after_lstat", lambda: self.swap(lambda p: os.link(self.tree.path(TRAIN), p)))
        arguments = self.tree.arguments()
        with patch, AccessMonitor(self.tree.canaries()) as monitor, self.assertRaises(ExcludedAccess):
            fi.verify(*arguments)
        self.assertEqual(len(monitor.attempts), 1)


# ------------------------------------------------------------------ section 9, items 24-30
class FailureAndAccountingTests(Base):
    def test_24_read_failure(self):
        arguments = self.tree.arguments()
        target, real_read = _identity(os.stat(self.tree.path(MODEL))), os.read

        def failing(descriptor, count):
            if _identity(os.fstat(descriptor)) == target:
                raise OSError(errno.EIO, "injected")
            return real_read(descriptor, count)

        with mock.patch.object(os, "read", failing):
            self.assert_artifact_outcome("READ_ERROR", self.run_verify(arguments))

    def test_24b_permission_error_and_size_limit(self):
        os.chmod(self.tree.path(MODEL), 0)
        if os.geteuid() != 0:
            self.assert_artifact_outcome("READ_ERROR")
        os.chmod(self.tree.path(MODEL), 0o644)
        with mock.patch.object(fi, "MAX_ARTIFACT_BYTES", len(MODEL_BYTES) - 1):
            self.assert_artifact_outcome("READ_ERROR")

    def test_25_hash_failure_at_each_step(self):
        class Broken:
            def __init__(self, step):
                self.step, self.real = step, hashlib.sha256()

            def update(self, data):
                if self.step == "update":
                    raise ValueError("injected SECRET text")
                self.real.update(data)

            def hexdigest(self):
                if self.step == "hexdigest":
                    raise ValueError("injected")
                return self.real.hexdigest()

        for step in ("construct", "update", "hexdigest"):
            with self.subTest(step=step):
                factory = mock.Mock(side_effect=ValueError("injected")) if step == "construct" \
                    else (lambda s=step: Broken(s))
                with mock.patch.object(fi, "_new_hash", factory):
                    report = self.run_verify()
                self.assertEqual((report["status"], set(self.outcomes(report)[p] for p in (MODEL, MANIFEST))),
                                 ("FAIL", {"HASH_ERROR"}))
                self.assertNotIn("injected", json.dumps(report))

    CLASSES = {"a": fi.APPROVED, "b": fi.EXCLUDED_DATASET}
    GOOD = [{"path": "a", "classification": fi.APPROVED, "outcome": "VERIFIED"},
            {"path": "b", "classification": fi.EXCLUDED_DATASET, "outcome": "EXCLUDED_DATASET"}]

    def test_26_empty_results(self):
        self.assertEqual(fi._accounting_failures([], self.CLASSES), ["empty_results"])
        self.assertEqual(fi._accounting_failures([], {}), ["empty_results"])
        self.tree.inventory["files"] = {"tracked": {}, "local_only": {}}
        self.tree.entries = []
        report = self.run_verify(protect=self.tree.all_artifacts())
        self.assertEqual((report["status"], report["failure_reasons"]), ("FAIL", ["empty_results"]))

    def test_26b_nothing_approved_cannot_pass(self):
        self.tree.entries = [self.tree.row(r["path"], fi.PENDING) for r in self.tree.entries]
        report = self.run_verify(protect=self.tree.all_artifacts())
        self.assertEqual((report["status"], report["failure_reasons"]), ("FAIL", ["no_mandatory_artifacts"]))

    def test_27_incomplete_results(self):
        self.assertEqual(fi._accounting_failures(self.GOOD, self.CLASSES), [])
        for bad in (self.GOOD[:1], self.GOOD[1:], self.GOOD + [dict(self.GOOD[0], path="c")]):
            self.assertEqual(fi._accounting_failures(bad, self.CLASSES), ["incomplete_or_unexpected_results"])

    def test_27b_interrupted_run_is_not_a_success(self):
        real, calls = fi._verify_artifact, []

        def interrupted(*args):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("injected SECRET text")
            return real(*args)

        with mock.patch.object(fi, "_verify_artifact", interrupted):
            report = self.run_verify()
        self.assertEqual((report["status"], report["failure_reasons"]), ("FAIL", ["internal_error"]))
        self.assertLess(len(report["results"]), 4)
        self.assertNotIn("injected", json.dumps(report))

    def test_28_duplicate_result_identity(self):
        self.assertEqual(fi._accounting_failures([self.GOOD[0], self.GOOD[0]], self.CLASSES),
                         ["duplicate_result_identity"])
        self.assertEqual(fi._accounting_failures(self.GOOD + [self.GOOD[1]], self.CLASSES),
                         ["duplicate_result_identity"])

    def test_28b_outcome_must_fit_its_class(self):
        for row in (dict(self.GOOD[0], outcome="EXCLUDED_DATASET"), dict(self.GOOD[0], outcome="OK"),
                    dict(self.GOOD[0], classification=fi.PENDING, outcome=fi.PENDING)):
            self.assertEqual(fi._accounting_failures([row, self.GOOD[1]], self.CLASSES), ["inconsistent_result"])
        self.assertEqual(fi._accounting_failures(
            [self.GOOD[0], dict(self.GOOD[1], outcome="VERIFIED")], self.CLASSES), ["inconsistent_result"])
        for outcome in ("MISMATCH", "MISSING_MANDATORY", "UNSAFE_PATH", "READ_ERROR", "HASH_ERROR"):
            self.assertEqual(fi._accounting_failures([dict(self.GOOD[0], outcome=outcome), self.GOOD[1]],
                                                     self.CLASSES), ["mandatory_artifact_not_verified"])

    def test_29_report_wording(self):
        passed = self.run_verify()
        self.assertEqual(passed["meaning"],
                         "Every approved mandatory non-dataset artifact in the staging root matched its recorded "
                         "SHA-256. Nothing else was verified. This is not a frozen-file guard pass.")
        self.assertEqual(passed["scope"],
                         "SHA-256 of approved mandatory non-dataset artifacts inside the supplied staging root only")
        self.tree.write(MODEL, b"changed")
        failed = self.run_verify()
        self.assertEqual(failed["meaning"], "Isolated verification failed or is incomplete.")
        for report in (passed, failed):
            text = json.dumps(report)
            self.assertNotIn(self.tree.base, text)
            self.assertNotIn("CANARY-SECRET", text)
            self.assertNotIn('"equivalent_to_full_guard": true', text)
            self.assertNotIn('"dataset_content_integrity": "VERIFIED"', text)
        self.assertEqual(fi._report(["empty_results"], [], [], {})["status"], "FAIL")

    def snapshot(self):
        out = {}
        for directory, _, files in os.walk(self.tree.root):  # synthetic temporary tree only
            for name in files:
                path = os.path.join(directory, name)
                status = os.lstat(path)
                with open(path, "rb") as handle:
                    out[os.path.relpath(path, self.tree.root)] = (handle.read(), status.st_mtime_ns, status.st_mode,
                                                                  status.st_ino)
        return out

    def test_30_inventory_and_staging_are_not_modified(self):
        arguments = self.tree.arguments()
        before = self.snapshot()
        self.run_verify(arguments)
        self.assertEqual(self.snapshot(), before)


# ------------------------------------------------------------------ section 9, items 31-32
class EnvironmentTests(Base):
    def no_filesystem_open(self, root_arguments, reason):
        with mock.patch.object(os, "open", side_effect=AssertionError("nothing may be opened")):
            report = fi.verify(*root_arguments)
        self.assertEqual((report["status"], report["failure_reasons"], report["results"]), ("FAIL", [reason], []))

    def test_31_missing_protection_fails_closed(self):
        arguments = self.tree.arguments()
        self.assertTrue(fi.PLATFORM_SUPPORT and all(fi.PLATFORM_SUPPORT.values()), fi.PLATFORM_SUPPORT)
        for name in list(fi.PLATFORM_SUPPORT):
            with self.subTest(support=name), mock.patch.dict(fi.PLATFORM_SUPPORT, {name: False}):
                self.no_filesystem_open(arguments, "filesystem_protection_unavailable")
        with mock.patch.dict(fi.PLATFORM_SUPPORT, clear=True):
            self.no_filesystem_open(arguments, "filesystem_protection_unavailable")
        for flag in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK", "O_CLOEXEC"):
            def lacking(target, name, flag=flag):
                return False if name == flag else builtins.hasattr(target, name)

            with self.subTest(flag=flag), mock.patch.object(fi, "hasattr", lacking, create=True):
                self.no_filesystem_open(arguments, "filesystem_protection_unavailable")
                with self.assertRaises(fi.ContractError):
                    fi.describe_metadata(self.tree.root, TRAIN)

    def test_32_identifiable_repository_roots_are_rejected(self):
        for marker, is_directory in ((".git", True), (".git", False), (".GIT", True)):
            with self.subTest(marker=marker, directory=is_directory):
                self.tree = Tree(self)
                os.mkdir(self.tree.path(marker)) if is_directory else self.tree.write(marker, b"gitdir: x")
                self.assert_contract_failure("staging_root_is_repository")
        own = os.path.dirname(os.path.dirname(os.path.realpath(fi.__file__)))
        self.no_filesystem_open([own, "inventory.json", "0" * 64, "classification.json", "0" * 64],
                                "staging_root_is_repository")

    def test_32b_root_must_be_explicit_canonical_and_not_a_system_root(self):
        arguments = self.tree.arguments()
        link = os.path.join(self.tree.base, "link")
        os.symlink(self.tree.root, link)
        for root, reason in (
            (None, "staging_root_not_explicit_absolute"), ("", "staging_root_not_explicit_absolute"),
            (".", "staging_root_not_explicit_absolute"), ("stage", "staging_root_not_explicit_absolute"),
            (link, "staging_root_not_canonical"), (self.tree.root + "/", "staging_root_not_canonical"),
            ("/", "staging_root_forbidden"), (os.path.realpath(os.path.expanduser("~")), "staging_root_forbidden"),
        ):
            with self.subTest(reason=reason, kind=type(root).__name__):
                self.no_filesystem_open([root, *arguments[1:]], reason)

    def test_32c_command_line_has_no_defaults_and_fixed_exit_codes(self):
        arguments = self.tree.arguments()
        flags = ("--staging-root", "--inventory", "--inventory-sha256", "--classification", "--classification-sha256")
        argv = [item for pair in zip(flags, arguments) for item in pair]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(fi.main(argv), 0)
        self.assertEqual(json.loads(out.getvalue())["status"], "PASS")
        os.unlink(self.tree.path(MODEL))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(fi.main(argv), 1)
        for incomplete in ([], argv[2:], argv[:-2], ["--staging", self.tree.root, *argv[2:]]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                fi.main(incomplete)
            self.assertEqual(caught.exception.code, 2)


# -------------------------------------------------------- negative controls for the monitor
class MonitorNegativeControls(Base):
    """Deliberately faulty readers that target ONLY synthetic excluded files."""

    def faulty_readers(self):
        canary = self.tree.path(TRAIN)
        symlink, hardlink = self.tree.path("alias-symlink.bin"), self.tree.path("alias-hardlink.bin")
        os.symlink(canary, symlink)
        os.link(canary, hardlink)

        def by_descriptor(path, **kwargs):
            descriptor = os.open(path, os.O_RDONLY, **kwargs)
            try:
                return os.read(descriptor, 100)
            finally:
                os.close(descriptor)

        def relative():
            parent = os.open(self.tree.path("data/synthetic"), os.O_RDONLY | os.O_DIRECTORY)
            try:
                return by_descriptor("CANARY-train.csv", dir_fd=parent)
            finally:
                os.close(parent)

        def builtin(path):
            with open(path, "rb") as handle:
                return handle.read()

        readers = {
            "direct_builtin": lambda: builtin(canary),
            "direct_pathlib": lambda: pathlib.Path(canary).read_bytes(),
            "direct_os_open": lambda: by_descriptor(canary),
            "dir_fd_relative": relative,
            "alias_symlink_builtin": lambda: builtin(symlink),
            "alias_symlink_os_open": lambda: by_descriptor(symlink),
            "alias_hardlink": lambda: builtin(hardlink),
            "alias_traversal": lambda: builtin(self.tree.path("artifacts/../" + TRAIN)),
        }
        upper = self.tree.path("DATA/SYNTHETIC/canary-TRAIN.csv")
        if os.path.exists(upper):  # case-insensitive filesystem
            readers["alias_case"] = lambda: builtin(upper)
        return readers

    def test_each_faulty_reader_really_reads_without_the_monitor(self):
        for name, reader in self.faulty_readers().items():
            with self.subTest(reader=name):
                self.assertEqual(reader(), SECRET)

    def test_monitor_rejects_each_faulty_reader_before_any_content_is_read(self):
        readers = self.faulty_readers()
        canaries = self.tree.canaries()
        for name, reader in readers.items():
            with self.subTest(reader=name):
                obtained, reads = [], []
                real_read = os.read
                with mock.patch.object(os, "read", lambda *a: (reads.append(a), real_read(*a))[1]):
                    with AccessMonitor(canaries) as monitor:
                        with self.assertRaises(ExcludedAccess):
                            obtained.append(reader())
                self.assertEqual((obtained, reads, len(monitor.attempts)), ([], [], 1))

    def test_monitor_allows_unprotected_files_and_blocks_processes(self):
        with AccessMonitor(self.tree.canaries()) as monitor:
            with open(self.tree.path(MANIFEST), "rb") as handle:
                self.assertEqual(handle.read(), MANIFEST_BYTES)
        self.assertEqual(monitor.attempts, [])
        with AccessMonitor(self.tree.canaries()) as monitor, self.assertRaises(ExcludedAccess):
            subprocess.run(["/bin/cat", self.tree.path(TRAIN)], capture_output=True)
        self.assertEqual(len(monitor.attempts), 1)

    def test_documented_monitor_gap_already_open_descriptor(self):
        descriptor = os.open(self.tree.path(TRAIN), os.O_RDONLY)
        try:
            with AccessMonitor(self.tree.canaries()) as monitor:
                self.assertEqual(os.read(descriptor, 100), SECRET)  # NOT intercepted
            self.assertEqual(monitor.attempts, [])
        finally:
            os.close(descriptor)


if __name__ == "__main__":
    unittest.main()
