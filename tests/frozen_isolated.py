"""Dataset-isolated frozen-artifact verifier (L8-3DI). Standard library only.

This is NOT the frozen-file guard (``tests/test_frozen_artifacts.py``) and is not equivalent to it.
It hashes only the paths a reviewed classification positively approves, inside an explicitly
supplied staging directory. Excluded paths are never passed to any filesystem call. Dataset
content integrity is always reported as NOT VERIFIED.

Permission to read an artifact comes from one thing only: an ``APPROVED_MANDATORY`` row in the
classification. Presence in the inventory, tracked status, extension, directory or absence from a
denylist grant nothing. The inventory and the whole classification are validated before the first
artifact is touched, and both control documents are pinned by SHA-256.

Platform assumptions: POSIX directory-relative, no-follow file operations. Their absence fails
closed. No protection is claimed against a privileged adversary or arbitrary concurrent mutation of
the execution environment; in particular, if a validated entry is swapped for a link to another
file between the pre-open check and the open, that file can be opened (it is rejected by the
identity check before anything is read). The primary isolation boundary is that the execution
environment contains no dataset at all, which this module cannot establish.

Usage (all arguments required):
    python frozen_isolated.py --staging-root DIR --inventory NAME --inventory-sha256 HEX \\
        --classification NAME --classification-sha256 HEX
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import stat
import sys

APPROVED = "APPROVED_MANDATORY"
EXCLUDED_DATASET = "EXCLUDED_DATASET"
PENDING = "EXCLUDED_PENDING_CLASSIFICATION"
CLASSES = (APPROVED, EXCLUDED_DATASET, PENDING)

VERIFIED = "VERIFIED"
MISMATCH = "MISMATCH"
MISSING = "MISSING_MANDATORY"
UNSAFE = "UNSAFE_PATH"
READ_ERROR = "READ_ERROR"
HASH_ERROR = "HASH_ERROR"
APPROVED_OUTCOMES = (VERIFIED, MISMATCH, MISSING, UNSAFE, READ_ERROR, HASH_ERROR)

NOT_VERIFIED = "NOT VERIFIED"
NOT_PERFORMED = "NOT PERFORMED"
REPORT_SCHEMA = "frozen-isolated-report/1"
PASS_MEANING = ("Every approved mandatory non-dataset artifact in the staging root matched its recorded "
                "SHA-256. Nothing else was verified. This is not a frozen-file guard pass.")
FAIL_MEANING = "Isolated verification failed or is incomplete."
SCOPE = "SHA-256 of approved mandatory non-dataset artifacts inside the supplied staging root only"

MAX_CONTROL_BYTES = 1 << 20
MAX_ARTIFACT_BYTES = 64 << 20
CHUNK = 1 << 16
_COMPONENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _platform_support() -> dict[str, bool]:
    """Evaluated once at import, from the original ``os`` functions."""
    support = {
        "open_dir_fd": os.open in os.supports_dir_fd,
        "stat_dir_fd": os.stat in os.supports_dir_fd,
        "stat_no_follow": os.stat in os.supports_follow_symlinks,
        "listdir_fd": os.listdir in os.supports_fd,
    }
    for flag in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK", "O_CLOEXEC"):
        support[flag] = hasattr(os, flag)
    return support


PLATFORM_SUPPORT = _platform_support()


class ContractError(Exception):
    """Verification cannot proceed. ``reason`` is a fixed code and carries no file content."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _Outcome(Exception):
    def __init__(self, outcome: str):
        super().__init__(outcome)
        self.outcome = outcome


def _checkpoint(stage: str, path: str) -> None:
    """Deterministic test seam; does nothing in normal use."""


def _require_protections() -> None:
    if not PLATFORM_SUPPORT or not all(PLATFORM_SUPPORT.values()):
        raise ContractError("filesystem_protection_unavailable")
    if not all(hasattr(os, f) for f in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK", "O_CLOEXEC")):
        raise ContractError("filesystem_protection_unavailable")


# ------------------------------------------------------------------------------- paths
def is_canonical(path: object) -> bool:
    if not isinstance(path, str) or not path:
        return False
    return all(len(part) <= 255 and _COMPONENT.fullmatch(part) is not None and not part.endswith(".")
               for part in path.split("/"))


def _check_staging_root(root: object) -> str:
    if not isinstance(root, str) or not root or not os.path.isabs(root):
        raise ContractError("staging_root_not_explicit_absolute")
    if os.path.realpath(root) != root:
        raise ContractError("staging_root_not_canonical")
    if root == os.path.sep or root == os.path.realpath(os.path.expanduser("~")):
        raise ContractError("staging_root_forbidden")
    if root == os.path.dirname(os.path.dirname(os.path.realpath(__file__))):
        raise ContractError("staging_root_is_repository")
    return root


def _open_root(root: str) -> int:
    try:
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        raise ContractError("staging_root_unavailable") from None
    try:
        if any(name.casefold() == ".git" for name in os.listdir(descriptor)):
            raise ContractError("staging_root_is_repository")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _identity(status: os.stat_result) -> tuple[int, int]:
    return status.st_dev, status.st_ino


def _entry_status(parent: int, name: str) -> os.stat_result:
    """lstat of an entry whose name must appear exactly, with no case-folded sibling."""
    names = os.listdir(parent)
    folded = [n for n in names if n.casefold() == name.casefold()]
    if not folded:
        raise _Outcome(MISSING)
    if folded != [name]:
        raise _Outcome(UNSAFE)
    return os.stat(name, dir_fd=parent, follow_symlinks=False)


def _walk(root_fd: int, parts: list[str]) -> int:
    """Descriptor of the parent directory of ``parts[-1]``; every step is checked and no-follow."""
    current = os.dup(root_fd)
    try:
        for name in parts[:-1]:
            before = _entry_status(current, name)
            if not stat.S_ISDIR(before.st_mode):
                raise _Outcome(UNSAFE)
            following = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
            os.close(current)
            current = following
            if _identity(os.fstat(current)) != _identity(before):
                raise _Outcome(UNSAFE)
    except BaseException:
        os.close(current)
        raise
    return current


def _new_hash():
    return hashlib.sha256()


def _read_validated(root_fd: int, path: str, limit: int, consume) -> None:
    """Pass the bytes of one safely opened regular file to ``consume``. The descriptor that was
    validated is the one that is read."""
    parts = path.split("/")
    parent = _walk(root_fd, parts)
    descriptor = None
    try:
        name = parts[-1]
        before = _entry_status(parent, name)
        # checked BEFORE opening, so that a link to another file is never opened
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise _Outcome(UNSAFE)
        _checkpoint("after_lstat", path)
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or _identity(opened) != _identity(before):
            raise _Outcome(UNSAFE)
        if opened.st_size > limit:
            raise _Outcome(READ_ERROR)
        _checkpoint("after_open", path)
        total = 0
        while total <= opened.st_size:
            chunk = os.read(descriptor, min(CHUNK, opened.st_size + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            consume(chunk)
        _checkpoint("after_read", path)
        closing = os.fstat(descriptor)
        unchanged = (closing.st_size, closing.st_mtime_ns, closing.st_ctime_ns, closing.st_nlink)
        if unchanged != (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns, opened.st_nlink) \
                or total != opened.st_size:
            raise _Outcome(UNSAFE)
        try:
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            raise _Outcome(UNSAFE) from None
        if _identity(current) != _identity(opened):
            raise _Outcome(UNSAFE)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _os_error_outcome(error: OSError) -> str:
    if error.errno == errno.ENOENT:
        return MISSING
    if error.errno in (errno.ELOOP, errno.ENOTDIR, errno.EMLINK):
        return UNSAFE
    return READ_ERROR


def _verify_artifact(root_fd: int, path: str, expected: str) -> str:
    try:
        digest = _new_hash()
    except Exception:
        return HASH_ERROR

    def consume(chunk: bytes) -> None:
        try:
            digest.update(chunk)
        except Exception:
            raise _Outcome(HASH_ERROR) from None

    try:
        _read_validated(root_fd, path, MAX_ARTIFACT_BYTES, consume)
    except _Outcome as stop:
        return stop.outcome
    except OSError as error:
        return _os_error_outcome(error)
    try:
        actual = digest.hexdigest()
    except Exception:
        return HASH_ERROR
    return VERIFIED if actual == expected else MISMATCH


# --------------------------------------------------------------------- control documents
def _no_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ContractError("duplicate_json_key")
        out[key] = value
    return out


def _reject_constant(_name):
    raise ContractError("malformed_json")


def _load_control(root_fd: int, name: object, pinned: object) -> dict:
    if not is_canonical(name) or "/" in name or not name.endswith(".json"):
        raise ContractError("control_document_path_not_allowed")
    if not isinstance(pinned, str) or not _DIGEST.fullmatch(pinned):
        raise ContractError("control_document_pin_malformed")
    chunks: list[bytes] = []
    try:
        _read_validated(root_fd, name, MAX_CONTROL_BYTES, chunks.append)
    except (_Outcome, OSError):
        raise ContractError("control_document_unreadable") from None
    data = b"".join(chunks)
    if hashlib.sha256(data).hexdigest() != pinned:
        raise ContractError("control_document_pin_mismatch")
    try:
        document = json.loads(data.decode("utf-8"), object_pairs_hook=_no_duplicate_keys,
                              parse_constant=_reject_constant)
    except ContractError:
        raise
    except (ValueError, RecursionError):
        raise ContractError("malformed_json") from None
    if not isinstance(document, dict):
        raise ContractError("malformed_json")
    return document


def _parse_inventory(document: dict) -> tuple[dict[str, str], str]:
    files = document.get("files")
    holdout = document.get("excluded_holdout")
    if (document.get("inventory_version") != "1.0.0" or not isinstance(files, dict)
            or set(files) != {"tracked", "local_only"} or not all(isinstance(v, dict) for v in files.values())
            or not isinstance(holdout, dict) or holdout.get("hashed_here") is not False
            or not isinstance(holdout.get("path"), str)):
        raise ContractError("inventory_structure")
    if set(files["tracked"]) & set(files["local_only"]):
        raise ContractError("inventory_duplicate_entry")
    expected = {**files["tracked"], **files["local_only"]}
    if not all(is_canonical(p) for p in (*expected, holdout["path"])):
        raise ContractError("noncanonical_path")
    if not all(isinstance(h, str) and _DIGEST.fullmatch(h) for h in expected.values()):
        raise ContractError("malformed_expected_hash")
    if holdout["path"] in expected:
        raise ContractError("holdout_in_inventory")
    return expected, holdout["path"]


def _parse_rows(rows: object) -> dict[str, dict]:
    if not isinstance(rows, list):
        raise ContractError("classification_structure")
    out: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "class", "reason"} \
                or not isinstance(row["reason"], str) or not row["reason"].strip():
            raise ContractError("classification_structure")
        if not is_canonical(row["path"]):
            raise ContractError("noncanonical_path")
        if row["class"] not in CLASSES:
            raise ContractError("unknown_classification")
        if row["path"] in out:
            raise ContractError("duplicate_classification" if out[row["path"]]["class"] == row["class"]
                                else "contradictory_classification")
        out[row["path"]] = row
    return out


def _parse_classification(document: dict, inventory_pin: str, expected: dict[str, str], holdout: str,
                          control_names: tuple[str, str]) -> tuple[dict[str, dict], dict[str, dict]]:
    if set(document) != {"classification_version", "review_status", "inventory_sha256", "entries",
                         "excluded_outside_inventory"} or document["classification_version"] != "1":
        raise ContractError("classification_structure")
    if document["review_status"] != "REVIEWED":
        raise ContractError("classification_not_reviewed")
    if document["inventory_sha256"] != inventory_pin:
        raise ContractError("classification_for_other_inventory")
    entries = _parse_rows(document["entries"])
    outside = _parse_rows(document["excluded_outside_inventory"])
    if set(entries) & set(outside):
        raise ContractError("contradictory_classification")
    if set(expected) - set(entries):
        raise ContractError("missing_classification")
    if set(entries) - set(expected):
        raise ContractError("classification_for_unknown_entry")
    if set(outside) & set(expected) or any(r["class"] == APPROVED for r in outside.values()):
        raise ContractError("outside_inventory_must_be_excluded")
    if holdout not in outside or outside[holdout]["class"] != EXCLUDED_DATASET:
        raise ContractError("holdout_not_explicitly_excluded")
    every = [*entries, *outside, *control_names]
    prefixes = {"/".join(p.split("/")[:i]) for p in every for i in range(1, p.count("/") + 2)}
    if len({p.casefold() for p in every}) != len(every) or len({p.casefold() for p in prefixes}) != len(prefixes):
        raise ContractError("case_collision_or_alias")
    for path, row in entries.items():
        if row["class"] == APPROVED and path.split("/")[0].casefold() == "data":
            raise ContractError("approved_path_under_data")
    return entries, outside


# --------------------------------------------------------------------------------- report
def _accounting_failures(results: list[dict], classes: dict[str, str]) -> list[str]:
    """Reasons the result set does not justify success. ``classes`` maps every inventory path to
    its classification."""
    if not results or not classes:
        return ["empty_results"]
    paths = [r.get("path") for r in results]
    if len(set(paths)) != len(paths):
        return ["duplicate_result_identity"]
    if len(results) != len(classes) or set(paths) != set(classes):
        return ["incomplete_or_unexpected_results"]
    for row in results:
        wanted = classes[row["path"]]
        allowed = APPROVED_OUTCOMES if wanted == APPROVED else (wanted,)
        if row.get("classification") != wanted or row.get("outcome") not in allowed:
            return ["inconsistent_result"]
    if APPROVED not in classes.values():
        return ["no_mandatory_artifacts"]
    if any(r["outcome"] != VERIFIED for r in results if r["classification"] == APPROVED):
        return ["mandatory_artifact_not_verified"]
    return []


def _report(reasons: list[str], results: list[dict], excluded: list[dict], pins: dict) -> dict:
    passed = not reasons
    counts: dict[str, int] = {}
    for row in results:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    return {
        "report_schema": REPORT_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "meaning": PASS_MEANING if passed else FAIL_MEANING,
        "failure_reasons": reasons,
        "dataset_content_integrity": NOT_VERIFIED,
        "equivalent_to_full_guard": False,
        "scope": SCOPE,
        "results": results,
        "counts": counts,
        "excluded_paths": excluded,
        "historical_holdout_cross_checks": NOT_PERFORMED,
        "historical_repository_addition_checks": NOT_PERFORMED,
        "staging_exclusivity_check": NOT_PERFORMED,
        "metadata_checks": NOT_PERFORMED,
        "pins": pins,
    }


def verify(staging_root: str, inventory_name: str, inventory_sha256: str, classification_name: str,
           classification_sha256: str) -> dict:
    """Run one isolated verification and return its report. Never raises for a verification
    failure: anything that prevents complete verification produces ``status: FAIL``."""
    pins: dict = {"inventory_sha256": None, "classification_sha256": None}
    results: list[dict] = []
    excluded: list[dict] = []
    root_fd = None
    try:
        _require_protections()
        root_fd = _open_root(_check_staging_root(staging_root))
        inventory = _load_control(root_fd, inventory_name, inventory_sha256)
        pins["inventory_sha256"] = inventory_sha256
        expected, holdout = _parse_inventory(inventory)
        document = _load_control(root_fd, classification_name, classification_sha256)
        pins["classification_sha256"] = classification_sha256
        entries, outside = _parse_classification(document, inventory_sha256, expected, holdout,
                                                 (inventory_name, classification_name))
        # ---- everything is validated; only now are artifacts touched ----
        excluded = [{"path": p, "classification": r["class"], "reason": r["reason"]}
                    for p, r in sorted({**entries, **outside}.items()) if r["class"] != APPROVED]
        for path in sorted(expected):
            kind = entries[path]["class"]
            outcome = _verify_artifact(root_fd, path, expected[path]) if kind == APPROVED else kind
            results.append({"path": path, "classification": kind, "outcome": outcome})
        reasons = _accounting_failures(results, {p: entries[p]["class"] for p in expected})
    except ContractError as failure:
        reasons = [failure.reason]
    except Exception:
        reasons = ["internal_error"]
    finally:
        if root_fd is not None:
            os.close(root_fd)
    return _report(reasons, results, excluded, pins)


# ------------------------------------------------------------- optional metadata helper
def describe_metadata(staging_root: str, path: str) -> dict:
    """SEPARATE from verification and never called by it. The final entry is only lstat-ed (never
    opened). Metadata establishes nothing about content."""
    _require_protections()
    if not is_canonical(path):
        raise ContractError("noncanonical_path")
    report = {"path": path, "present": False, "type": None, "size": None, "link_count": None,
              "content_integrity": NOT_VERIFIED}
    root_fd = _open_root(_check_staging_root(staging_root))
    try:
        parts = path.split("/")
        parent = _walk(root_fd, parts)
        try:
            status = _entry_status(parent, parts[-1])
        finally:
            os.close(parent)
        kind = ("regular" if stat.S_ISREG(status.st_mode) else
                "symlink" if stat.S_ISLNK(status.st_mode) else "other")
        report.update(present=True, type=kind, size=status.st_size, link_count=status.st_nlink)
    except _Outcome as stop:
        if stop.outcome != MISSING:
            raise ContractError("unsafe_path") from None
    except OSError:
        raise ContractError("metadata_unavailable") from None
    finally:
        os.close(root_fd)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dataset-isolated frozen-artifact verification.",
                                     allow_abbrev=False)
    for name in ("--staging-root", "--inventory", "--inventory-sha256", "--classification",
                 "--classification-sha256"):
        parser.add_argument(name, required=True)
    args = parser.parse_args(argv)  # exits with status 2 on a usage error
    report = verify(args.staging_root, args.inventory, args.inventory_sha256, args.classification,
                    args.classification_sha256)
    json.dump(report, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
