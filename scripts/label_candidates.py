#!/usr/bin/env python3
"""Resumably exact-label CSV candidates in independently verifiable shards.

The work directory is deliberately durable. A shard is only marked complete
after its temporary CSV has been checked and atomically renamed, so a killed
run can resume without relabeling valid work or merging a partial shard.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", type=Path, default=Path("build/estimate"))
    parser.add_argument("--arch", type=Path, default=Path("arch"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=5_000)
    parser.add_argument("--limit", type=int)
    # Test hook: proves a completed real shard is reused after an interruption.
    parser.add_argument("--stop-after-shards", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash architecture contents and names, so a renamed JSON is noticed."""
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def count_rows(path: Path) -> int:
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ["From", "To"]:
            raise ValueError(f"{path} must have exactly the header From,To")
        return sum(1 for _ in reader)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def make_manifest(args: argparse.Namespace, total: int, work_dir: Path) -> dict[str, Any]:
    shards = []
    for number, offset in enumerate(range(0, total, args.shard_size)):
        limit = min(args.shard_size, total - offset)
        shards.append({
            "id": number, "offset": offset, "limit": limit,
            "path": f"shards/shard_{number:06d}.csv", "state": "pending", "sha256": None,
            "exact_labeled_rows": None, "exact_unreachable_rows": None,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "input": {"path": str(args.input.resolve()), "sha256": sha256_file(args.input)},
        "architecture": {"path": str(args.arch.resolve()), "sha256": sha256_tree(args.arch)},
        "estimate": {"path": str(args.estimate.resolve()), "sha256": sha256_file(args.estimate)},
        "total_rows": total, "shard_size": args.shard_size, "work_dir": str(work_dir.resolve()),
        "shards": shards, "merged_output": None,
    }


def validate_manifest(manifest: dict[str, Any], args: argparse.Namespace, total: int) -> None:
    expected = make_manifest(args, total, args.work_dir)
    for key in ("schema_version", "input", "architecture", "estimate", "total_rows", "shard_size"):
        if manifest.get(key) != expected[key]:
            raise ValueError(
                f"existing manifest does not match current {key}; use a new --work-dir rather than mixing runs"
            )


def input_rows(path: Path, offset: int, limit: int) -> list[tuple[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for _ in range(offset):
            if next(reader, None) is None:
                raise ValueError("input ended before shard offset")
        result = []
        for _ in range(limit):
            row = next(reader, None)
            if row is None or len(row) != 2:
                raise ValueError("input ended inside shard or has a malformed row")
            result.append((row[0], row[1]))
    return result


def validate_shard(path: Path, expected_rows: list[tuple[str, str]]) -> str:
    if not path.is_file():
        raise ValueError(f"missing shard {path}")
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ["From", "To", "delay"]:
            raise ValueError(f"invalid shard header in {path}")
        actual = list(reader)
    if len(actual) != len(expected_rows):
        raise ValueError(f"{path} has {len(actual)} rows, expected {len(expected_rows)}")
    for index, (row, expected) in enumerate(zip(actual, expected_rows)):
        if len(row) != 3 or (row[0], row[1]) != expected:
            raise ValueError(f"{path} endpoint mismatch at relative row {index}")
        try:
            int(row[2])
        except ValueError as error:
            raise ValueError(f"{path} has non-integer delay at relative row {index}") from error
    return sha256_file(path)


def parse_summary(stdout: str) -> dict[str, int]:
    summary: dict[str, int] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"exact_labeled_rows", "exact_unreachable_rows"}:
            summary[key] = int(value)
    return summary


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.shard_size <= 0:
        raise ValueError("--workers and --shard-size must be positive")
    if args.stop_after_shards is not None and args.stop_after_shards <= 0:
        raise ValueError("--stop-after-shards must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    for path, label in ((args.estimate, "estimate"), (args.arch, "arch"), (args.input, "input")):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    input_total = count_rows(args.input)
    total = min(input_total, args.limit) if args.limit is not None else input_total
    if total == 0:
        raise ValueError("input contains no request rows")
    args.work_dir = args.work_dir or Path(str(args.output) + ".work")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "shards").mkdir(exist_ok=True)
    manifest_path = args.work_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest, args, total)
    else:
        manifest = make_manifest(args, total, args.work_dir)
        atomic_json(manifest_path, manifest)

    # Recover a completed shard if a process died between atomic rename and the
    # manifest write; reject corrupt files rather than trusting their filename.
    changed = False
    for shard in manifest["shards"]:
        shard_path = args.work_dir / shard["path"]
        if shard_path.exists():
            try:
                digest = validate_shard(
                    shard_path, input_rows(args.input, int(shard["offset"]), int(shard["limit"]))
                )
            except ValueError:
                if shard["state"] == "complete":
                    raise
            else:
                if shard["state"] != "complete" or shard["sha256"] != digest:
                    shard["state"] = "complete"
                    shard["sha256"] = digest
                    changed = True
    if changed:
        atomic_json(manifest_path, manifest)

    pending = [shard for shard in manifest["shards"] if shard["state"] != "complete"]

    def run(shard: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int], str]:
        offset, limit = int(shard["offset"]), int(shard["limit"])
        destination = args.work_dir / shard["path"]
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.unlink(missing_ok=True)
        command = [
            str(args.estimate), "--exact-label", "-in", str(args.input), "-out", str(temporary),
            "--arch", str(args.arch), "--offset", str(offset), "--limit", str(limit),
        ]
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        digest = validate_shard(temporary, input_rows(args.input, offset, limit))
        os.replace(temporary, destination)
        return shard, parse_summary(completed.stdout), digest

    completed_this_run = 0
    with ThreadPoolExecutor(max_workers=min(args.workers, len(pending) or 1)) as executor:
        futures = {executor.submit(run, shard): shard for shard in pending}
        for future in as_completed(futures):
            shard, summary, digest = future.result()
            shard["state"] = "complete"
            shard["sha256"] = digest
            shard["exact_labeled_rows"] = summary.get("exact_labeled_rows")
            shard["exact_unreachable_rows"] = summary.get("exact_unreachable_rows")
            atomic_json(manifest_path, manifest)
            completed_this_run += 1
            if args.stop_after_shards is not None and completed_this_run >= args.stop_after_shards:
                for unfinished in futures:
                    unfinished.cancel()
                raise RuntimeError("intentional interruption after completed shard(s); rerun without test hook")

    merge_tmp = args.output.with_name(args.output.name + ".tmp")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with merge_tmp.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["From", "To", "delay"])
        for shard in sorted(manifest["shards"], key=lambda item: int(item["offset"])):
            if shard["state"] != "complete":
                raise RuntimeError("cannot merge incomplete shard")
            with (args.work_dir / shard["path"]).open(newline="") as source:
                reader = csv.reader(source)
                next(reader)
                writer.writerows(reader)
    os.replace(merge_tmp, args.output)
    report = {
        "input_rows": total,
        "labeled_rows": sum(int(item["limit"]) for item in manifest["shards"]),
        "unreachable_rows": sum(int(item.get("exact_unreachable_rows") or 0) for item in manifest["shards"]),
        "workers": args.workers, "work_dir": str(args.work_dir), "output": str(args.output),
        "output_sha256": sha256_file(args.output), "shards": manifest["shards"],
    }
    manifest["merged_output"] = {"path": str(args.output.resolve()), "sha256": report["output_sha256"]}
    atomic_json(manifest_path, manifest)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "shards"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
