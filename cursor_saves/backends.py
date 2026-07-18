"""Sync backends for cursaves snapshot storage.

Each backend handles the transport layer: syncing snapshot files between
the local ``~/.cursaves/snapshots/`` directory and a remote store (git repo,
S3 bucket, Azure container, etc.).

The local snapshots directory is always the source of truth for reads —
backends just keep it in sync with a remote.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional, Sequence


_CONFIG_PATH = Path.home() / ".config" / "cursaves" / "config.json"


def expand_snapshot_export_paths(saved: Sequence[Path]) -> list[Path]:
    """Expand checkpoint return paths to include meta sidecars and shards.

    ``save_snapshot`` returns the logical ``{id}.json.gz`` path even when the
    payload was written as ``{id}.json.gz.00`` shards.
    """
    out: list[Path] = []
    seen: set[Path] = set()
    for path in saved:
        path = Path(path)
        parent = path.parent
        name = path.name
        if name.endswith(".json.gz"):
            composer_id = name[: -len(".json.gz")]
        elif ".json.gz." in name:
            composer_id = name.split(".json.gz.", 1)[0]
        else:
            # Unknown shape — still try to add the path itself
            if path.exists() and path not in seen:
                seen.add(path)
                out.append(path)
            continue

        for candidate in parent.glob(f"{composer_id}.json.gz*"):
            if candidate.name.endswith(".meta.json"):
                continue
            if candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
        meta = parent / f"{composer_id}.meta.json"
        if meta.exists() and meta not in seen:
            seen.add(meta)
            out.append(meta)
    return out


def _format_bytes(num: int) -> str:
    if num < 1024:
        return f"{num} B"
    if num < 1024 * 1024:
        return f"{num / 1024:.1f} KiB"
    return f"{num / (1024 * 1024):.1f} MiB"


def _paths_total_size(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


# ── Abstract base ────────────────────────────────────────────────────────


class SyncBackend(ABC):
    """Interface every sync backend must implement."""

    @abstractmethod
    def pull(self, snapshots_dir: Path) -> bool:
        """Download remote snapshots into *snapshots_dir*.

        Must be idempotent — running twice without changes is a no-op.
        Returns True on success, False on failure.
        """

    @abstractmethod
    def push(
        self,
        snapshots_dir: Path,
        only_paths: Optional[Sequence[Path]] = None,
    ) -> bool:
        """Upload local snapshots from *snapshots_dir* to the remote.

        If *only_paths* is given, upload/commit just those files (plus callers
        should already expand meta/shards). If None, sync the whole tree
        (used for deletes / full flush).

        Returns True on success, False on failure.
        """
    @abstractmethod
    def has_remote(self) -> bool:
        """Return True if a remote target is configured."""

    @abstractmethod
    def is_initialized(self) -> bool:
        """Return True if the backend has been set up (init already run)."""


# ── Git backend ──────────────────────────────────────────────────────────


class GitBackend(SyncBackend):
    """Original backend: a local git repo at *sync_dir* with an optional remote."""

    def __init__(self, sync_dir: Path):
        self.sync_dir = sync_dir

    # -- SyncBackend interface ------------------------------------------

    def pull(self, snapshots_dir: Path) -> bool:
        if not self.has_remote():
            return True
        return self._reset_to_origin()

    def push(
        self,
        snapshots_dir: Path,
        only_paths: Optional[Sequence[Path]] = None,
    ) -> bool:
        if only_paths is not None:
            files = expand_snapshot_export_paths(list(only_paths))
            if not files:
                return True
            self._warn_snapshot_backlog(snapshots_dir, files)
            rels = self._rels_under_sync_dir(files)
            if not rels:
                return True
            subprocess.run(
                ["git", "add", "--"] + rels,
                cwd=str(self.sync_dir), capture_output=True,
            )
        else:
            subprocess.run(
                ["git", "add", "snapshots/"],
                cwd=str(self.sync_dir), capture_output=True,
            )

        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(self.sync_dir), capture_output=True,
        )
        if result.returncode == 0:
            return True  # nothing to commit

        staged = self._staged_snapshot_files()
        staged_size = _paths_total_size(staged)
        print(
            f"  Commit {len(staged)} file(s), {_format_bytes(staged_size)}"
            + (" (scoped to this export)" if only_paths is not None else " (full snapshots/)"),
            flush=True,
        )

        from . import paths
        hostname = paths.get_machine_id()
        msg = f"[{hostname}] sync snapshots"
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(self.sync_dir), capture_output=True,
        )

        if self.has_remote():
            print(
                f"  Pushing {_format_bytes(staged_size)} to origin...",
                end="",
                flush=True,
            )
            try:
                push_result = subprocess.run(
                    ["git", "push", "-u", "origin", "main"],
                    cwd=str(self.sync_dir),
                    capture_output=True, text=True, timeout=600,
                )
                if push_result.returncode != 0:
                    print(" failed", file=sys.stderr)
                    print(f"  Push failed: {push_result.stderr.strip()}", file=sys.stderr)
                    return False
            except subprocess.TimeoutExpired:
                print(" timed out", file=sys.stderr)
                return False
            print(" done", flush=True)
        return True

    def _rels_under_sync_dir(self, files: Sequence[Path]) -> list[str]:
        rels: list[str] = []
        sync_root = self.sync_dir.resolve()
        for path in files:
            try:
                resolved = path.resolve()
                rel = resolved.relative_to(sync_root)
            except (OSError, ValueError):
                continue
            rels.append(str(rel))
        return rels

    def _staged_snapshot_files(self) -> list[Path]:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--", "snapshots/"],
            cwd=str(self.sync_dir), capture_output=True, text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        return [
            self.sync_dir / line
            for line in result.stdout.splitlines()
            if line.strip()
        ]

    def _warn_snapshot_backlog(
        self,
        snapshots_dir: Path,
        exporting: Sequence[Path],
    ) -> None:
        """Warn when other uncommitted snapshot files exist beyond this export."""
        exporting_set = {p.resolve() for p in exporting if p.exists()}
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", "snapshots/"],
            cwd=str(self.sync_dir), capture_output=True, text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return

        backlog: list[Path] = []
        for line in result.stdout.splitlines():
            # porcelain: XY path  (or XY origin -> path for renames)
            if len(line) < 4:
                continue
            entry = line[3:].strip()
            if " -> " in entry:
                entry = entry.split(" -> ", 1)[1].strip()
            # Unquoted paths only (cursaves snapshot names are safe)
            path = (self.sync_dir / entry).resolve()
            if path in exporting_set:
                continue
            if path.exists() and path.is_file():
                backlog.append(path)

        if not backlog:
            return

        backlog_size = _paths_total_size(backlog)
        export_size = _paths_total_size(exporting)
        print(
            f"  Note: {_format_bytes(backlog_size)} in {len(backlog)} other uncommitted "
            f"snapshot file(s) not included in this push "
            f"(this export is {_format_bytes(export_size)}). "
            f"Use a full `cursaves push` / migrate if you want them synced.",
            file=sys.stderr,
            flush=True,
        )

    def has_remote(self) -> bool:
        try:
            result = subprocess.run(
                ["git", "remote"],
                capture_output=True, text=True,
                cwd=str(self.sync_dir),
            )
            return result.returncode == 0 and result.stdout.strip() != ""
        except FileNotFoundError:
            return False

    def is_initialized(self) -> bool:
        return (self.sync_dir / ".git").exists()

    # -- Git-specific helpers -------------------------------------------

    def _reset_to_origin(self) -> bool:
        """Fetch + hard-reset to origin/main.  Remote is ground truth."""
        if not self.sync_dir.exists():
            return False

        for abort_cmd in (
            ["git", "rebase", "--abort"],
            ["git", "merge", "--abort"],
            ["git", "cherry-pick", "--abort"],
        ):
            subprocess.run(abort_cmd, cwd=str(self.sync_dir), capture_output=True)

        if not self.has_remote():
            subprocess.run(
                ["git", "checkout", "-f", "-B", "main"],
                cwd=str(self.sync_dir), capture_output=True,
            )
            return True

        try:
            fetch = subprocess.run(
                ["git", "fetch", "--depth", "1", "origin"],
                cwd=str(self.sync_dir),
                capture_output=True, text=True, timeout=180,
            )
            if fetch.returncode != 0:
                return False

            subprocess.run(
                ["git", "checkout", "-f", "-B", "main", "origin/main"],
                cwd=str(self.sync_dir), capture_output=True,
            )
            subprocess.run(
                ["git", "reset", "--hard", "origin/main"],
                cwd=str(self.sync_dir), capture_output=True,
            )
            subprocess.run(
                ["git", "branch", "--set-upstream-to=origin/main", "main"],
                cwd=str(self.sync_dir), capture_output=True,
            )
            subprocess.run(
                ["git", "clean", "-fd"],
                cwd=str(self.sync_dir), capture_output=True,
            )
            return True
        except subprocess.TimeoutExpired:
            return False

    def init_repo(self, remote: Optional[str] = None):
        """Create the git repo and optionally add a remote."""
        self.sync_dir.mkdir(parents=True, exist_ok=True)
        (self.sync_dir / "snapshots").mkdir(exist_ok=True)

        subprocess.run(
            ["git", "init", "-b", "main"],
            cwd=str(self.sync_dir), capture_output=True,
        )

        gitignore = self.sync_dir / ".gitignore"
        gitignore.write_text(".DS_Store\n")

        subprocess.run(
            ["git", "add", "."],
            cwd=str(self.sync_dir), capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initialize cursaves sync repo"],
            cwd=str(self.sync_dir), capture_output=True,
        )

        if remote:
            subprocess.run(
                ["git", "remote", "add", "origin", remote],
                cwd=str(self.sync_dir), capture_output=True,
            )

    def update_remote(self, remote: str):
        """Add or update the origin remote."""
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(self.sync_dir),
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            subprocess.run(
                ["git", "remote", "set-url", "origin", remote],
                cwd=str(self.sync_dir), capture_output=True,
            )
        else:
            subprocess.run(
                ["git", "remote", "add", "origin", remote],
                cwd=str(self.sync_dir), capture_output=True,
            )


# ── S3 backend ───────────────────────────────────────────────────────────


class S3Backend(SyncBackend):
    """Sync snapshots to/from an S3 bucket.

    Requires ``boto3`` — install with ``pip install cursaves[s3]``.

    Configuration (in ~/.config/cursaves/config.json)::

        {
            "backend": "s3",
            "s3": {
                "bucket": "my-cursor-saves",
                "prefix": "snapshots/",
                "region": "us-east-1"      // optional
            }
        }

    Authentication uses the standard AWS credential chain:
    env vars, ~/.aws/credentials, IAM roles, etc.
    """

    def __init__(self, bucket: str, prefix: str = "snapshots/", region: Optional[str] = None):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        self.region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError:
                print(
                    "Error: boto3 is required for S3 backend.\n"
                    "Install it with: pip install cursaves[s3]  or  pip install boto3",
                    file=sys.stderr,
                )
                sys.exit(1)
            kwargs = {}
            if self.region:
                kwargs["region_name"] = self.region
            self._client = boto3.client("s3", **kwargs)
        return self._client

    # -- SyncBackend interface ------------------------------------------

    def pull(self, snapshots_dir: Path) -> bool:
        """Download all remote snapshot files that are newer or missing locally."""
        client = self._get_client()
        try:
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)

            downloaded = 0
            for page in pages:
                for obj in page.get("Contents", []):
                    remote_key = obj["Key"]
                    rel_path = remote_key[len(self.prefix):]
                    if not rel_path:
                        continue

                    local_path = snapshots_dir / rel_path
                    remote_mtime = obj["LastModified"].timestamp()

                    if local_path.exists():
                        local_mtime = local_path.stat().st_mtime
                        local_size = local_path.stat().st_size
                        if local_size == obj["Size"] and local_mtime >= remote_mtime:
                            continue

                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    client.download_file(self.bucket, remote_key, str(local_path))
                    os.utime(str(local_path), (remote_mtime, remote_mtime))
                    downloaded += 1

            if downloaded:
                print(f"  Downloaded {downloaded} file(s) from s3://{self.bucket}")
            return True

        except Exception as e:
            print(f"S3 pull failed: {e}", file=sys.stderr)
            return False

    def push(
        self,
        snapshots_dir: Path,
        only_paths: Optional[Sequence[Path]] = None,
    ) -> bool:
        """Upload local snapshot files that are newer or missing remotely."""
        client = self._get_client()
        try:
            # Build index of remote objects
            remote_index: dict[str, tuple[float, int]] = {}
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)
            for page in pages:
                for obj in page.get("Contents", []):
                    rel = obj["Key"][len(self.prefix):]
                    if rel:
                        remote_index[rel] = (obj["LastModified"].timestamp(), obj["Size"])

            if only_paths is not None:
                candidates = expand_snapshot_export_paths(list(only_paths))
            else:
                candidates = [p for p in snapshots_dir.rglob("*") if p.is_file()]

            uploaded = 0
            uploaded_bytes = 0
            for local_path in candidates:
                if not local_path.is_file():
                    continue
                try:
                    rel = str(local_path.resolve().relative_to(snapshots_dir.resolve()))
                except ValueError:
                    continue
                remote_key = self.prefix + rel

                local_mtime = local_path.stat().st_mtime
                local_size = local_path.stat().st_size

                if rel in remote_index:
                    remote_mtime, remote_size = remote_index[rel]
                    if local_size == remote_size and local_mtime <= remote_mtime:
                        continue

                client.upload_file(str(local_path), self.bucket, remote_key)
                uploaded += 1
                uploaded_bytes += local_size

            if uploaded:
                print(
                    f"  Uploaded {uploaded} file(s) ({_format_bytes(uploaded_bytes)}) "
                    f"to s3://{self.bucket}"
                )
            return True

        except Exception as e:
            print(f"S3 push failed: {e}", file=sys.stderr)
            return False

    def has_remote(self) -> bool:
        return True  # S3 is always remote

    def is_initialized(self) -> bool:
        try:
            client = self._get_client()
            client.head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False


# ── Configuration ────────────────────────────────────────────────────────


def load_config() -> dict:
    """Load cursaves config from ~/.config/cursaves/config.json."""
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_config(config: dict):
    """Persist cursaves config."""
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")


def get_backend() -> SyncBackend:
    """Instantiate the configured sync backend.

    Falls back to GitBackend if nothing is configured (backward-compatible).
    """
    from . import paths

    config = load_config()
    backend_type = config.get("backend", "git")

    if backend_type == "s3":
        s3_cfg = config.get("s3", {})
        bucket = s3_cfg.get("bucket")
        if not bucket:
            print("Error: S3 backend configured but no bucket specified.", file=sys.stderr)
            print("Run: cursaves init --backend s3 --bucket <name>", file=sys.stderr)
            sys.exit(1)
        return S3Backend(
            bucket=bucket,
            prefix=s3_cfg.get("prefix", "snapshots/"),
            region=s3_cfg.get("region"),
        )

    # Default: git
    return GitBackend(paths.get_sync_dir())
