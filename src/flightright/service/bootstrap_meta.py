from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

import boto3


def _env(key: str, default: str | None = None) -> str | None:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def _s3_client():
    # Uses AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION automatically.
    # Optional endpoint override for S3-compatible stores (e.g. iDrive E2).
    endpoint = _env("FLIGHTRIGHT_S3_ENDPOINT") or _env("E2_ENDPOINT")
    return boto3.client("s3", endpoint_url=endpoint)


def ensure_meta_files(
    *,
    bucket: str,
    downloads: Iterable[Tuple[str, Path]],
) -> None:
    """
    Ensures each (object_key -> local_path) exists. If missing, downloads from S3.
    """
    s3 = _s3_client()
    for object_key, local_path in downloads:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists() and local_path.stat().st_size > 0:
            continue

        tmp = local_path.with_suffix(local_path.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()

        s3.download_file(bucket, object_key, str(tmp))
        tmp.replace(local_path)