"""Minimal Google Cloud Storage client for PhysioNet's requester-pays buckets.

Auth reuses the gcloud login (scripts/env/gcloud.sh); every request names the billing project.
"""

import hashlib
import os
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import quote

import requests

API = "https://storage.googleapis.com/storage/v1"
# 403 is included because PhysioNet's bucket answers bursts of requests with transient 403s
RETRY_STATUS = {403, 429, 500, 502, 503, 504}
GCLOUD_ENV = {
    "CLOUDSDK_CONFIG": "/fs/ess/PAS2836/mugezhang/.gcloud",
    "PATH": "/fs/scratch/PAS2836/mugezhang/tools/google-cloud-sdk/bin:" + os.environ.get("PATH", ""),
}


class Client:
    def __init__(self, bucket: str, billing_project: str, token_ttl: float = 1800):
        self.bucket, self.project, self.ttl = bucket, billing_project, token_ttl
        self._token, self._t, self._lock = None, 0.0, threading.Lock()
        self._local = threading.local()

    def _auth(self) -> dict:
        with self._lock:
            if self._token is None or time.time() - self._t > self.ttl:
                self._token = subprocess.run(
                    ["gcloud", "auth", "print-access-token"], env={**os.environ, **GCLOUD_ENV},
                    check=True, capture_output=True, text=True,
                ).stdout.strip()
                self._t = time.time()
            return {"Authorization": f"Bearer {self._token}"}

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "s"):
            self._local.s = requests.Session()
        return self._local.s

    def _get_json(self, url: str, params: dict, retries: int = 6) -> dict:
        for attempt in range(retries):
            r = self._session().get(url, params=params, headers=self._auth(), timeout=120)
            if r.status_code in RETRY_STATUS and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise AssertionError("unreachable")

    def list(self, prefix: str):
        """Yield (name, size_bytes) for every object under prefix."""
        params = {"prefix": prefix, "fields": "items(name,size),nextPageToken",
                  "maxResults": 1000, "userProject": self.project}
        while True:
            j = self._get_json(f"{API}/b/{self.bucket}/o", params)
            for it in j.get("items", []):
                yield it["name"], int(it["size"])
            if "nextPageToken" not in j:
                return
            params["pageToken"] = j["nextPageToken"]

    def download(self, name: str, dst: Path, expected_sha256: str, retries: int = 5) -> int:
        """Stream one object to dst, verifying SHA-256 before the atomic rename. Returns bytes written."""
        url = f"{API}/b/{self.bucket}/o/{quote(name, safe='')}"
        params = {"alt": "media", "userProject": self.project}
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.part{os.getpid()}_{threading.get_ident()}")
        for attempt in range(retries):
            try:
                h, n = hashlib.sha256(), 0
                with self._session().get(url, params=params, headers=self._auth(), stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                            h.update(chunk)
                            n += len(chunk)
                if h.hexdigest() != expected_sha256:
                    raise IOError(f"checksum mismatch for {name}")
                os.replace(tmp, dst)
                return n
            except (requests.RequestException, IOError):
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
            finally:
                if tmp.exists():
                    tmp.unlink()
        raise AssertionError("unreachable")
