from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


@lru_cache
def load(name: str) -> dict:
    """Load configs/<name>.yaml."""
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    return Path(load("paths")[key])
