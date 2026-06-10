"""Atomic JSON persistence helpers."""

import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any


def atomic_write_json(path: str, data: Any) -> None:
    """Write JSON with a same-directory temp file and atomic replace."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(path):
            os.replace(path, f"{path}.bak")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def load_json(path: str, default: Any) -> Any:
    """Load JSON, quarantining a corrupt primary file and falling back to .bak."""
    for candidate, is_backup in ((path, False), (f"{path}.bak", True)):
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
            if is_backup:
                print(f"[storage] 主文件不可用，已从备份恢复: {candidate}")
            return data
        except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
            if not is_backup:
                _quarantine(path, e)
    print(f"[storage] 警告: {path} 及备份均不可用，使用默认值")
    return default


def _quarantine(path: str, error: Exception) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dst = f"{path}.corrupt.{ts}"
    try:
        shutil.copy2(path, dst)
        print(f"[storage] 损坏文件 {path} ({error})，已备份现场到 {dst}")
    except OSError as copy_err:
        print(f"[storage] 损坏文件 {path} 备份失败: {copy_err}")
