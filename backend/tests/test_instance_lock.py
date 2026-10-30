"""One Forge server per data directory (forge.instance_lock)."""
from __future__ import annotations

import subprocess
import sys

import pytest

from forge import instance_lock


def test_second_holder_is_refused_with_the_first_holders_pid(tmp_path):
    f = instance_lock.acquire(tmp_path, 8790)
    try:
        code = ("import sys; from pathlib import Path; from forge import instance_lock as il\n"
                "try:\n    il.acquire(Path(sys.argv[1]), 8765)\nexcept il.InstanceLockError as e:\n"
                "    print(e); sys.exit(3)\n")
        p = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True)
        assert p.returncode == 3, p.stderr
        assert "port 8790" in p.stdout and "another Forge server" in p.stdout
    finally:
        f.close()


def test_lock_is_released_when_the_holder_closes(tmp_path):
    instance_lock.acquire(tmp_path, 1).close()
    g = instance_lock.acquire(tmp_path, 2)
    g.close()


def test_same_process_second_acquire_refused(tmp_path):
    f = instance_lock.acquire(tmp_path, 1)
    try:
        with pytest.raises(instance_lock.InstanceLockError):
            instance_lock.acquire(tmp_path, 2)
    finally:
        f.close()
