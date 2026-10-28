"""跨进程重计算锁：确保同一时刻只有一个 heavy 任务在跑。

Windows 用 msvcrt.locking，Linux CI 用 fcntl。
锁文件放在 data root 下，包含 PID、启动时间、job_id。
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional


class LockOwner:
    """锁持有者信息。"""
    def __init__(self, pid: int, started_at: str, job_id: str):
        self.pid = pid
        self.started_at = started_at
        self.job_id = job_id


@contextmanager
def acquire_heavy_lock(
    lock_file: Path,
    job_id: str,
    timeout: float = 5.0,
):
    """获取跨进程 heavy 锁。

    Args:
        lock_file: 锁文件路径（通常在 data root 下）
        job_id: 当前任务 ID
        timeout: 等待锁的最大秒数

    Yields:
        LockOwner 信息
    """
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    info = LockOwner(pid=pid, started_at=datetime.now().isoformat(), job_id=job_id)

    start = time.time()
    acquired = False

    try:
        while time.time() - start < timeout:
            # 检查是否有陈旧锁
            if lock_file.exists():
                try:
                    old_info = json.loads(lock_file.read_text())
                    old_pid = old_info.get("pid", 0)
                    # 检查进程是否还活着
                    if not _pid_alive(old_pid):
                        # 陈旧锁，删除
                        try:
                            lock_file.unlink()
                        except OSError:
                            pass
                    else:
                        # 锁被持有，等待
                        time.sleep(0.5)
                        continue
                except (json.JSONDecodeError, OSError):
                    # 锁文件损坏，删除
                    try:
                        lock_file.unlink()
                    except OSError:
                        pass
                    continue

            # 尝试获取锁
            try:
                fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, json.dumps({
                    "pid": pid,
                    "started_at": info.started_at,
                    "job_id": job_id,
                }).encode())
                os.close(fd)
                acquired = True
                break
            except FileExistsError:
                time.sleep(0.3)
                continue

        if not acquired:
            # 读取当前锁持有者信息
            try:
                current = json.loads(lock_file.read_text())
                raise RuntimeError(
                    f"Heavy lock held by PID {current.get('pid')} "
                    f"(job {current.get('job_id')}) for {current.get('started_at')}"
                )
            except (json.JSONDecodeError, OSError):
                raise RuntimeError("Could not acquire heavy lock")

        yield info

    finally:
        if acquired:
            try:
                lock_file.unlink()
            except OSError:
                pass


def _pid_alive(pid: int) -> bool:
    """检查 PID 是否还在运行。"""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (OSError, Exception):
        return False
