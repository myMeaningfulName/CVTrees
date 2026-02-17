"""
Logging, monitoring, and heartbeat module for GP experiments.

Provides:
1. Structured file+console logging with per-worker prefixes
2. Per-process RAM monitoring (RSS, VMS, percentage)
3. Heartbeat thread that periodically writes "alive" signals to a log file
4. Exception tracing helpers for multiprocessing workers

Usage:
    from src.gp_logging import setup_experiment_logging, get_logger, start_heartbeat, log_memory

    # At experiment start (main process)
    setup_experiment_logging("my_experiment", output_dir="experiments")
    logger = get_logger("main")
    start_heartbeat(interval=60)

    # In workers (call after fork)
    setup_worker_logging(worker_pid)
    logger = get_logger(f"worker.{worker_pid}")
"""

import os
import sys
import time
import logging
import threading
import traceback
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_log_dir: Optional[str] = None
_heartbeat_thread: Optional[threading.Thread] = None
_heartbeat_stop_event = threading.Event()
_experiment_start_time: Optional[float] = None

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)-25s | PID %(process)d | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_experiment_logging(
    experiment_name: str,
    output_dir: str = "experiments",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> str:
    """
    Configure logging for an experiment.

    Creates two handlers on the root logger:
      - Console handler at *console_level*
      - File handler (``experiment.log``) at *file_level*

    Returns:
        Path to the log file.
    """
    global _log_dir, _experiment_start_time

    _experiment_start_time = time.time()
    _log_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(_log_dir, exist_ok=True)

    log_file = os.path.join(_log_dir, "experiment.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove any existing handlers to avoid duplicate output when
    # setup is called more than once (e.g. in tests).
    root.handlers.clear()

    # --- File handler (detailed) ---
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(fh)

    # --- Console handler (concise) ---
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(ch)

    logger = logging.getLogger("gp.setup")
    logger.info("=" * 80)
    logger.info("EXPERIMENT LOGGING STARTED: %s", experiment_name)
    logger.info("Log file: %s", log_file)
    logger.info("PID: %d  |  Python: %s  |  Platform: %s",
                os.getpid(), sys.version.split()[0], sys.platform)
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        logger.info("System RAM: %.1f GB total, %.1f GB available (%.1f%% used)",
                     mem.total / (1024**3), mem.available / (1024**3), mem.percent)
    else:
        logger.warning("psutil not installed – RAM monitoring will be unavailable. "
                        "Install with: pip install psutil")
    logger.info("=" * 80)

    return log_file


def setup_worker_logging(worker_pid: Optional[int] = None):
    """
    Minimal logging setup inside a forked worker process.

    Workers inherit the root logger config from the parent (file handler path,
    formatters, etc.) via fork, so we only need to ensure the handlers are
    present and the process-specific fields are correct.

    Call this once at the top of ``_init_worker``.
    """
    pid = worker_pid or os.getpid()
    logger = logging.getLogger(f"gp.worker.{pid}")
    logger.info("Worker process initialised (PID %d)", pid)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger under the ``gp.*`` namespace.

    Examples::

        get_logger("main")          -> gp.main
        get_logger("worker.12345")  -> gp.worker.12345
        get_logger("evaluator")     -> gp.evaluator
    """
    return logging.getLogger(f"gp.{name}")


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------

@dataclass
class MemorySnapshot:
    """Point-in-time memory usage for a single process."""
    pid: int
    rss_mb: float          # Resident Set Size in MB
    vms_mb: float          # Virtual Memory Size in MB
    percent: float         # % of total system RAM
    timestamp: str         # ISO-8601
    label: str = ""        # e.g. "worker", "main"

    def to_dict(self) -> dict:
        return asdict(self)


def log_memory(label: str = "", logger: Optional[logging.Logger] = None) -> Optional[MemorySnapshot]:
    """
    Log the current process's memory usage and return a snapshot.

    If ``psutil`` is not installed the call is a no-op (returns ``None``).
    """
    if not HAS_PSUTIL:
        return None

    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    pct = proc.memory_percent()

    snap = MemorySnapshot(
        pid=os.getpid(),
        rss_mb=round(mem.rss / (1024 ** 2), 1),
        vms_mb=round(mem.vms / (1024 ** 2), 1),
        percent=round(pct, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
        label=label,
    )

    if logger is None:
        logger = get_logger("memory")

    logger.info(
        "MEMORY [%s] PID=%d  RSS=%.1f MB  VMS=%.1f MB  (%.2f%% of system)",
        label, snap.pid, snap.rss_mb, snap.vms_mb, snap.percent,
    )

    return snap


def log_all_workers_memory(worker_pids: list, label: str = "") -> list:
    """
    Log memory usage for a list of worker PIDs (called from the main process).

    Returns a list of MemorySnapshots for workers that are still alive.
    """
    if not HAS_PSUTIL:
        return []

    logger = get_logger("memory")
    snapshots = []

    for pid in worker_pids:
        try:
            proc = psutil.Process(pid)
            mem = proc.memory_info()
            pct = proc.memory_percent()
            snap = MemorySnapshot(
                pid=pid,
                rss_mb=round(mem.rss / (1024 ** 2), 1),
                vms_mb=round(mem.vms / (1024 ** 2), 1),
                percent=round(pct, 2),
                timestamp=datetime.now(timezone.utc).isoformat(),
                label=label,
            )
            snapshots.append(snap)
            logger.info(
                "MEMORY [%s] PID=%d  RSS=%.1f MB  VMS=%.1f MB  (%.2f%%)",
                label, pid, snap.rss_mb, snap.vms_mb, snap.percent,
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            logger.warning("MEMORY [%s] PID=%d  – process not found / access denied", label, pid)

    # Also log aggregate
    if snapshots:
        total_rss = sum(s.rss_mb for s in snapshots)
        logger.info(
            "MEMORY [%s] TOTAL across %d workers: RSS=%.1f MB",
            label, len(snapshots), total_rss,
        )

    return snapshots


def get_system_memory_info() -> Optional[Dict[str, Any]]:
    """Return a dict with system-wide memory stats, or None if psutil missing."""
    if not HAS_PSUTIL:
        return None
    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024**3), 2),
        "available_gb": round(mem.available / (1024**3), 2),
        "used_gb": round(mem.used / (1024**3), 2),
        "percent_used": mem.percent,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

def start_heartbeat(
    interval: int = 60,
    log_dir: Optional[str] = None,
    include_memory: bool = True,
) -> None:
    """
    Start a daemon thread that writes periodic heartbeat entries.

    The heartbeat writes to both:
      - The standard experiment log (via the ``gp.heartbeat`` logger)
      - A dedicated ``heartbeat.jsonl`` file (one JSON object per line)

    Args:
        interval: Seconds between heartbeats (default 60).
        log_dir: Directory for ``heartbeat.jsonl``. Defaults to the
                 experiment log directory set by ``setup_experiment_logging``.
        include_memory: Whether to include memory stats in each heartbeat.
    """
    global _heartbeat_thread, _heartbeat_stop_event

    # Stop any existing heartbeat
    stop_heartbeat()

    target_dir = log_dir or _log_dir
    if target_dir is None:
        target_dir = "."

    _heartbeat_stop_event = threading.Event()

    def _heartbeat_loop():
        hb_logger = get_logger("heartbeat")
        hb_file = os.path.join(target_dir, "heartbeat.jsonl")
        beat_count = 0

        while not _heartbeat_stop_event.is_set():
            _heartbeat_stop_event.wait(timeout=interval)
            if _heartbeat_stop_event.is_set():
                break

            beat_count += 1
            now = datetime.now(timezone.utc).isoformat()
            uptime = time.time() - (_experiment_start_time or time.time())

            entry: Dict[str, Any] = {
                "beat": beat_count,
                "timestamp": now,
                "uptime_seconds": round(uptime, 1),
                "pid": os.getpid(),
                "alive": True,
            }

            if include_memory:
                sys_mem = get_system_memory_info()
                if sys_mem:
                    entry["system_memory"] = sys_mem

                snap = log_memory(label="heartbeat", logger=hb_logger)
                if snap:
                    entry["main_process_memory"] = snap.to_dict()

            hb_logger.info(
                "HEARTBEAT #%d  |  uptime=%.0fs  |  alive=True",
                beat_count, uptime,
            )

            # Append to JSONL file
            try:
                with open(hb_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as exc:
                hb_logger.warning("Failed to write heartbeat file: %s", exc)

    _heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True,
                                          name="gp-heartbeat")
    _heartbeat_thread.start()

    get_logger("heartbeat").info(
        "Heartbeat started (interval=%ds, file=%s/heartbeat.jsonl)",
        interval, target_dir,
    )


def stop_heartbeat():
    """Signal the heartbeat thread to stop."""
    global _heartbeat_thread
    _heartbeat_stop_event.set()
    if _heartbeat_thread is not None and _heartbeat_thread.is_alive():
        _heartbeat_thread.join(timeout=5)
        _heartbeat_thread = None
    _heartbeat_stop_event.clear()


# ---------------------------------------------------------------------------
# Exception helpers
# ---------------------------------------------------------------------------

def log_exception(
    logger: logging.Logger,
    msg: str,
    exc: Exception,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Log an exception with full traceback and optional structured context.

    Returns the formatted traceback string (useful for serialisation in
    ``EvaluationResult.error_msg``).
    """
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tb_str = "".join(tb)

    logger.error(
        "%s  |  %s: %s",
        msg, type(exc).__name__, exc,
    )
    logger.debug("Full traceback:\n%s", tb_str)

    if extra:
        logger.debug("Context: %s", json.dumps(extra, default=str))

    return tb_str


def log_worker_exception(
    exc: Exception,
    phase: str = "",
    individual_idx: int = -1,
    expr_str: str = "",
) -> str:
    """
    Convenience wrapper for exceptions inside worker processes.

    Args:
        exc: The caught exception.
        phase: E.g. "TRAIN", "VAL", "TEST", "COMPILE".
        individual_idx: Index of the individual being evaluated.
        expr_str: The S-expression string of the individual.

    Returns:
        Formatted traceback string.
    """
    pid = os.getpid()
    logger = get_logger(f"worker.{pid}")

    extra = {
        "phase": phase,
        "individual_idx": individual_idx,
        "expression_preview": expr_str[:200] if expr_str else "",
        "pid": pid,
    }

    # Also log memory at exception time – OOM is a likely culprit
    snap = log_memory(label=f"exception-{phase}", logger=logger)
    if snap:
        extra["memory_at_exception"] = snap.to_dict()

    return log_exception(
        logger,
        f"[WORKER {pid}] Exception in {phase} for individual {individual_idx}",
        exc,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

class TimingContext:
    """
    Simple context manager that logs elapsed wall-clock time.

    Usage::

        with TimingContext("Generation 0 evaluation", logger):
            evaluate(...)
    """

    def __init__(self, description: str, logger: Optional[logging.Logger] = None):
        self.description = description
        self.logger = logger or get_logger("timing")
        self.start: float = 0
        self.elapsed: float = 0

    def __enter__(self):
        self.start = time.time()
        self.logger.info("TIMER START: %s", self.description)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start
        if exc_type is not None:
            self.logger.warning(
                "TIMER END (with error): %s  |  %.2fs elapsed  |  %s: %s",
                self.description, self.elapsed, exc_type.__name__, exc_val,
            )
        else:
            self.logger.info(
                "TIMER END: %s  |  %.2fs elapsed",
                self.description, self.elapsed,
            )
        return False  # Don't suppress exceptions
