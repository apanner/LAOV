"""Colab / Desk lane — stage status on Drive + flushed console lines.

Writes ``MyDrive/VDA_Jobs/status/{job_id}_status.json`` so the Desk job monitor
(or manual Drive check) can see batch progress. Always prints ``[STAGE]`` lines
with flush so Colab notebooks update during long SAM3 / BiRefNet runs.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger("colab_run_status")

ProgressCallback = Callable[[float, str], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ColabRunStatus:
    """Drive-backed batch status + console stage lines."""

    def __init__(
        self,
        mount: Path,
        job_id: str,
        *,
        batch_id: str | None = None,
        job_name: str | None = None,
        total_sequences: int = 1,
        sequences: list[dict[str, Any]] | None = None,
    ) -> None:
        self.mount = mount
        self.job_id = job_id
        self.path = (mount / "VDA_Jobs" / "status" / f"{job_id}_status.json").resolve()
        seq_rows: list[dict[str, Any]] = []
        if sequences:
            for seq in sequences:
                seq_rows.append(
                    {
                        "shot_name": str(seq.get("shot_name", "shot")),
                        "status": "pending",
                        "stage": "",
                        "progress_percent": 0.0,
                    }
                )
        self._data: dict[str, Any] = {
            "job_id": job_id,
            "batch_id": batch_id or job_id,
            "job_name": job_name or job_id,
            "status": "pending",
            "stage": "",
            "message": "",
            "progress_percent": 0.0,
            "updated_at": _utc_now(),
            "started_at": _utc_now(),
            "total_sequences": max(1, total_sequences),
            "successful_sequences": 0,
            "failed_sequences": 0,
            "sequences": seq_rows,
        }

    def begin_run(self, message: str = "Starting LAOV batch") -> None:
        self._data["status"] = "running"
        self.stage(message, percent=0.0)

    def stage(
        self,
        message: str,
        *,
        percent: float | None = None,
        shot_name: str | None = None,
    ) -> None:
        if percent is not None:
            pct = max(0.0, min(100.0, float(percent)))
            print(f"[STAGE] [{pct:5.1f}%] {message}", flush=True)
            self._data["progress_percent"] = pct
        else:
            print(f"[STAGE] {message}", flush=True)
        self._data["stage"] = message
        self._data["message"] = message
        self._data["updated_at"] = _utc_now()
        if shot_name:
            row = self._sequence_row(shot_name)
            if row is not None:
                row["stage"] = message
                if percent is not None:
                    row["progress_percent"] = self._data["progress_percent"]
        _log.info("%s", message)
        self._write()

    def shot_begin(self, shot_name: str, index: int, total: int) -> None:
        row = self._sequence_row(shot_name)
        if row is not None:
            row["status"] = "processing"
            row["stage"] = "shot start"
            row["progress_percent"] = 0.0
        self.stage(f"Shot {index}/{total}: {shot_name} — preflight & setup", shot_name=shot_name)

    def shot_end(self, shot_name: str, *, ok: bool, message: str = "") -> None:
        row = self._sequence_row(shot_name)
        if row is not None:
            row["status"] = "completed" if ok else "failed"
            row["stage"] = message or ("done" if ok else "failed")
            row["progress_percent"] = 100.0 if ok else row.get("progress_percent", 0.0)
        if ok:
            self._data["successful_sequences"] = int(self._data.get("successful_sequences", 0)) + 1
        else:
            self._data["failed_sequences"] = int(self._data.get("failed_sequences", 0)) + 1
        label = message or ("done" if ok else "failed")
        self.stage(f"Shot {shot_name}: {label}", shot_name=shot_name)

    def finish_run(self, *, ok: bool, message: str = "") -> None:
        failed = int(self._data.get("failed_sequences", 0))
        total = int(self._data.get("total_sequences", 1))
        success = int(self._data.get("successful_sequences", 0))
        if ok and failed == 0 and success >= total:
            self._data["status"] = "completed"
        elif success > 0:
            self._data["status"] = "partial"
        else:
            self._data["status"] = "failed"
        self._data["completed_at"] = _utc_now()
        final = message or (
            f"Batch finished — {success}/{total} shot(s) OK"
            if ok
            else f"Batch failed — {success}/{total} shot(s) OK, {failed} failed"
        )
        self.stage(final, percent=100.0 if ok else self._data.get("progress_percent", 0.0))

    def make_laov_callback(
        self,
        shot_name: str,
        shot_index: int,
        shot_total: int,
    ) -> ProgressCallback:
        """Map LocalExecutor 0..1 progress into this batch + shot."""

        base = shot_index / max(1, shot_total)
        span = 1.0 / max(1, shot_total)

        def report(fraction: float, label: str) -> None:
            overall = (base + float(fraction) * span) * 100.0
            self.stage(f"{shot_name}: {label}", percent=overall, shot_name=shot_name)

        return report

    def status_path_hint(self) -> str:
        try:
            rel = self.path.relative_to(self.mount)
            return str(rel).replace("\\", "/")
        except ValueError:
            return str(self.path)

    def _sequence_row(self, shot_name: str) -> dict[str, Any] | None:
        for row in self._data.get("sequences") or []:
            if str(row.get("shot_name")) == shot_name:
                return row
        return None

    def _write(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            _log.warning("Could not write status file %s: %s", self.path, exc)


def reporter_from_job_json(mount: Path, cfg: dict) -> ColabRunStatus | None:
    job_id = str(cfg.get("job_id") or cfg.get("batch_id") or "").strip()
    if not job_id:
        return None
    sequences = cfg.get("sequences") or []
    return ColabRunStatus(
        mount,
        job_id,
        batch_id=str(cfg.get("batch_id") or job_id),
        job_name=str(cfg.get("job_name") or job_id),
        total_sequences=len(sequences) or 1,
        sequences=sequences,
    )


def configure_flushed_logging() -> None:
    """Ensure log lines appear immediately in Colab notebook output."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for handler in root.handlers:
        if not isinstance(handler, logging.StreamHandler):
            continue
        if getattr(handler, "_laov_flush_emit", False):
            continue
        original_emit = handler.emit

        def emit(record: logging.LogRecord, *, _orig=original_emit, _h=handler) -> None:
            _orig(record)
            if hasattr(_h, "flush"):
                _h.flush()

        handler.emit = emit  # type: ignore[method-assign]
        handler._laov_flush_emit = True  # type: ignore[attr-defined]
