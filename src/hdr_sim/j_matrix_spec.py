"""J-matrix provenance specification and validation."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class JMatrixSpec:
    """Immutable specification capturing the identity and shape of a J-matrix CSV."""

    csv_path: str
    version: str
    source_repo: str
    sha256: str
    n_axes: int
    sign_counts: dict
    axes: tuple[str, ...]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        version: str | None = None,
        source_repo: str | None = None,
    ) -> JMatrixSpec:
        """Build a JMatrixSpec by inspecting a J-matrix CSV on disk."""
        csv_path = Path(csv_path).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"J-matrix CSV not found: {csv_path}")

        # SHA-256
        raw = csv_path.read_bytes()
        sha256 = hashlib.sha256(raw).hexdigest()

        # Parse rows
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        # Axes
        axis_set: set[str] = set()
        for row in rows:
            axis_set.add(row["axis_from"].strip())
            axis_set.add(row["axis_to"].strip())
        axes = tuple(sorted(axis_set))

        # Sign counts
        positive = 0
        negative = 0
        unknown = 0
        for row in rows:
            sign = row["sign"].strip().lower()
            if sign in ("+", "positive"):
                positive += 1
            elif sign in ("-", "negative"):
                negative += 1
            else:
                unknown += 1
        sign_counts = {"positive": positive, "negative": negative, "unknown": unknown}

        # Auto-version from filename if not provided
        if version is None:
            version = csv_path.stem

        if source_repo is None:
            source_repo = "hdr-aging-sim"

        return cls(
            csv_path=str(csv_path),
            version=version,
            source_repo=source_repo,
            sha256=sha256,
            n_axes=len(axes),
            sign_counts=sign_counts,
            axes=axes,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary."""
        return {
            "csv_path": self.csv_path,
            "version": self.version,
            "source_repo": self.source_repo,
            "sha256": self.sha256,
            "n_axes": self.n_axes,
            "sign_counts": dict(self.sign_counts),
            "axes": list(self.axes),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_against(self, other: JMatrixSpec) -> list[str]:
        """Return human-readable descriptions of every difference vs *other*."""
        diffs: list[str] = []

        if self.sha256 != other.sha256:
            diffs.append(
                f"sha256 differs: self={self.sha256[:12]}..., "
                f"other={other.sha256[:12]}..."
            )

        if self.n_axes != other.n_axes:
            diffs.append(
                f"n_axes differs: self={self.n_axes}, other={other.n_axes}"
            )

        if self.axes != other.axes:
            diffs.append(
                f"axes differ: self={list(self.axes)}, other={list(other.axes)}"
            )

        if self.sign_counts != other.sign_counts:
            def _fmt(sc: dict) -> str:
                return f"{sc['positive']}+/{sc['negative']}-/{sc['unknown']}?"
            diffs.append(
                f"sign_counts differ: self has {_fmt(self.sign_counts)}, "
                f"other has {_fmt(other.sign_counts)}"
            )

        if self.version != other.version:
            diffs.append(
                f"version differs: self={self.version!r}, other={other.version!r}"
            )

        if self.source_repo != other.source_repo:
            diffs.append(
                f"source_repo differs: self={self.source_repo!r}, "
                f"other={other.source_repo!r}"
            )

        return diffs


# ======================================================================
# Module-level loaders
# ======================================================================

def _repo_root() -> Path:
    """Walk up from this file to find the repository root (contains pyproject.toml)."""
    d = Path(__file__).resolve().parent
    for _ in range(10):
        if (d / "pyproject.toml").exists():
            return d
        d = d.parent
    raise RuntimeError("Cannot locate repository root from j_matrix_spec.py")


def load_provenance_spec() -> JMatrixSpec:
    """Load the frozen provenance J-matrix (R6 ontology v1.6)."""
    path = _repo_root() / "data" / "provenance" / "J_R6_ontology_v1.6.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Provenance J-matrix not found at {path}. "
            "Ensure data/provenance/J_R6_ontology_v1.6.csv exists."
        )
    return JMatrixSpec.from_csv(path, version="R6-ontology-v1.6", source_repo="hdr-aging-sim/provenance")


def load_default_spec() -> JMatrixSpec:
    """Load the current default compiled J-matrix."""
    path = _repo_root() / "data" / "J_matrix_compiled_9x9.csv"
    return JMatrixSpec.from_csv(path)
