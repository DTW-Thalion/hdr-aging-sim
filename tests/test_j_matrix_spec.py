"""Tests for JMatrixSpec dataclass and provenance helpers."""

import json
from pathlib import Path

import pytest

from hdr_sim.j_matrix_spec import JMatrixSpec, load_default_spec, load_provenance_spec

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_9x9 = REPO_ROOT / "data" / "J_matrix_compiled_9x9.csv"
CSV_8x8 = REPO_ROOT / "data" / "J_matrix_compiled.csv"


# ------------------------------------------------------------------
# 9x9 spec
# ------------------------------------------------------------------

class TestFromCsv9x9:
    def test_n_axes(self):
        spec = JMatrixSpec.from_csv(CSV_9x9)
        assert spec.n_axes == 9

    def test_sign_counts(self):
        spec = JMatrixSpec.from_csv(CSV_9x9)
        assert spec.sign_counts == {"positive": 57, "negative": 11, "unknown": 4}

    def test_axes_sorted(self):
        spec = JMatrixSpec.from_csv(CSV_9x9)
        assert spec.axes == tuple(sorted(spec.axes))


# ------------------------------------------------------------------
# 8x8 spec (legacy)
# ------------------------------------------------------------------

@pytest.mark.skipif(not CSV_8x8.exists(), reason="Legacy 8x8 CSV not present")
class TestFromCsv8x8:
    def test_n_axes(self):
        spec = JMatrixSpec.from_csv(CSV_8x8)
        assert spec.n_axes == 8


# ------------------------------------------------------------------
# Serialisation round-trip
# ------------------------------------------------------------------

class TestToDict:
    def test_roundtrip_json(self):
        spec = JMatrixSpec.from_csv(CSV_9x9)
        d = spec.to_dict()
        serialised = json.dumps(d)
        restored = json.loads(serialised)
        assert restored["sha256"] == spec.sha256
        assert restored["n_axes"] == spec.n_axes
        assert restored["sign_counts"] == spec.sign_counts
        assert restored["axes"] == list(spec.axes)


# ------------------------------------------------------------------
# Validate against
# ------------------------------------------------------------------

@pytest.mark.skipif(not CSV_8x8.exists(), reason="Legacy 8x8 CSV not present")
class TestValidateAgainst:
    def test_detects_differences(self):
        spec_9 = JMatrixSpec.from_csv(CSV_9x9)
        spec_8 = JMatrixSpec.from_csv(CSV_8x8)
        diffs = spec_9.validate_against(spec_8)
        assert len(diffs) > 0
        assert any("n_axes" in d for d in diffs)
        assert any("sign_counts" in d for d in diffs)


# ------------------------------------------------------------------
# SHA-256 determinism
# ------------------------------------------------------------------

class TestSha256Determinism:
    def test_same_file_same_hash(self):
        spec_a = JMatrixSpec.from_csv(CSV_9x9)
        spec_b = JMatrixSpec.from_csv(CSV_9x9)
        assert spec_a.sha256 == spec_b.sha256


# ------------------------------------------------------------------
# Module-level loaders
# ------------------------------------------------------------------

class TestLoaders:
    def test_load_default_spec(self):
        spec = load_default_spec()
        assert spec.n_axes == 9

    def test_load_provenance_spec(self):
        spec = load_provenance_spec()
        assert spec.version == "R6-ontology-v1.6"
        assert spec.n_axes == 9
