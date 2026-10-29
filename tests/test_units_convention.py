"""Tests for the Task 1.1 units decision (docs/units_convention.md).

These lock in the three claims the decision rests on:

1. For a non-negative gain matrix G, A = -D(I - G) is Hurwitz iff rho(G) < 1,
   for any positive diagonal D. This is the whole point of Convention G: the
   stability condition is tau-free.
2. The same equivalence FAILS once G carries negative entries, which is why the
   protective couplings have to move to the damping modulation before the gain
   condition can be used (docs/protective_coupling_formulation.md).
3. The rate and gain forms agree: -D + J == -D(I - G) when J = D G.

Plus the headline figures, so a silent change to the compiled CSV or the tau
registry fails a test rather than quietly rewriting the decision.
"""

import json
import os

import numpy as np
import pytest

from hdr_sim.csv_loader import load_J_csv, build_J_basin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL9 = ("I", "M", "E", "mito", "P", "C", "N", "F", "B")
FAST7 = ("I", "M", "mito", "P", "C", "N", "F")


def alpha(A):
    return float(np.max(np.real(np.linalg.eigvals(A))))


def rho(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def A_from_G(G, tau):
    return -np.diag(1.0 / tau) @ (np.eye(len(tau)) - G)


def J_from_G(G, tau):
    return G / tau[:, None]


# ---------------------------------------------------------------------------
# 1 & 2: the stability equivalence, and where it breaks
# ---------------------------------------------------------------------------

def _random_system(rng, n, signed):
    scale = rng.choice([0.2, 0.6, 1.2, 2.0])
    G = (rng.random((n, n)) - (0.4 if signed else 0.0)) * scale
    G *= rng.random((n, n)) > 0.35
    np.fill_diagonal(G, 0.0)
    tau = 10 ** rng.uniform(-3, 3, size=n)   # 6 orders of magnitude
    return G, tau


def test_hurwitz_iff_spectral_radius_for_nonnegative_G():
    """Convention G's stability condition is exact and tau-free for G >= 0."""
    rng = np.random.default_rng(7)
    tested = 0
    for _ in range(5000):
        G, tau = _random_system(rng, int(rng.integers(2, 8)), signed=False)
        r = rho(G)
        if abs(r - 1.0) < 1e-6:
            continue            # boundary case, not informative
        tested += 1
        assert (alpha(A_from_G(G, tau)) < 0) == (r < 1.0), (
            f"rho(G) = {r}, tau = {tau}")
    assert tested > 4000


def test_equivalence_fails_for_signed_G():
    """With negative entries the gain condition is not a stability condition.

    This is the formal reason the protective F- and B-row entries must move to
    the damping modulation rather than staying in G.
    """
    rng = np.random.default_rng(11)
    tested = violations = 0
    for _ in range(5000):
        G, tau = _random_system(rng, int(rng.integers(2, 8)), signed=True)
        if (G >= 0).all():
            continue
        r = rho(G)
        if abs(r - 1.0) < 1e-6:
            continue
        tested += 1
        if (alpha(A_from_G(G, tau)) < 0) != (r < 1.0):
            violations += 1
    assert tested > 3000
    assert violations / tested > 0.02, (
        "expected the equivalence to fail materially for signed G")


# ---------------------------------------------------------------------------
# 3: the two forms are the same system
# ---------------------------------------------------------------------------

def test_rate_and_gain_forms_agree():
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(2, 9))
        G, tau = _random_system(rng, n, signed=False)
        np.testing.assert_allclose(
            A_from_G(G, tau),
            -np.diag(1.0 / tau) + J_from_G(G, tau),
            rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# headline figures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rows():
    return load_J_csv(os.path.join(ROOT, "data", "J_matrix_compiled_9x9.csv"))


def _fast(rows, basin):
    M = build_J_basin(rows, basin, ALL9)
    idx = [ALL9.index(a) for a in FAST7]
    M = M[np.ix_(idx, idx)].copy()
    np.fill_diagonal(M, 0.0)
    return M


@pytest.mark.parametrize("basin,expected,n_nonzero", [
    ("healthy", 0.749, 31),
    ("pre_disease", 0.969, 31),
    ("disease", 1.311, 33),
])
def test_rho_G_fast_per_basin(rows, basin, expected, n_nonzero):
    G = np.abs(_fast(rows, basin))
    assert rho(G) == pytest.approx(expected, abs=0.001)
    assert int((G > 0).sum()) == n_nonzero


def test_protective_entries_are_exactly_the_documented_eleven(rows):
    """8 F-row + 3 B-row, and nothing else, carries a negative sign."""
    neg = sorted((r["axis_from"].strip(), r["axis_to"].strip())
                 for r in rows if r["sign"].strip() == "-")
    expected = sorted(
        [("F", t) for t in ("I", "M", "E", "mito", "P", "C", "N", "B")] +
        [("B", t) for t in ("I", "M", "N")])
    assert neg == expected


def test_G_is_all_positive_once_protective_sources_are_removed(rows):
    for basin in ("healthy", "pre_disease", "disease"):
        G = _fast(rows, basin)
        G[:, FAST7.index("F")] = 0.0
        assert (G >= 0).all(), f"{basin} still has negative entries"


def test_tau_registry_covers_every_axis_with_anchors_and_pmids():
    with open(os.path.join(ROOT, "data", "tau_registry.json")) as f:
        reg = json.load(f)
    assert set(reg["axes"]) == set(ALL9)
    for name, e in reg["axes"].items():
        for k in ("tau25", "tau80", "tau120", "range", "interpretation",
                  "pmids", "dynamical_role"):
            assert k in e, f"{name} missing {k}"
        assert e["tau25"] <= e["tau80"] <= e["tau120"], name
        assert e["pmids"], f"{name} has no PMID"
        assert e["dynamical_role"] in ("fast", "quasi_static")
    assert set(reg["subsystem_partition"]["fast"]) == set(FAST7)
    # N is the one axis recorded with two sub-channels
    assert set(reg["axes"]["N"]["sub_channels"]) == {"vagal", "hpa"}
    assert reg["axes"]["N"]["fast_subsystem_uses"] == "vagal"


def test_phi_table_is_consumable_as_a_gain_calibration():
    path = os.path.join(ROOT, "results", "lead_lag_phi_table.json")
    with open(path) as f:
        tbl = json.load(f)
    assert len(tbl["pairs"]) == 38
    for p in tbl["pairs"]:
        assert p["beta"] is not None
        assert p["phi_jj"] is not None
        assert p["interval_years_mean"] > 0
        if p["gamma"] < 0:
            # long_run_gain = beta / (1 - phi_jj) = -beta / gamma.
            # Tolerance is 1e-4 rather than machine precision because the
            # InCHIANTI rows are stored to 6 significant figures (see
            # provenance.rounding in the file).
            assert p["long_run_gain"] == pytest.approx(
                p["beta"] / (1.0 - p["phi_jj"]), rel=1e-4)
            assert p["long_run_gain"] == pytest.approx(
                -p["beta"] / p["gamma"], rel=1e-4)
