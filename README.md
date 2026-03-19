# HDR Aging Dynamics Toy Simulation

Numerical simulation of the Homeodynamic Remediation (HDR) framework's aging dynamics,
demonstrating how age-related parameter degradation produces critical slowing-down,
increased vulnerability to perturbation, and basin transitions.

Companion code for: White (2026), "Aging as Controller Failure: A Formal Ontology
for Multi-Axis Physiological Decline."

## Key result

The simulation shows that a 4-axis linear dynamical system with biologically parameterised
recovery times (τ_i) and inter-axis coupling (J), when degraded according to known
age-related trajectories, produces:

- Progressive drift of the spectral abscissa toward instability (α → 0⁻)
- Critical slowing-down: recovery timescale diverges as α → 0
- Identical perturbations produce age-dependent responses (fast recovery at 30,
  near-catastrophic excursion at 80)
- Cross-axis propagation of perturbations through the coupling matrix

These are the dynamical signatures of aging predicted by the HDR formalism and
independently observed by Scheffer et al. (2018, PNAS) and Pyrkov et al. (2021,
Nat Commun).

## Setup

```bash
pip install numpy scipy matplotlib
```

## Usage

```bash
python scripts/run_figure2b.py        # Main figure: 4-panel aging demo
python scripts/run_figure_frailty.py   # Frailty perturbation-response
python scripts/run_figure_t2d.py       # T2D phase portrait
```

Figures are saved to `outputs/`.
