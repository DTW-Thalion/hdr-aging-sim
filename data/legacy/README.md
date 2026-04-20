# Legacy Data Files

These files are archived for reproducibility of pre-v2.5 analyses.
They are no longer used by the active codebase.

- `J_matrix_compiled_8x8.csv` — Original 8-axis J-matrix (56 off-diagonal entries,
  42 nonzero). Used by `configure()` in versions prior to v2.5. Replaced by
  `data/J_matrix_compiled_9x9.csv` (72 off-diagonal entries, 68 nonzero with
  qual_only imputation).

As of v2.5, all code uses the 9x9 compiled CSV via `configure()` which
delegates to `configure_v2()`.
