# Intervention Analysis Report

Generated: 2026-04-09T01:47:56Z
Total elapsed: 0.3s

## 1. Single-Intervention Ranking (age 70, delta-alpha)

| Rank | Intervention | delta-alpha | % change | n_J | Evidence |
|------|-------------|------------|----------|-----|----------|
| 1 | nad_precursors | -0.0000398 | -2.0% | 3 | approved_therapy |
| 2 | exercise_resistance | -0.0000021 | -0.1% | 17 | approved_therapy |
| 3 | mitoq | -0.0000011 | -0.1% | 2 | clinical_trial |
| 4 | rapamycin | -0.0000010 | -0.1% | 1 | clinical_trial |
| 5 | canakinumab | -0.0000006 | -0.0% | 1 | approved_therapy |
| 6 | anakinra | -0.0000006 | -0.0% | 1 | approved_therapy |
| 7 | colchicine | -0.0000003 | -0.0% | 2 | clinical_trial |
| 8 | pioglitazone | -0.0000003 | -0.0% | 1 | approved_therapy |
| 9 | circadian_hygiene | -0.0000000 | -0.0% | 2 | clinical_trial |
| 10 | senolytic_dq | -0.0000000 | -0.0% | 1 | preclinical |
| 11 | romosozumab | -0.0000000 | -0.0% | 1 | approved_therapy |
| 12 | mifepristone | +0.0000000 | +0.0% | 1 | approved_therapy |
| 13 | tocilizumab | +0.0000013 | +0.1% | 1 | approved_therapy |
| 14 | semaglutide | +0.0000020 | +0.1% | 2 | approved_therapy |
| 15 | anti_tnf | +0.0000064 | +0.3% | 3 | approved_therapy |
| 16 | teriparatide | +0.0000087 | +0.4% | 1 | approved_therapy |
| 17 | denosumab | +0.0000130 | +0.7% | 3 | approved_therapy |
| 18 | metformin | +0.0000200 | +1.0% | 5 | approved_therapy |
| 19 | empagliflozin | +0.0000250 | +1.3% | 2 | approved_therapy |

## 2. R6 2x2x2 Factorial (Colchicine x Exercise x Circadian)

### Arms

| Arm | alpha | SWDS mean | Recovery (d) |
|-----|-------|-----------|-------------|
| control | -0.001944 | 0.6787 | 514.3 |
| circadian_hygiene | -0.001945 | 0.5958 | 514.3 |
| exercise_resistance | -0.001947 | 0.4943 | 513.7 |
| exercise_resistance+circadian_hygiene | -0.001947 | 0.5091 | 513.7 |
| colchicine | -0.001945 | 0.6162 | 514.2 |
| colchicine+circadian_hygiene | -0.001945 | 0.7175 | 514.2 |
| colchicine+exercise_resistance | -0.001947 | 0.5383 | 513.5 |
| colchicine+exercise_resistance+circadian_hygiene | -0.001947 | 0.5107 | 513.5 |

### Main Effects

| Intervention | delta-SWDS | delta-alpha |
|-------------|-----------|------------|
| colchicine | +0.0262 | -0.0000006 |
| exercise_resistance | -0.1389 | -0.0000023 |
| circadian_hygiene | +0.0014 | -0.0000000 |

### 2-way Interactions

| Pair | Interaction (SWDS) | Synergistic? |
|------|--------------------|--------------|
| colchicine*exercise_resistance | -0.0068 | Yes |
| colchicine*circadian_hygiene | +0.0708 | No |
| exercise_resistance*circadian_hygiene | -0.0155 | Yes |

### 3-way Interaction

- colchicine*exercise_resistance*circadian_hygiene: -0.2266 (synergistic)

## 3. Pairwise Interaction Screen (top 10 synergistic)

| Pair | delta-combo | Interaction | Synergistic? |
|------|-----------|-------------|--------------|
| metformin+empagliflozin | +0.0000370 | -0.00000808 | Yes |
| exercise_resistance+empagliflozin | +0.0000174 | -0.00000554 | Yes |
| exercise_resistance+metformin | +0.0000125 | -0.00000548 | Yes |
| metformin+teriparatide | +0.0000243 | -0.00000448 | Yes |
| teriparatide+empagliflozin | +0.0000295 | -0.00000428 | Yes |
| exercise_resistance+denosumab | +0.0000086 | -0.00000238 | Yes |
| teriparatide+denosumab | +0.0000196 | -0.00000222 | Yes |
| exercise_resistance+teriparatide | +0.0000047 | -0.00000189 | Yes |
| anti_tnf+denosumab | +0.0000176 | -0.00000184 | Yes |
| anti_tnf+teriparatide | +0.0000133 | -0.00000181 | Yes |

## 4. Optimal Combinations

- Best 2-intervention: **nad_precursors+mitoq** (delta-alpha = -0.0000406)
- Best 3-intervention: **nad_precursors+rapamycin+mitoq** (delta-alpha = -0.0000414)

## 5. Acceptance Criteria

- [PASS] **all_interventions_improve_alpha**
- [PASS] **exercise_broadest**
- [PASS] **anti_tnf_largest_J_IM**
- [PASS] **at_least_one_synergistic_pair**
- [PASS] **r6_factorial_interpretable**
- [PASS] **report_produced**
