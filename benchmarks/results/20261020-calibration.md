**Null study: shadow_p_value hits and noise groups selected (fraction of all null groups / fraction of runs with at least one)**

| Task | Group | Mode | Runs | p <= 0.05 | 95% CI | share | q | fam1 | fam2 | fam3 |
|---|---|---|---|---|---|---|---|---|---|---|
| reg | fast | joint | 20 | 0.000 (0/160) | 0.000 to 0.023 | 0.113 / 0.35 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 |
| reg | fast | independent | 20 | 0.000 (0/200) | 0.000 to 0.018 | 0.070 / 0.40 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 |
| clf | fast | joint | 20 | 0.000 (0/200) | 0.000 to 0.018 | 0.010 / 0.10 | 0.000 / 0.00 | 0.010 / 0.05 | 0.010 / 0.05 | 0.010 / 0.05 |
| clf | fast | independent | 20 | 0.000 (0/200) | 0.000 to 0.018 | 0.010 / 0.10 | 0.000 / 0.00 | 0.010 / 0.10 | 0.005 / 0.05 | 0.005 / 0.05 |
| reg | tree | joint | 20 | 0.000 (0/160) | 0.000 to 0.023 | 0.037 / 0.20 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 |
| reg | tree | independent | 20 | 0.000 (0/180) | 0.000 to 0.020 | 0.067 / 0.35 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 |
| clf | tree | joint | 20 | 0.000 (0/200) | 0.000 to 0.018 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 |
| clf | tree | independent | 20 | 0.000 (0/200) | 0.000 to 0.018 | 0.000 / 0.00 | 0.000 / 0.00 | 0.005 / 0.05 | 0.000 / 0.00 | 0.000 / 0.00 |

**Signal study: mean over seeds**

| Task | Group | Mode | Rule | Recall | Noise selected | Copy kept | c,d recall | e recall | Runs |
|---|---|---|---|---|---|---|---|---|---|
| reg | fast | joint | share | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | fast | joint | q | 0.90 | 0.00 | 1.00 | 0.75 | 1.00 | 10 |
| reg | fast | joint | fam1 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | fast | joint | fam2 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | fast | joint | fam3 | 0.96 | 0.00 | 1.00 | 0.90 | 1.00 | 10 |
| reg | fast | independent | share | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | fast | independent | q | 0.86 | 0.00 | 1.00 | 0.65 | 1.00 | 10 |
| reg | fast | independent | fam1 | 1.00 | 0.40 | 1.00 | 1.00 | 1.00 | 10 |
| reg | fast | independent | fam2 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | fast | independent | fam3 | 0.98 | 0.00 | 1.00 | 0.95 | 1.00 | 10 |
| clf | fast | joint | share | 0.60 | 0.00 | 1.00 | 0.00 | 1.00 | 10 |
| clf | fast | joint | q | 0.52 | 0.00 | 0.40 | 0.00 | 0.60 | 10 |
| clf | fast | joint | fam1 | 1.00 | 0.20 | 1.00 | 1.00 | 1.00 | 10 |
| clf | fast | joint | fam2 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| clf | fast | joint | fam3 | 0.62 | 0.00 | 1.00 | 0.05 | 1.00 | 10 |
| clf | fast | independent | share | 0.66 | 0.00 | 1.00 | 0.15 | 1.00 | 10 |
| clf | fast | independent | q | 0.60 | 0.00 | 0.90 | 0.00 | 1.00 | 10 |
| clf | fast | independent | fam1 | 1.00 | 0.10 | 1.00 | 1.00 | 1.00 | 10 |
| clf | fast | independent | fam2 | 1.00 | 0.10 | 1.00 | 1.00 | 1.00 | 10 |
| clf | fast | independent | fam3 | 0.66 | 0.10 | 1.00 | 0.15 | 1.00 | 10 |
| reg | tree | joint | share | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | tree | joint | q | 0.90 | 0.00 | 1.00 | 0.75 | 1.00 | 10 |
| reg | tree | joint | fam1 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | tree | joint | fam2 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | tree | joint | fam3 | 0.74 | 0.00 | 0.90 | 0.35 | 1.00 | 10 |
| reg | tree | independent | share | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | tree | independent | q | 0.86 | 0.00 | 1.00 | 0.65 | 1.00 | 10 |
| reg | tree | independent | fam1 | 1.00 | 0.10 | 1.00 | 1.00 | 1.00 | 10 |
| reg | tree | independent | fam2 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | tree | independent | fam3 | 0.70 | 0.00 | 1.00 | 0.25 | 1.00 | 10 |
| clf | tree | joint | share | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| clf | tree | joint | q | 0.64 | 0.00 | 0.70 | 0.10 | 1.00 | 10 |
| clf | tree | joint | fam1 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| clf | tree | joint | fam2 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| clf | tree | joint | fam3 | 0.56 | 0.00 | 0.70 | 0.05 | 0.70 | 10 |
| clf | tree | independent | share | 0.98 | 0.00 | 1.00 | 0.95 | 1.00 | 10 |
| clf | tree | independent | q | 0.68 | 0.00 | 0.70 | 0.20 | 1.00 | 10 |
| clf | tree | independent | fam1 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| clf | tree | independent | fam2 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| clf | tree | independent | fam3 | 0.58 | 0.00 | 1.00 | 0.00 | 0.90 | 10 |

**Decision Q (hits test), per configuration**

- `fast` models, reg, shadow_mode=joint: 0.0% (95% CI 0.0% to 2.3%, 0 of 160) -> ALLOWED: on this study's null data, the binomial reference values were conservative for built-in `fast` models
- `fast` models, reg, shadow_mode=independent: 0.0% (95% CI 0.0% to 1.8%, 0 of 200) -> ALLOWED: on this study's null data, the binomial reference values were conservative for built-in `fast` models
- `fast` models, clf, shadow_mode=joint: 0.0% (95% CI 0.0% to 1.8%, 0 of 200) -> ALLOWED: on this study's null data, the binomial reference values were conservative for built-in `fast` models
- `fast` models, clf, shadow_mode=independent: 0.0% (95% CI 0.0% to 1.8%, 0 of 200) -> ALLOWED: on this study's null data, the binomial reference values were conservative for built-in `fast` models
- `tree` models, reg, shadow_mode=joint: 0.0% (95% CI 0.0% to 2.3%, 0 of 160) -> ALLOWED: on this study's null data, the binomial reference values were conservative for built-in `tree` models
- `tree` models, reg, shadow_mode=independent: 0.0% (95% CI 0.0% to 2.0%, 0 of 180) -> ALLOWED: on this study's null data, the binomial reference values were conservative for built-in `tree` models
- `tree` models, clf, shadow_mode=joint: 0.0% (95% CI 0.0% to 1.8%, 0 of 200) -> ALLOWED: on this study's null data, the binomial reference values were conservative for built-in `tree` models
- `tree` models, clf, shadow_mode=independent: 0.0% (95% CI 0.0% to 1.8%, 0 of 200) -> ALLOWED: on this study's null data, the binomial reference values were conservative for built-in `tree` models

**Decision F (min_shadow_families), per task**

- reg: recommended N = 1
  - N=1: worst null group-selection fraction 0.000, worst fast-group c,d recall 1.00, passes
  - N=2: worst null group-selection fraction 0.000, worst fast-group c,d recall 1.00, passes
  - N=3: worst null group-selection fraction 0.000, worst fast-group c,d recall 0.90, passes
- clf: recommended N = 1
  - N=1: worst null group-selection fraction 0.010, worst fast-group c,d recall 1.00, passes
  - N=2: worst null group-selection fraction 0.010, worst fast-group c,d recall 1.00, passes
  - N=3: worst null group-selection fraction 0.010, worst fast-group c,d recall 0.05, fails

**Decision S (shadow_mode)**

- independent recommended for collinear data: False (pooled gain in e recall +0.030, in copy kept +0.045; null fractions raised above 0.05: 1)
  - reg tree share: joint 0.037 -> independent 0.067
