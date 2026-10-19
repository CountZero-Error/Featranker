## 1. Setup

- [x] 1.1 Create `spikes/investigator/` with `spike.py`, `run.py`, `test_spike.py`, and `results/`; confirm `python -m build --sdist` and the wheel contain no `spikes/` files
- [x] 1.2 Write the spike environment instructions in `FINDINGS.md` (Python 3.12 venv, `pip install -e . shap openai pytest`)

## 2. Benchmark and scoring

- [x] 2.1 Test: per-seed data generation is deterministic, roles are permuted across seeds, and the ground truth lists the 5 mechanisms and 4 noise features
- [x] 2.2 Implement the synthetic benchmark (D2) and ground truth
- [x] 2.3 Test: scoring matches findings order-free, ignores duplicates, counts each false-finding kind from D2, and scores a malformed reply as zero findings
- [x] 2.4 Implement findings parsing and scoring

## 3. Evidence battery and actions

- [x] 3.1 Test: the battery has train/eval R² per model, mean |SHAP| and direction per model and feature, and only |r| ≥ 0.5 pairs; the planted `e`/`f` pair appears
- [x] 3.2 Implement the evidence battery (D4) on the four seeded models (D3)
- [x] 3.3 Test: each action returns the documented keys; `refit` rejects params outside the whitelist; `shap_interaction` rejects a non-tree model
- [x] 3.4 Implement the four actions

## 4. Arms, loop, and trail

- [x] 4.1 Test with a scripted fake LLM: narrator and evidence make exactly 1 call; the investigator stops after 3 follow-ups; prediction verdicts come out `confirmed`/`refuted`/`invalid` as expected; an invalid reply forces a final reply
- [x] 4.2 Implement the prompts and the investigator loop (D4, D5)
- [x] 4.3 Test: `replay(trail)` reproduces every action result and verdict exactly without an LLM
- [x] 4.4 Implement trail recording and replay
- [x] 4.5 Test: confirmed-false hypotheses are counted against ground truth
- [x] 4.6 Implement confirmed-false counting
- [x] 4.7 Test: the D6 decision rule returns each of the three outcomes on hand-made summaries, including the 4-of-5-seeds and borderline (within 0.05) cases
- [x] 4.8 Implement the D6 decision rule in code, so the rule is frozen with the prompts

## 5. Runner and dry run

- [x] 5.1 Implement `run.py` (flags: `--model`, `--base-url`, `--seeds`, `--arms`; key from `DEEPSEEK_API_KEY`; writes `results/<model>-<timestamp>.json` with model id, commit hash, seeds, per-arm findings, trails, scores)
- [x] 5.2 Dry run end to end with the fake LLM on all 5 seeds; full spike tests pass; `python -m pytest -q` for the package still passes
- [x] 5.3 Freeze: record the SHA-256 of `spike.py` (prompts and decision rule) in `FINDINGS.md` before any real run; `run.py` stores it in every result file

## 6. Real runs and decision (maintainer runs the LLM)

- [x] 6.1 Maintainer runs `run.py` with their DeepSeek model on 5 seeds × 3 arms in their own terminal
- [x] 6.2 Analyse the result files and apply the D6 rule; if NO-GO or borderline, the maintainer runs one stronger model and both are analysed
- [x] 6.3 Write `FINDINGS.md`: per-arm recall and false findings per seed, confirmed-false hypotheses, example trails, the outcome under D6, and any deviations from the frozen setup
- [x] 6.4 Signal review of `FINDINGS.md`

## 7. Roadmap

- [x] 7.1 Update the roadmap in `openspec/config.yaml` with the outcome (1.1 `evidence-for-why`; 1.2 `llm-investigator` or deprecation of `explain()`/`to_prompt()`)

## 8. Confirmatory v2 (D8)

- [x] 8.1 Test: a finding or claim with the wrong feature count for its mechanism is a false finding; correct-arity findings score as before
- [x] 8.2 Implement the arity rule in scoring and state it in the final format; `run.py` defaults to seeds 5–9
- [x] 8.3 Freeze: record the new `spike.py` SHA-256 in `FINDINGS.md`; spike and package tests pass
- [x] 8.4 Maintainer runs `run.py --model deepseek-flash` once in their own terminal
- [x] 8.5 Analyse, write the v2 section and final outcome in `FINDINGS.md`; Signal review
