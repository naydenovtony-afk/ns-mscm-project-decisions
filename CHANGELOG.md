# Changelog

Changes after the design freeze (commit "freeze design"). Each entry: what, why, and confirmation that everything was rerun.

## 2026-11-17: replications R (not a change of the DGP or the estimators)

- `experiments.R` was `null` at the freeze, to be set after the Step 4 timing. One replication of all
  12 cells took about 5,100 worker-seconds (6 workers, 4 physical cores); R = 50 needs about 12 h.
- Decided before any replication result (the Step 4 single-replication timing run in `results/timing/`
  is not used for any reported number):
  - **Phase 1:** every cell (S1-S4 x N in {300, 1000, 5000}) to R = 50
    (`python run_all.py --stage simulate --reps 50 --workers 6 --resume`).
  - **Phase 2 (pre-declared, run regardless of the phase 1 results):** N = 300 and N = 1000 cells,
    all scenarios, extended to R = 100 (`python run_all.py --stage simulate --workers 6 --resume`).
  - Final tables: R = 100 for N in {300, 1000}, R = 50 for N = 5000 (`experiments.R` in config.yaml).
- Nothing in M4 or elsewhere is changed for speed.
- Step 4 code (`nsmscm/dml.py`, `nsmscm/experiment.py`, the simulate stage, `tests/test_estimators.py`)
  implements the frozen estimator settings; it was written after the freeze, and the M3/M5 agreement
  test passed on its first run (`results/test_m3_m5_agreement.json`), so no bug fix was needed.

## 2026-11-17: analysis code and hypothesis decision rules (before any replication result)

- `nsmscm/analysis.py` written while phase 1 was running and before any of its rows had been read.
  It was smoke-tested only on the Step 4 single-replication timing files, in a scratch folder.
- The draft states H1-H3 without operational criteria. The decision rules were fixed here, before
  results (also in the module docstring):
  - **H1** (S1, N in {300, 1000}): RMSE(M4) < RMSE(M3) and the paired mean difference of squared errors
    (M4 - M3) is below 0 by more than 2 MC SE; per intervention.
  - **H2** (one nuisance rule wrong: S3 all interventions; S4 A1, A3; every N): coverage(M4) >= 0.90 and
    |bias(M4) - bias(M3)| <= 2 MC SE of the paired difference.
  - **H3** (both wrong: S4 A2): bias(M4) at N = 5000 differs from 0 by more than 2 MC SE, and |bias(M4)| at
    N = 5000 >= |bias(M4)| at N = 300 - 2 MC SE of that difference.
- Pending: `run_all.py` gets the `analyze` stage and a default `all` stage after the simulation
  finishes (it is not edited while the detached run uses it).

## 2026-11-17: amendments to the hypothesis decision rules (before any phase-1 result was read)

Requested by the author and committed while phase 1 was running; no phase-1 row had been read.
The running simulation is not affected (only `nsmscm/analysis.py` changes).

1. **H3** (S4, A2) gets a third required criterion at N = 5000: |bias(M4)| - |bias(M3)| > 2 MC SE, paired
   across replications. With s4, s3 the signs of the two biases, the difference is
   mean_r[s4 (e4_r - theta) - s3 (e3_r - theta)] with MC SE sd_r(...)/sqrt(R). Without it, a bias shared
   by M3 would be attributed to the rules. H3 is supported only if all three criteria hold.
2. **Cell by cell.** Every hypothesis is evaluated and reported separately for each intervention x N cell
   it covers, as "supported" / "not supported". There are no counts across cells, no pooling and no
   overall verdict.
3. **H2's bias criterion** (|bias(M4) - bias(M3)| <= 2 MC SE) is a failure-to-reject criterion and is
   labelled that way in SUMMARY.md.

## 2026-11-17: interruption of phase 1 and relaunch with --resume (no design change)

- Phase 1 was interrupted at about 21:29 (launched 19:05). The log ends with a console `^C` after job
  42/600; the launcher was terminated with it (no `phase1 exit=` line), so phase 2 never started.
  Windows logged no crash, reboot, sleep or shutdown event (machine up since 2026-11-11); the source
  of the console Ctrl+C was not identified.
- Saved at the interruption: S1, N = 5000, replications 0-41 (42 complete row blocks of 15 rows; the
  last write, 21:22:00, preceded the interruption, so no partial rows). The 6 replications in progress
  were lost; replications are seed-deterministic, so `--resume` recomputes exactly the missing jobs.
- Relaunch: same design, settings, seeds and commands (phase 1 `--reps 50 --workers 6 --resume`, then
  phase 2 `--workers 6 --resume`, chained). Only the launch mechanism changed, to make the run
  independent of any console and of the Claude Code session: Windows scheduled task
  `paper3_ns_mscm_sim` runs `launch_sim.ps1`, which starts each python run with `Start-Process` in its
  own hidden console (new process group), keeps the "system required" execution state, writes the
  launcher PID to `results/launcher.pid`, and appends a heartbeat (time, PIDs, live workers, finished
  replications per cell) to `results/sim_run.log` every 5 minutes. Python output goes to
  `results/sim_phase<k>_<timestamp>.out/.err`.
- Relaunch log: a first start at 23:26:52 was stopped after about one minute (no replication finished),
  because the heartbeat counted only direct children of the venv `python.exe` redirector (1 instead of the
  6 pool workers); `launch_sim.ps1` now counts all descendants. The run was restarted at 23:27:56
  (scheduled task parent: Task Scheduler service).

## 2026-11-17: 6 -> 4 workers (runtime setting only; no design change)

- Reason: the laptop was running hot; 4 workers = the number of physical cores.
- Results do not depend on the number of workers: every random stream is derived from
  (seeds.replications, scenario, N, rep), each job uses single-thread BLAS and torch, and no state is
  shared between jobs. Checked on the saved data: replication 0 of (S1, N = 5000), computed once in the
  Step 4 timing run and once in phase 1 (separate process pools, different job order), is identical in
  every column of all 15 rows.
- The running batch was allowed to finish (S1, N = 5000 replications 42-47 saved at 23:47:57, giving 48
  complete replications); the task and its process tree were stopped at 23:48:09. Jobs started after
  about 23:45 were lost and are recomputed by `--resume`.
- `launch_sim.ps1` now passes `--workers 4` to both phases; task, heartbeat and watcher are unchanged.

## 2026-11-18: run_all.py analyze stage and default "all" stage (no change to simulation or estimators)

- Phase 1 finished 2026-11-18 15:19:15 (exit 0) and phase 2 at 22:48:12 (exit 0): R = 100 for N in
  {300, 1000}, R = 50 for N = 5000, every cell complete (run log; not in the public repository).
- `run_all.py` gained the `analyze` stage (calls `nsmscm/analysis.py`, committed before any result) and a
  default `all` stage (truth, rulecheck, simulate with resume, analyze). The edit was deferred until the
  detached run had finished, because the run was using the file. Nothing in the DGP, the estimators, the
  seeds or the simulate stage changed.
- `python run_all.py --stage analyze` produced the Step 5 outputs (tables, figures, paper_numbers.json,
  SUMMARY.md); the full test suite passed (23 tests).

## 2026-11-19: single-column RMSE figure (report only; no change to simulation, estimators or numbers)

- `nsmscm/analysis.py` gained `fig_rmse_1col`: the three panels of `fig_rmse_vs_N` stacked vertically,
  y-axis label on each panel, width 3.3 in, PNG at 300 dpi (`results/fig_rmse_vs_N_1col.png`). It reads
  the same data as `fig_rmse_vs_N`; every table and number is unchanged.

## 2026-11-19: public repository (no change to code, settings or results)

- Published as a single-commit snapshot. Left out: the unpublished paper draft and the reference
  material from another study (`reference/`), the Windows launcher script `launch_sim.ps1` (it contains
  local paths), and the run logs (`results/sim_run.log`, `results/sim_phase*.out/.err`,
  `results/launcher.pid`; they contain process IDs). Process IDs were removed from the entries above.
- Added `LICENSE` (MIT), `CITATION.cff` and the reproduction section of `README.md`.
