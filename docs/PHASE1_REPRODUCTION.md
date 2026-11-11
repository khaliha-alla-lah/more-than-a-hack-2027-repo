**Phase 1 verification — 11 November 2026**

Branch: `codex/phase-1-reproducible-baseline`.
Historical implementation reference: `0bf453e5032f00327681a29f838994a3fe109359`.

The reproducibility infrastructure is implemented and the rules baseline is verified in a fresh environment. Restoring the learned E0 baseline remains pending: the original weights are unavailable, the 14 external DFL XML files are absent, and LightGBM is not installed. No new learned score is claimed.

**Implemented**

- Explicit ordered feature schemas preserve the original 27-feature matrix and six-feature status input. Reordering a Python dictionary cannot silently reorder model inputs.
- `python -m possession.train` trains and evaluates without generating a target submission. Default `python -m possession` loads a frozen release without reading DFL training data. The explicit legacy `--fit` convenience remains available.
- Model releases contain the models, benchmark report and a versioned manifest. Hashes bind weights, feature schema, inference source, external inputs, splits, seeds, augmentation settings and runtime versions. Releases are staged atomically and existing candidates cannot be overwritten by training.
- Inference rejects missing releases, altered model/report files, incompatible feature schemas and changed inference code. It writes a validated UTF-8/LF CSV and a local provenance receipt.
- `python -m possession.preflight` reports inference/training prerequisites before an expensive run. It runs even without NumPy installed; missing or unloadable packages are reported in JSON with a nonzero exit status.
- `python -m possession.fetch_dfl` downloads only the 14 required XML files from the pinned public revision, verifies size and hub checksums, reuses verified inputs and preserves invalid existing files for inspection. Its failure paths are tested; a live download has not succeeded in this environment.
- `requirements-rules.txt` pins the clean, locally verified rules environment. `requirements-possession.txt` adds the existing LightGBM/SciPy pins. This changes some dependencies from the historical Windows environment, so future training is explicitly a reconstruction.
- README and POSSESSION.md now introduce the team implementation, document macOS/Linux and Windows entry points, and distinguish historical scores from current verification.

The feature formulas, independent-noise augmentation, model hyperparameters, historical split and temporal decoding behavior remain unchanged. Noise repair belongs to the next implementation phase.

**Verification results**

| Check | Result |
|---|---|
| Fresh isolated environment | Repository `.venv`, Python 3.11.15, macOS arm64; dependencies installed offline from cached wheels |
| Dependency consistency | `python -m pip check` passes |
| Synthetic regression suite | 18 tests run: 17 pass, one real-LightGBM round trip explicitly skipped because LightGBM is unavailable |
| Feature contract | Matrix values/order match the historical extraction, including NaNs; missing/extra feature names fail |
| Release behavior | Corrupted weights/report, reordered schema, changed source, invalid penalties and unsafe filenames rejected; existing releases preserved |
| Download behavior | Size, SHA-256 and Git-blob checksum checks tested with local fixtures; incomplete downloads are removed and invalid existing files preserved |
| Submission behavior | Half masks, the metadata-defined second-half D interval, LF encoding, repeated output and preservation on failed validation tested synthetically |
| Full target rules runs | Two 199,319-row files are byte-identical and match the pre-change review's checksum |
| Official validator | Passes: frames 0–199318, valid schema and labels |
| Missing prerequisites | Learned inference, training and training-preflight return clear nonzero failures without a traceback |
| Learned benchmark / accuracy | Not run; no inference about the private competition score |

Rules CSV SHA-256:

```text
ead9c832dbb747afc2dcd027ca291f665b1d70677cc4a4ff9cf0b7d2e9831987
```

Local artifacts are under ignored `data/phase1-baseline/`: `rules.csv`, `rules-repeat.csv`, their `.manifest.json` receipts, and `verification.json`. Each receipt identifies the source and runtime of its own run, including whether the working tree was dirty. Neither the restricted inputs nor the generated CSVs are added to Git.

**Frozen E0 comparison contract**

When the external inputs become available, reconstruct the historical reference before improving it:

| Property | Fixed value |
|---|---|
| Training matches | J03WMX, J03WOH, J03WPY, J03WR9 |
| Development match | J03WOY |
| Reporting matches | J03WN1, J03WQQ; already inspected historically |
| Supervision | Raw DFL BallStatus/BallPossession mapping; unknown live ownership remains X |
| Scenarios | Clean, historical degraded, historical stress |
| Model seed / rounds | 417 / 100 |
| Training subsampling | Every fourth frame, unchanged |
| Corruption seed | `int(match[-2:], 36) * 10 + zero_based_half`; stress adds 10000 |
| Selected family | Collective live/dead head plus full-feature team-control head |
| Decoder choices | None, (5,7), (12,16), (25,30), selected on degraded/stress development spellF1 |
| Evaluation | Existing official metric implementation, unchanged |
| Target public labels | Excluded from training, tuning and candidate selection |

This reproduces the historical experiment design. It does not turn the reporting games into an untouched test set. A grouped/nested protocol for selecting improvements is a later phase. The known independent-noise defect remains in E0 deliberately so that its repair can be measured separately.

**Next executable steps**

The existing `.venv` already runs the rules baseline. Once network access is available, install the learned dependencies, retrieve the public XMLs and reconstruct E0 into a new directory:

```bash
.venv/bin/python -m pip install -r requirements-possession.txt
.venv/bin/python -m possession.fetch_dfl --output data/idsse
.venv/bin/python -m possession.preflight --training --model-dir data/releases/phase1-e0
.venv/bin/python -m possession.train --model-dir data/releases/phase1-e0
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m possession --model-dir data/releases/phase1-e0 --output data/submission.csv
.venv/bin/python evaluation/validate_submission.py data/submission.csv
```

The dataset download was attempted and failed with a DNS/network error. LightGBM was absent from the installed environments and locally cached wheels examined. Installing the learned requirements from PyPI was also attempted; pip could not retrieve a matching LightGBM distribution in this environment. Retraining, real model serialization/loading and a learned submission remain unverified until those prerequisites are available. The model-artifact unit tests use explicit test doubles; they are not a substitute for the skipped LightGBM integration test.

**Rebase onto the integrated motion candidate**

The branch was rebased onto `origin/main` at `ebebe1c`. The Phase 1 commit
`e04863c` became `2c8cf19`, with an unchanged patch according to `git range-diff`.
The uncommitted work was restored, and the original branch tip remains at
`codex/phase-1-before-rebase-20261111`.

Git reported no textual conflicts. Integration checks found that the new
experiment exporter still wrote benchmark-only bundles, which the Phase 1
production loader correctly rejects. Experimental export now uses the same
atomic release writer, preserving the experiment recipe and recording its
own provenance. Historical short-spell audits accept an explicit
`--legacy-model` flag with a provenance warning; modern manifests are always
validated, and production inference still requires them. Auditing a release
with no decoder is also supported. The upstream three-model motion pipeline
and its separate bundle format remain intact.

Rebase verification: 21 tests ran, 20 passed and the LightGBM round trip was
skipped. SciPy was installed from a cached wheel for the upstream experiment
checks. Six available command-line entry points passed help/import checks,
all possession Python modules parsed, and the complete `rules-rebased.csv`
passed official validation and matched the reference SHA-256 above. The
learned motion candidate remains unexecuted because LightGBM is unavailable.
