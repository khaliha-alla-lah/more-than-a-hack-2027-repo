# Possession pipeline

## Integrated motion candidate

Install `requirements-possession.txt` in a Python 3.11+ environment.
The candidate includes the required SAM modules under
`possession/vendor/sam/`, with license and source attribution.

```powershell
$data = '../More Than A Hack 2027 - Initial Challenge - Dataset'
.venv/Scripts/python.exe -B -m possession.candidate_cli --fit --data $data --diagnostics data/motion-candidate/diagnostics.csv
.venv/Scripts/python.exe -B evaluation/validate_submission.py data/motion-candidate/submission.csv
```

Omit `--fit` to regenerate from saved weights. Use `--output` to choose
a destination and `--model-dir` to select a model bundle. The manifest
records the three model files and checksums. Training publishes a new
manifest after all model files have been saved.

The three-model motion bundle has its own format and is loaded through
`possession.candidate_cli`. The baseline and two-model corruption experiments
use the Phase 1 release format and `python -m possession`; these formats are
not interchangeable. On macOS/Linux use `.venv/bin/python` in the examples.

`candidate.py` owns feature preparation, seven-frame velocity estimation,
and inference. `candidate_training.py` owns the benchmarked training
recipe. `candidate_cli.py` handles external data, model storage, validated
CSV export and diagnostics. `candidate_eval.py` runs whole-match folds.
All use the existing Tracking adapters. Diagnostic rows sample regular
intervals, predicted transitions and decoder/frame-classifier disagreements.

```powershell
.venv/Scripts/python.exe -B -m possession.candidate_eval --data $data --matches J03WMX --output data/evaluation/motion-j03wmx.json
```

Omit `--matches` for all seven folds. Each fold excludes its evaluation
match from all training stages. The development protocol uses two fixed
corruption seeds.

## Comparing tracking corruption and public fork components

From the repository root, run the following commands with the existing environment.

```powershell
$data = '../More Than A Hack 2027 - Initial Challenge - Dataset'
.venv/Scripts/python.exe -B -m possession.experiments --data $data
.venv/Scripts/python.exe -B -m possession.short_audit --data $data
.venv/Scripts/python.exe -B -m possession.fork_comparison --data $data
```

The first command evaluates fixed configurations with each DFL match excluded
from training in turn. It uses the same corrupted inputs for every method in a
fold and writes metrics after each completed match. Correlated jitter uses
an experimental AR(1) coefficient of 0.9. Measure its fit to the target tracking
before adopting it for training. The short-spell audit uses the frozen model
and original holdout corruption to compare ball visibility, team probabilities,
and interval overlap before and after decoding.

The third command expects two public checkouts in `data/`. Use Blaze2fire's
`solution` branch at `2d9f95aad4f081ebc2e82397d208ff9c70646fb3` in
`data/fork-blaze`, and SAM Analytics' `main` branch at
`3b800fb11184a57dc84baab819c21142fbc57513` in `data/fork-sam`.
It runs Blaze2fire's rules and HMM and tests SAM's duration-constrained decoder
with our rules probabilities. These scores cover the named components only.

Reports are written to `data/experiments/`. The short-spell audit loads the
baseline from `data/frozen-candidate/`, which should contain the model files
and a Phase 1 release manifest. To inspect archived weights that have only
the original `benchmark.json`, add `--legacy-model` to `possession.short_audit`.
This diagnostic-only path warns that model/source hashes are unverified;
production inference continues to require a verified release. An existing
release manifest is always validated, even with the legacy flag.

To train and export the experimental full-feature candidate:

```powershell
.venv/Scripts/python.exe -B -m possession.experiments --data $data --fit-candidate correlated
.venv/Scripts/python.exe -B -m possession --data $data --model-dir data/experiments/candidate --output data/experiments/candidate.csv
.venv/Scripts/python.exe -B evaluation/validate_submission.py data/experiments/candidate.csv
```

Experimental export now writes a Phase 1 release with feature schemas,
checksums and training provenance. Choose a new `--model-dir` for each run;
existing candidates are preserved. A previous benchmark-only export must
be retrained into a new release directory before production inference.

The pipeline produces one label per tactical frame. The learned entry estimates whether play is live and which team has control, then decodes the sequence across each half. The rules entry uses the same tracking features and decoder.

## Run locally

Phase 1 preserves the original features, corruption recipe, model settings and decoder. It adds explicit feature ordering, frozen model releases and inference receipts. The original weights and DFL inputs were not present in this checkout. The results further below are historical; see [the current verification record](docs/PHASE1_REPRODUCTION.md).

Use Python 3.11 or newer. The rules path was verified in a fresh Python 3.11 environment on macOS arm64. From the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-rules.txt
.venv/bin/python -m possession.preflight --rules
.venv/bin/python -m possession --rules --output data/rules.csv
.venv/bin/python evaluation/validate_submission.py data/rules.csv
```

On Windows, create the environment with `py -3.11 -m venv .venv` and replace `.venv/bin/python` with `.venv/Scripts/python.exe`. No shell activation is required.

Place the organiser inputs under `data/`, set `MTAH_DATA_DIR`, or pass `--data /path/to/dataset` to the commands. Inference needs `tactical/BAR-ATH_tactical_atd.xml` and `tactical/BAR-ATH_tactical_atd.txt`. Videos, broadcast tracking, homography and DFL XMLs are not required for inference. `--diagnostics` additionally needs tactical Smart Tags.

Every baseline inference through `python -m possession` writes a validated UTF-8/LF CSV and `<output>.manifest.json` recording input/output SHA-256 hashes, source-file hashes, dependency versions and decoder settings. The receipt contains hashes, not tracking rows. Generated artifacts remain under ignored `data/` by default. The separate motion-candidate CLI retains its upstream export behavior.

## Reconstruct the learned baseline

The new environment uses the verified rules dependency versions plus pinned LightGBM and SciPy. It differs from the original Windows environment at commit `0bf453e`; any newly trained model is a reconstruction, not the recovered original candidate. Learned training has not yet been rerun in this checkout.

Download the 14 required XMLs for the seven games from [the pinned IDSSE mirror](https://huggingface.co/datasets/pysport/idsse-data/tree/a715a38dfbaf5f58e431727c2b78d174101a703c), then train into a new directory:

```bash
.venv/bin/python -m pip install -r requirements-possession.txt
.venv/bin/python -m possession.fetch_dfl --output data/idsse
.venv/bin/python -m possession.preflight --training --model-dir data/releases/phase1-e0
.venv/bin/python -m possession.train --model-dir data/releases/phase1-e0
```

The downloader uses the fixed revision in `possession/external.py`, verifies file sizes and the hub's LFS SHA-256 or Git-blob checksum, and records local SHA-256 hashes. Existing valid files are reused. Invalid existing files are preserved with an error; partial downloads never become final inputs. Download and preflight commands can run before numerical dependencies are installed. Internet access is needed for uncached packages and the public dataset.

Training writes `manifest.json`, `benchmark.json` and model text files atomically. It refuses to overwrite an existing release directory. The manifest binds model hashes, ordered feature schemas, source hashes, training input fingerprints, splits, seeds, augmentation settings and dependency versions. It records the original four-training/one-development/two-reporting split without presenting the reporting games as historically untouched.

Generate from that frozen release without downloading or loading the training data:

```bash
.venv/bin/python -m possession.preflight --model-dir data/releases/phase1-e0
.venv/bin/python -m possession --model-dir data/releases/phase1-e0 --output data/submission.csv
.venv/bin/python evaluation/validate_submission.py data/submission.csv
```

Inference checks the release schema, model and benchmark hashes, and the inference-source hashes before loading tracking. It records the current runtime in the receipt; restore the recorded dependency versions for reproduction. A legacy `benchmark.json` without a release manifest is rejected. The previous `--fit` command remains an explicit training-plus-inference convenience, but the default command only loads weights. Rename the validated CSV and technical document to the registered team name when submitting.

Run the synthetic integrity tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The suite has no dependency on the target data or public labels. A real LightGBM save/load test is skipped explicitly when LightGBM cannot be imported. The remaining tests exercise feature ordering, release tampering, failed writes, download checksums, half masks and repeatability; they do not establish model accuracy.

## Local video diagnostics

The optional diagnostics CSV lists low-confidence decoded transitions with tactical frame IDs, timestamps, model probabilities, ball and player evidence, and whether the transition falls inside a Smart Tag set-piece window. Extract a frame locally using its ID. Supply `--ffmpeg` with the path to an existing FFmpeg executable, or install the pinned helper below.

```bash
.venv/bin/python -m pip install imageio-ffmpeg==0.6.0
.venv/bin/python -m possession.frame 113742
```

The image stays under `data/local-frames/` by default. Use this for local checks such as video alignment or tracking loss. The challenge rules require algorithmic outputs and reserve the official public labels for pipeline checks. Keep screenshots and other restricted match assets out of hosted tools and public repositories.

## Design

- `tracking.py` and `dfl.py` convert provider-specific formats to metre coordinates on a centred pitch. DFL labels remain separate from model inputs.
- `features.py` extracts ball motion and proximity, team motion and shape, and the nearest observations on either side of a ball gap. Feature windows stay within their half.
- `learn.py` introduces bursts of missing tracking and coordinate noise before feature extraction. The deployed live/dead model uses only player movement and shape because the ball's stoppage behavior differs substantially between providers. The team-control model uses both ball and player evidence. LightGBM sees missing values explicitly.
- `decode.py` combines the binary predictions into H, A, and D scores and applies Viterbi transition costs. Direct turnovers and same-team restarts are allowed. A possession has no fixed minimum duration.
- `benchmark.py` trains on four DFL matches, selects the model and transition costs on J03WOY, and reports results on J03WN1 and J03WQQ. Whole games stay separate across these stages.

External supervision comes from Sportec Open DFL (IDSSE), released under CC BY 4.0 by Bassek, Rein, Weber et al. Four full matches supply live/dead and team-control labels from the tracking XML. A fifth match selects the decoder costs, and two matches supply reporting scores. Those two matches have already informed development discussions and are not a pristine test set. The challenge tracking supplies the unlabeled distribution of gap lengths for augmentation. See the [dataset paper](https://doi.org/10.1038/s41597-025-04505-y) and [original data release](https://doi.org/10.6084/m9.figshare.28196177.v1).

The decoder requires accumulated evidence to change team when a ball is loose or deflected. It can still switch directly when the receiving team gains control. A stoppage ends the current possession and the next live frame can belong to either side. The output includes the graded dead-ball interval before the second-half kickoff described in the [organizer FAQ](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/FAQ.md).

The benchmark uses DFL labels directly. A timing audit on one training match found 39 of 96 recorded restarts had inactive tracking half a second before the event timestamp and active tracking half a second after it. Event times and tracking boundaries can differ, so this audit did not relabel frames. The synthetic tracking losses approximate some Metrica failure modes, and the match sample is small.

## Historical benchmark results

These numbers were recorded before the Phase 1 reconstruction work. They have not been independently reproduced from the missing original artifacts. Phase 1 changes the packaging and contracts; it does not claim a new learned score.

The first two-model version scored 0.3160 on the degraded and stress development scenarios combined. Its stoppage model relied heavily on the ball being nearly stationary or out of bounds in DFL. Applied to tactical tracking, it predicted only 15 dead frames after accounting for the published pre-kickoff interval. We restricted the stoppage model to six player-motion and shape features to remove that source-dependent signal. The revised version scored 0.2692 on the same development scenarios and predicted stoppages throughout both tactical halves.

| Held-out match | Tracking | Nearest-player vote | Rules entry | Final entry |
| --- | --- | ---: | ---: | ---: |
| J03WN1 | Clean | 0.2838 | 0.3513 | 0.3670 |
| J03WN1 | Degraded | 0.1971 | 0.1771 | 0.2853 |
| J03WN1 | Stress | 0.1774 | 0.1559 | 0.2036 |
| J03WQQ | Clean | 0.3608 | 0.3078 | 0.4558 |
| J03WQQ | Degraded | 0.2313 | 0.2081 | 0.3949 |
| J03WQQ | Stress | 0.2194 | 0.1826 | 0.2365 |

Scores are spellF1 on DFL. The final entry averaged 0.3401 across the two degraded matches versus 0.2142 for the nearest-player reference. Short possessions remain difficult. At 0.5 temporal IoU, it recovered 1 of 96 and 2 of 46 spells lasting 0.5 to 1.5 seconds in those two runs. These results measure behavior on external matches with simulated tracking loss. The private competition score remains unknown.

The final output uses tactical tracking. Smart Tags appear in local diagnostics. A broadcast probe found 21 plausible matches in 36 sampled windows, with 1.16 m median ball-position discrepancy on held-back frames. The probe could not establish whether those matches were live or replayed or whether they covered the tactical gaps. The submitted predictions use tactical tracking alone.
