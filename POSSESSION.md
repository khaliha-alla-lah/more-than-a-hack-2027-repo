# Possession pipeline

## Integrated motion candidate

Use the existing pinned requirements in a Python 3.14 environment.
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
and their original benchmark manifest.

To train and export the experimental full-feature candidate:

```powershell
.venv/Scripts/python.exe -B -m possession.experiments --data $data --fit-candidate correlated
.venv/Scripts/python.exe -B -m possession --data $data --model-dir data/experiments/candidate --output data/experiments/candidate.csv
.venv/Scripts/python.exe -B evaluation/validate_submission.py data/experiments/candidate.csv
```

The pipeline produces one label per tactical frame. The learned entry estimates whether play is live and which team has control, then decodes the sequence across each half. The rules entry uses the same tracking features and decoder.

## Run locally

Use 64-bit Python 3.14. From the repository root in PowerShell, set the dataset directory to the folder containing `tactical` and `broadcast`.

```powershell
python -m venv .venv
$python = '.venv/Scripts/python.exe'
& $python -m pip install -r requirements-possession.txt
$env:MTAH_DATA_DIR = (Resolve-Path '../More Than A Hack 2027 - Initial Challenge - Dataset').Path
& $python -m possession --rules --output data/rules.csv
```

For the learned entry, download the match-information and observed-position XML files for the seven DFL games from [the pinned IDSSE mirror](https://huggingface.co/datasets/pysport/idsse-data/tree/a715a38dfbaf5f58e431727c2b78d174101a703c) into `data/idsse`. The PowerShell commands below fetch and verify the file lengths before renaming each download.

```powershell
$revision = 'a715a38dfbaf5f58e431727c2b78d174101a703c'
$dir = 'data/idsse'
New-Item -ItemType Directory -Force $dir | Out-Null
$files = Invoke-RestMethod "https://huggingface.co/api/datasets/pysport/idsse-data/tree/$revision"
foreach ($file in ($files | Where-Object { $_.path -match '^DFL_(02_01_matchinformation|04_03_positions_raw_observed)_.*\.xml$' })) {
    $dest = Join-Path $dir $file.path
    if ((Test-Path -LiteralPath $dest) -and (Get-Item -LiteralPath $dest).Length -eq $file.size) { continue }
    Invoke-WebRequest "https://huggingface.co/datasets/pysport/idsse-data/resolve/$revision/$($file.path)" -OutFile "$dest.part"
    if ((Get-Item -LiteralPath "$dest.part").Length -ne $file.size) { throw "Incomplete: $dest" }
    Move-Item -LiteralPath "$dest.part" -Destination $dest -Force
}
```

Train, evaluate, and generate the CSV in one command. With an existing trained model, omit `--fit` to regenerate predictions.

```powershell
& $python -m possession --fit --diagnostics data/uncertain-transitions.csv
& $python evaluation/validate_submission.py data/submission.csv
```

The command writes the comparison table and model metadata to `data/possession-model/benchmark.json`. Generated files and input data stay under the organizer's ignored `data/` directory. Rename the validated CSV and technical document to the registered team name when packaging the form.

## Local video diagnostics

The optional diagnostics CSV lists low-confidence decoded transitions with tactical frame IDs, timestamps, model probabilities, ball and player evidence, and whether the transition falls inside a Smart Tag set-piece window. Extract a frame locally using its ID. Supply `--ffmpeg` with the path to an existing FFmpeg executable, or install the pinned helper below.

```powershell
& $python -m pip install imageio-ffmpeg==0.6.0
& $python -m possession.frame 113742
```

The image stays under `data/local-frames/` by default. Use this for local checks such as video alignment or tracking loss. The challenge rules require algorithmic outputs and reserve the official public labels for pipeline checks. Keep screenshots and other restricted match assets out of hosted tools and public repositories.

## Design

- `tracking.py` and `dfl.py` convert provider-specific formats to metre coordinates on a centred pitch. DFL labels remain separate from model inputs.
- `features.py` extracts ball motion and proximity, team motion and shape, and the nearest observations on either side of a ball gap. Feature windows stay within their half.
- `learn.py` introduces bursts of missing tracking and coordinate noise before feature extraction. The deployed live/dead model uses only player movement and shape because the ball's stoppage behavior differs substantially between providers. The team-control model uses both ball and player evidence. LightGBM sees missing values explicitly.
- `decode.py` combines the binary predictions into H, A, and D scores and applies Viterbi transition costs. Direct turnovers and same-team restarts are allowed. A possession has no fixed minimum duration.
- `benchmark.py` trains on four DFL matches, selects the model and transition costs on J03WOY, and reports results on J03WN1 and J03WQQ. Whole games stay separate across these stages.

External supervision comes from Sportec Open DFL (IDSSE), released under CC BY 4.0 by Bassek, Rein, Weber et al. Four full matches supply live/dead and team-control labels from the tracking XML. A fifth match selects the decoder costs, and two matches supply held-out scores. The challenge tracking supplies the unlabeled distribution of gap lengths for augmentation. See the [dataset paper](https://doi.org/10.1038/s41597-025-04505-y) and [original data release](https://doi.org/10.6084/m9.figshare.28196177.v1).

The decoder requires accumulated evidence to change team when a ball is loose or deflected. It can still switch directly when the receiving team gains control. A stoppage ends the current possession and the next live frame can belong to either side. The output includes the graded dead-ball interval before the second-half kickoff described in the [organizer FAQ](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/FAQ.md).

The benchmark uses DFL labels directly. A timing audit on one training match found 39 of 96 recorded restarts had inactive tracking half a second before the event timestamp and active tracking half a second after it. Event times and tracking boundaries can differ, so this audit did not relabel frames. The synthetic tracking losses approximate some Metrica failure modes, and the match sample is small.

## Benchmark results

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
