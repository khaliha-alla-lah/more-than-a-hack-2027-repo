# Source provenance

The tracking loaders, feature extraction, classifiers, and decoder in `core/` include
adapted external code. The original source snapshot supplied no licence file; no licence
is asserted for those adapted portions.

Local adaptations:

- No augmentation or synthetic dead-ball reconstruction code is included.
- Ball short-gap limit, touch lag, smoothing for touches, and ball speed differencing scale with native frame rate (0.4, 0.12, and 0.08 seconds as applicable).
- Possession weighting uses eight seconds at the native frame rate. Training stride is `round(4 * fps / 25)`, giving every fourth frame at 25 Hz and every second at 10 Hz.
- The wrapper extracts features separately for each half to prevent cross-half context leakage.
- The model uses two tracking classifiers. Unused second-stage, handover, and tag-prior code has been removed.
- Barcelona outputs apply the known dead-ball interval `[110808, 113742)` after decoding. This local policy does not alter training or the public first-half evaluation window; refreshed result files record its checksum.

New local code provides SkillCorner parsing/labels, downloads with provenance, the training pipeline, audits, tests, and reports.

External data:

- [SkillCorner open data](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656), credited to SkillCorner and its open-data initiative with PySport.
- Local BAR–ATH tactical tracking and public possession labels supplied by this challenge repository.

Raw data and model artifacts stay in ignored directories. `data/skillcorner/manifest.json` records downloaded-file SHA-256 hashes. Each result records a combined source/data fingerprint and model checksum.

The active checkpoint in `artifacts/` is trained on all 20 matches, with
121 dead-ball inputs and 108 team-possession inputs. Duplicate and unused inputs and
redundant centroid averages were removed after retraining checks. See the
[feature inventory](FEATURES.md), [evaluation](RESULTS.md),
[validation details](reports/validation.json), and [training provenance](reports/training.json).
