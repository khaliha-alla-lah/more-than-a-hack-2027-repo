# Release verification

The production entry point is `python -m possession.submission`. The two checkpoint
files and their hashes are in `models/submission/manifest.json`.

Required inputs are `BAR-ATH_tactical_atd.xml` and `BAR-ATH_tactical_atd.txt` from
the original competition download. Supply the dataset root or tactical folder with
`--data`; paths are not tied to a developer's machine. Use `--output` for any CSV path.

The frozen CSV hash is
`fa313d146666e97536e86b33eb3555e05e6823efb1c91afc2f8941cd13fffe7b`.
This identifies the file to upload, independently of its filename. Models were
repackaged into repository namespaces without changing their fitted parameters.

Verification for this release uses a fresh environment and an exported tracked-file
snapshot. Run the export
from that snapshot against original inputs and compare the SHA-256. Successful export
requires exact equality; the previous destination is preserved if verification fails.

The upload PDF can be generated from `docs/TECHNICAL.md` with the document helper
in `tools/build_submission_pdf.py`. PDF-generation dependencies are optional and
are not needed for inference. CSV and PDF uploads stay in ignored `submission-output/`.

The repository URL entered in the form must be public and refer to the reviewed
release after the team merges it. This branch does not submit the form or change
repository visibility.
