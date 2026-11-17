# khaliha aala lah — technical approach

## Inputs and frame alignment

Inference uses the supplied tactical EPTS XML metadata and TXT tracking. Normalized
coordinates are converted to metres on the 105.3 by 68 m pitch. EPTS frame IDs are
one-based; output video frames are zero-based. Features and decoding reset at the
published half boundaries. Frames outside them are X. The published second-half
setup before kickoff is D, matching the retained boundary policy. No possession
annotations are read during inference.

## Features and models

Features describe ball movement and visibility, player proximity, relative team
shape, motion, and previous/next control evidence. Windows include future frames:
this is an offline algorithm. Short ball gaps up to 0.4 s may be interpolated;
long gaps remain missing. Both teams are represented symmetrically.

The baseline has two scikit-learn HistGradientBoostingClassifier models: stopped
play and home possession conditional on live play. A robust pair has the same
architecture. Each model uses 600 iterations, learning rate 0.05, up to 31 leaves,
minimum leaf size 400, L2 regularization 1, random seed 0, and no early stopping.

## External training

Both checkpoint pairs use all 20 SkillCorner open-data matches at revision
4340d274572876239c154c90bc507a9b3250a656. Home/away labels come from phases of play;
remaining frames within halves are assumed dead. These provider-derived labels
are imperfect. No provided-game hand labels enter fitting or selection.

The baseline uses clean examples sampled every second native 10 Hz frame. The
robust model sees three views: clean, added ball gaps drawn from the label-free
target gap-duration distribution, and false ball observations affecting 3% of
observed samples in 0.4–2 s bursts with coordinate offsets up to 12 m. The three
views are sampled every sixth native frame to preserve approximately the original
training sample count. Possession weights use an 8 s reference, clipped to 0.25–4.

## Reliability and temporal decoding

A ball observation is suspect when absent after feature cleaning, or near an
observed step faster than the existing 40 m/s limit. Such step warnings extend
0.4 s on either side and do not cross halftime. Stopped-play and conditional-team
probabilities use 25% baseline and 75% robust-model confidence in suspect regions;
otherwise the baseline is retained. Stationarity or distance from players alone
does not trigger this rule.

A three-state Viterbi decoder produces H/A/D intervals. Emission strengths are
0.75 for dead/live and 2 for team confidence. At 25 Hz, switching team costs 64,
stopping/restarting costs 40 each, and minimum dead duration is 25 frames. These
frame-dependent settings scale with frame rate. Exact inference runs from the
frozen checkpoints; retraining is unnecessary to reproduce the submission.

## Validation and limitations

Candidate selection used external development splits before full-data fitting.
In a repeated check on two DFL matches excluded from those candidate fits, mean
spellF1 with target-length gaps improved from 0.4435 to 0.5736; native tracking
scored 0.6650 versus 0.6641; false-position bursts scored 0.6572 versus 0.6583.
The check pools had prior project use and are not a fresh blind test. Halves are
averaged within matches; they are not independent additional matches.

After freezing and fitting on all external training matches, the public five-
minute pipeline diagnostic scored spellF1 0.6483 and macroF1 0.9089. It was not
used for training or parameter selection. Small wrong-position gains do not prove
that false detections are solved. These results do not establish private-match
performance or ranking.

## Reproduction and provenance

Install `requirements-submission.txt`, then run:

`python -m possession.submission --data <dataset_folder> --output "khaliha aala lah.csv"`

Expected output: 199,319 rows, exactly `frame,label`, SHA-256
`fa313d146666e97536e86b33eb3555e05e6823efb1c91afc2f8941cd13fffe7b`.
The loader verifies checkpoint hashes; the exporter verifies CSV format and checksum.
The technical implementation adapts the source credited in THIRD_PARTY_NOTICES.md;
original notices are retained. Match data and local research artifacts are excluded.
