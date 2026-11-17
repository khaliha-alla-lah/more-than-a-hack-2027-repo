# Source and data provenance

The selected pipeline adapts the EPTS loader, tracking features, symmetric gradient-
boosted classifiers, half-wise extraction and sequence decoder from:

https://github.com/hadisotudeh/more-than-a-hack-2027/tree/e733b0864653870ca5820d0ed7a82ee01ad75291/model

The retained files live in `possession/vendor/tracking/`. The source repository's LICENSE
and its original PROVENANCE.md are retained there. Its root LICENSE states MIT for
repository contents; its provenance separately notes that the original snapshot
underlying adapted core code supplied no license and asserts none for those portions.
Those upstream notices are preserved; this release does not claim original authorship
or expand the stated licensing of those portions.

Local changes: namespace relocation; checksum-verified checkpoint loading; ball-
reliability warnings; reliability-weighted stopped-play and team predictions; a
standalone export command; and balanced training on clean, missing-ball and false-
observation examples. Fitted numeric model parameters are unchanged by namespace
repackaging, and the exact frozen submission checksum is verified after export.

Training uses the 20 matches in SkillCorner open data at revision
`4340d274572876239c154c90bc507a9b3250a656`:
https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656

Challenge match assets remain separately licensed and are not redistributed. Only
the learned checkpoints, code and aggregate evaluation results are included here.
Older optional pipeline components retain their existing notices.
