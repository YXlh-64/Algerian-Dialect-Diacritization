# Lyes Track 4 strategy results

All released-dev scores below use the same 607 sentences and 15,897 scored
letters. `Neural` is the model-only accuracy; `V2` applies the fixed,
training-only lexical fallback. The detailed WER, CER, macro-F1, word,
sentence, Shadda, Tanween, and char-BLEU values are in
[`results/PAPER_METRICS.md`](results/PAPER_METRICS.md).

## Architecture progression

| Strategy | Neural | V2 | Outcome |
|---|---:|---:|---|
| ConvLocal Transformer | 0.917469 | 0.936089 | Initial baseline |
| DziriFormer Large 11M | 0.914764 | 0.933447 | Rejected |
| J16 Gated v3 | 0.917469 | 0.935397 | Retained as legacy ensemble expert |
| GL v3 | 0.916399 | 0.933824 | Retained as legacy ensemble expert |
| Mixed v3 | 0.918601 | 0.935522 | Retained as legacy ensemble expert |
| Hier v4 | 0.919104 | 0.935963 | Retained as legacy ensemble expert |
| HierMixed v4 | 0.923130 | 0.936277 | Accepted multi-seed architecture |
| Direct16 v3 | 0.917846 | 0.933950 | Accepted stability ablation |
| GL Curriculum v4 | 0.918161 | 0.933635 | Accepted stability ablation |
| HGL v4 | 0.923445 | 0.936340 | Accepted; improved all three matched seeds |
| DualRoPE CE v6 | 0.928729 | 0.940177 | Accepted encoder |
| DualRoPE CRF v7 | 0.932000 | 0.941184 | Accepted decoder; primary final ensemble |
| BoundaryCRF v8 | 0.933321 | 0.942127 | Accepted standalone; final ensemble +1 letter |
| WordPos CRF v10 | 0.931434 | 0.940555 | Rejected |
| FactorizedEmission CRF v10 | 0.929735 | 0.940932 | Rejected |
| LowRankBoundary CRF v10 | 0.930993 | 0.940681 | Rejected |
| Emission R-Drop v13 | 0.933824 | 0.943008 | Standalone accepted; ensemble gate rejected |

## Controlled campaigns after v7

| Campaign | Result | Decision |
|---|---|---|
| v7 four-group ensemble + V2 | 14,977 / 15,897 (0.942127) | Approved primary submission |
| v8 BoundaryCRF replacement ensemble | 14,978 / 15,897 (0.942190) | Local gate passed; not promoted over v7 |
| v8 cross-fitted lexical gate | 14,973 / 15,897 | Rejected (-5) |
| v9 calibrated simplex stacking | 14,968 / 15,897 | Rejected (-9) |
| v10 snapshot average | 14,963 / 15,897 | Rejected (+1, below robustness gate) |
| v11 context-boundary model | 14,820 neural; 14,956 V2 | Rejected; gate collapsed almost always open |
| v11 equal five-group probe | 14,963 / 15,897 | Rejected (-14) |
| v12 BoundaryCRF SWA tail | 14,831 neural; 14,968 V2 | Rejected (-9 and protected regressions) |
| v13 R-Drop standalone | 14,845 neural; 14,991 V2 | Standalone gate passed |
| v13 replacement ensemble | 14,978 / 15,897 | Campaign rejected; required 14,987 |
| v14 filtered word lattice | Mean calibration gain +7 letters/+7 words | Rejected before released-dev evaluation |
| v15 context contrastive | 14,902 neural; 14,982 V2 | Rejected: V2 Shadda fell by two letters |

## Submission decision

The approved primary artifact remains the v7 final ensemble. Its local score
is 14,977/15,897 and its recorded SHA-256 is
`51dd19b57e3af6498f8b772725cb3737670ae2d44f83a720af0321bf3c625590`.
Generated CSV files and checkpoints are deliberately excluded from this
branch; the immutable experiment reports record their names, hashes, and
accept/reject decisions.
