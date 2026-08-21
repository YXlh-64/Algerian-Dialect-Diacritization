# Paper Dev Metrics

All systems use the same 607-sentence released dev split and the metric definitions embedded in each JSON artifact.

| Model | Variant | Accuracy | Macro F1 | WER | CER | Word Acc. | Sentence Acc. | Shadda Acc. | Tanween Acc. | char-BLEU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ConvLocal Transformer | neural | 0.917469 | 0.490574 | 0.255814 | 0.046494 | 0.744186 | 0.260297 | 0.979996 | 0.999937 | 0.890395 |
| ConvLocal Transformer | v2 | 0.936089 | 0.520962 | 0.197933 | 0.035909 | 0.802067 | 0.347611 | 0.985343 | 0.999874 | 0.915528 |
| DziriFormer-Large-11M | neural | 0.914764 | 0.509367 | 0.264083 | 0.047532 | 0.735917 | 0.228995 | 0.980374 | 0.999937 | 0.886843 |
| DziriFormer-Large-11M | v2 | 0.933447 | 0.520539 | 0.204910 | 0.037240 | 0.795090 | 0.319605 | 0.984588 | 0.999874 | 0.912233 |
| DziriFormer-J16-Gated-v3 | neural | 0.917469 | 0.431505 | 0.255039 | 0.046266 | 0.744961 | 0.248764 | 0.980059 | 0.999874 | 0.891042 |
| DziriFormer-J16-Gated-v3 | v2 | 0.935397 | 0.520183 | 0.201809 | 0.036526 | 0.798191 | 0.327842 | 0.983896 | 0.999874 | 0.914293 |
| DziriFormer-GL-v3 | neural | 0.916399 | 0.488038 | 0.258398 | 0.046461 | 0.741602 | 0.253707 | 0.980185 | 0.999874 | 0.890965 |
| DziriFormer-GL-v3 | v2 | 0.933824 | 0.518619 | 0.205685 | 0.036591 | 0.794315 | 0.324547 | 0.984966 | 0.999874 | 0.913361 |
| DziriFormer-Mixed-v3 | neural | 0.918601 | 0.496843 | 0.254264 | 0.045682 | 0.745736 | 0.263591 | 0.980248 | 0.999937 | 0.892101 |
| DziriFormer-Mixed-v3 | v2 | 0.935522 | 0.517871 | 0.202584 | 0.036266 | 0.797416 | 0.327842 | 0.984651 | 0.999874 | 0.914200 |
| DziriFormer-Hier-v4 | neural | 0.919104 | 0.506534 | 0.252196 | 0.045162 | 0.747804 | 0.270181 | 0.981191 | 0.999937 | 0.892849 |
| DziriFormer-Hier-v4 | v2 | 0.935963 | 0.526876 | 0.201292 | 0.035617 | 0.798708 | 0.331137 | 0.986224 | 0.999874 | 0.914988 |
| DziriFormer-HierMixed-v4 | neural | 0.923130 | 0.515062 | 0.240827 | 0.043052 | 0.759173 | 0.283361 | 0.981066 | 0.999937 | 0.898525 |
| DziriFormer-HierMixed-v4 | v2 | 0.936277 | 0.525152 | 0.199483 | 0.035812 | 0.800517 | 0.336079 | 0.985217 | 0.999874 | 0.915381 |
| DziriFormer-Direct16-v3 | neural | 0.917846 | 0.427179 | 0.257881 | 0.046234 | 0.742119 | 0.243822 | 0.979870 | 0.999874 | 0.889888 |
| DziriFormer-Direct16-v3 | v2 | 0.933950 | 0.513188 | 0.205943 | 0.037240 | 0.794057 | 0.306425 | 0.983645 | 0.999874 | 0.911889 |
| DziriFormer-GL-Curriculum-v4 | neural | 0.918161 | 0.467976 | 0.252713 | 0.045877 | 0.747287 | 0.263591 | 0.980185 | 0.999874 | 0.891303 |
| DziriFormer-GL-Curriculum-v4 | v2 | 0.933635 | 0.514938 | 0.205685 | 0.037208 | 0.794315 | 0.329489 | 0.984525 | 0.999874 | 0.911911 |
| DziriFormer-HGL-v4 | neural | 0.923445 | 0.501973 | 0.243669 | 0.043149 | 0.756331 | 0.270181 | 0.981129 | 0.999937 | 0.898068 |
| DziriFormer-HGL-v4 | v2 | 0.936340 | 0.524485 | 0.202842 | 0.035844 | 0.797158 | 0.322900 | 0.985092 | 0.999874 | 0.915108 |
| DziriFormer-DualRoPE-CE-v6 | neural | 0.928729 | 0.445779 | 0.227907 | 0.040455 | 0.772093 | 0.285008 | 0.982324 | 0.999874 | 0.904528 |
| DziriFormer-DualRoPE-CE-v6 | v2 | 0.940177 | 0.526876 | 0.190698 | 0.033734 | 0.809302 | 0.354201 | 0.985532 | 0.999874 | 0.920800 |
| DziriFormer-DualRoPE-CRF-v7 | neural | 0.932000 | 0.451328 | 0.214729 | 0.038377 | 0.785271 | 0.296540 | 0.983204 | 0.999874 | 0.909737 |
| DziriFormer-DualRoPE-CRF-v7 | v2 | 0.941184 | 0.528078 | 0.187080 | 0.033052 | 0.812920 | 0.352554 | 0.985783 | 0.999874 | 0.922334 |
| DziriFormer-DualRoPE-BoundaryCRF-v8 | neural | 0.933321 | 0.453566 | 0.214212 | 0.037500 | 0.785788 | 0.306425 | 0.983833 | 0.999874 | 0.910994 |
| DziriFormer-DualRoPE-BoundaryCRF-v8 | v2 | 0.942127 | 0.531023 | 0.186822 | 0.032403 | 0.813178 | 0.359143 | 0.986413 | 0.999874 | 0.923247 |
| DziriFormer-DualRoPE-WordPos-CRF-v10 | neural | 0.931434 | 0.522380 | 0.217829 | 0.038799 | 0.782171 | 0.288303 | 0.983708 | 0.999937 | 0.909282 |
| DziriFormer-DualRoPE-WordPos-CRF-v10 | v2 | 0.940555 | 0.530153 | 0.187339 | 0.033344 | 0.812661 | 0.344316 | 0.985846 | 0.999874 | 0.922041 |
| DziriFormer-DualRoPE-FactorizedEmission-CRF-v10 | neural | 0.929735 | 0.514262 | 0.217829 | 0.039221 | 0.782171 | 0.304778 | 0.983393 | 0.999937 | 0.908347 |
| DziriFormer-DualRoPE-FactorizedEmission-CRF-v10 | v2 | 0.940932 | 0.528449 | 0.186822 | 0.032890 | 0.813178 | 0.362438 | 0.986916 | 0.999874 | 0.922281 |
| DziriFormer-DualRoPE-LowRankBoundaryCRF-v10 | neural | 0.930993 | 0.448995 | 0.213953 | 0.039058 | 0.786047 | 0.326194 | 0.982638 | 0.999874 | 0.908793 |
| DziriFormer-DualRoPE-LowRankBoundaryCRF-v10 | v2 | 0.940681 | 0.526659 | 0.188630 | 0.033442 | 0.811370 | 0.359143 | 0.985595 | 0.999874 | 0.921510 |
| DziriFormer-DualRoPE-CRF-EmissionRDrop-v13 | neural | 0.933824 | 0.453511 | 0.209819 | 0.037273 | 0.790181 | 0.319605 | 0.984085 | 0.999874 | 0.911548 |
| DziriFormer-DualRoPE-CRF-EmissionRDrop-v13 | v2 | 0.943008 | 0.531358 | 0.181395 | 0.031981 | 0.818605 | 0.365733 | 0.986601 | 0.999874 | 0.924385 |

Registered neural models: 17.
Each model has a neural row and an unchanged V2 lexical-fallback row.
Every skeleton mismatch count is expected to be zero.

## Recomputed-score integrity notes

Fresh decoded predictions are authoritative for every paper metric. Historical checkpoint differences of more than one letter fail the report.

- DziriFormer-HGL-v4: stored 14681, freshly decoded 14680 (-1 letter).
