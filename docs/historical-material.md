# Historical and optional material

[Start the numbered lessons](../lessons/README.md). None of the folders below is another required lesson. They now live together under [archive/](../archive/README.md), preserving the original repository's examples and references; some are incomplete or have different dependencies. The [file-level audit](repository-audit.md) explains specific limitations.

| Folder | Topic / possible use after the core path | Status and route to current material |
|---|---|---|
| [Traditional Generative Models](../archive/Traditional%20Generative%20Models/README.md) | Naive Bayes, HMM and GMM refreshers | Optional historical illustrations; not covered by core execution tests |
| [Neural Generative Models](../archive/Neural%20Generative%20Models/README.md) | Original VAE and GAN scripts/notebooks | Some sketches lack complete training; use Lesson 01's tested VAE for the required experiment |
| [Transformers](../archive/Transformers/README.md) | Small historical attention calculation | Scaling and softmax were corrected; Lesson 01 provides the guided causal-attention exercise |
| [LM](../archive/LM/README.md) | Historical BERT/GPT-2 demos and statistical LM notebook | Downloads/optional dependencies; the HMM notebook has unresolved semantic issues |
| [other](../archive/other/README.md) | RNN classifier and sequence-to-sequence sketch | Optional comparisons, not complete generative-model lessons |
| [Books and Tutorials](../archive/Books%20and%20Tutorials/README.md) | Inherited reference PDFs | Not required reading; third-party rights remain applicable |
| [data](../archive/data/README.md) | Inherited `Alice.txt` | Used by archival experiments; core labs generate synthetic data |

The historical topic folders have been physically moved out of the repository root. Their internal folder structure and assets are preserved, while local documentation links and the historical-attention test use the new paths. Old unpinned external links must add `archive/` to the path; links pinned to earlier commits still refer to the original layout. See the [archive map](../archive/README.md) for the move list.

The 2023 environment freeze is also archived at [archive/requirements/legacy-2023.txt](../archive/requirements/legacy-2023.txt). Install the current core environment from `requirements/core.lock`, not this historical snapshot.

Archiving does not certify historical scripts as correct or ready to run. Known limitations in the file-level audit remain applicable.
