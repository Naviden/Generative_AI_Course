# Repository audit — 10 November 2026

**Layout update, 12 November 2026:** the paths below reflect the move into `archive/`. The findings remain the original content audit; relocation does not imply that unverified examples were repaired.

Base reviewed: `f90f602` (19 May 2024). The original README had one sentence; there was no syllabus, prerequisites, assessment, supported setup, test suite or CI. The old requirements were a 2023 environment freeze, including both TensorFlow and tensorflow-macos, and omitted notebook imports such as NLTK, sklearn-crfsuite and imageio.

## Supported course path

The three-class syllabus, notes, reading guide, assessments and four `genai_course` labs form the required teaching path. Core dependencies are separated from optional legacy frameworks. Mathematical regression tests and end-to-end CPU smoke tests cover the new path. Original topic directories are now available under `archive/`; their presence does not imply they are part of the required curriculum.

## Original material: findings and disposition

| Material | Finding | Disposition |
|---|---|---|
| `archive/Transformers/attention_calculation.py` | Unscaled dot-product scores and overflow-prone softmax | Fixed scaling and stable row-wise normalization; tested against known values |
| `archive/Neural Generative Models/VAE/VAE.py` | KL averaged over latent coordinates, changing the objective's scale; custom training step with validation but no matching custom test step | KL reduction corrected. Original training/validation remains unverified; tested PyTorch VAE is used in class |
| `archive/Neural Generative Models/GAN/DCGAN.py` | Linear discriminator output with probability-mode binary cross-entropy; trainability toggling and a very long loop | Loss corrected to `from_logits=True`; comment corrected. Trainability and full training remain unverified; not in required labs |
| `archive/Neural Generative Models/GAN/GAN.py` | Long MNIST training at import time; image shape conventions differ between real and generated data | Legacy, not executed or assigned; review shapes and add bounded CLI execution before reuse |
| VAE/GAN “from scratch” notebooks | Forward/loss sketches without a complete learning implementation | Illustrative sketches, not working training labs; use new VAE lab for optimization |
| `archive/LM/Statistical_Language_Models.ipynb` | Bare `Alice.txt` path differs from `archive/data/Alice.txt`; missing original dependencies; categorical IDs fed to `MultinomialHMM`; sampler confuses emitted word IDs with hidden states | Incomplete archival notebook. Do not use as a functioning HMM language generator. Correct the emission API and conditional hidden-state inference before promoting it |
| `archive/other/seq2se1.py` | Uses a space token absent from the encoder vocabulary; conflates start/stop/padding conventions; only three training pairs | Incomplete sequence-to-sequence sketch, not a credible translation experiment |
| `archive/other/RNN.py` | IMDB sentiment classifier, not a generative model; download and training at import | Optional discriminative comparison only |
| `archive/LM/BERT.py` | Base hidden states incorrectly described as scores that could become class probabilities | Explanation corrected. Pretrained model/pipeline downloads remain optional and untested |
| `archive/LM/GPT2.py` | Historically useful generation demo; no controlled evaluation or pinned model revision | Optional historical example, not a current frontier model or core research lab |
| Traditional GMM, HMM and naive Bayes scripts | Useful foundations; external libraries, plotting and uncontrolled execution paths | Optional prerequisite examples, not covered by core execution tests |
| GMM GIF notebooks and assets | Exploratory/duplicate materials and local plotting conventions | Preserved as archival material; schema validation only |
| Existing book/tutorial PDFs and `archive/data/Alice.txt` | Third-party resources; provenance and redistribution status not established by this refresh | Retained unchanged; not required by new labs. Confirm rights/provenance before redistributing course bundles |

The categorical/multinomial distinction follows [hmmlearn's official documentation](https://hmmlearn.readthedocs.io/en/stable/auto_examples/plot_multinomial_hmm.html). Optional dependency groups include missing imports but do not repair all legacy program semantics.

## Scope and readiness

This refresh supplies a research-oriented 12-hour teaching package. It does not certify every historical file as correct, execute full legacy neural-network training, or claim the course replaces a doctoral semester. The remaining legacy limitations are intentionally excluded from the required course path. Read the [validation record](validation.md) for executed checks and remaining platform limits.
