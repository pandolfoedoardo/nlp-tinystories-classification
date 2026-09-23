# nlp-tinystories-classification
Multi-label text classification and narrative feature tagging on the TinyStories dataset (~2.74M stories) using PyTorch, Hugging Face Transformers, and scikit-learn:
* **Data Curation & Imbalance Mitigation:** Empirical distribution analysis across six narrative tags and implementation of an optimal subset downsampling algorithm.
* **Comparative Architecture Benchmarking:** Systematic evaluation of three paradigms:
  * *TF-IDF + Logistic Regression:* Multi-output baseline with hyperparameter regularization search.
  * *DistilBERT **Fine-Tuning**:* Sequence classification with tokenization, cross-entropy multi-label loss, and PR-curve threshold calibration.
  * *Custom 1D-CNN:* Multi-scale convolutional feature extractors comparing single-branch vs. dual-branch (local vs. global receptive fields) architectures.
* **Performance vs. Efficiency Analysis:** Evaluation across per-tag F1-scores, accuracy, training footprint, and parameter efficiency (600k vs. 5.3M vs. 67M parameters).
