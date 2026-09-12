# Memory Model Benchmark for Spaced Repetition

The purpose of this repository is to provide reference implementations for the models presented in the following paper:

> A. Schill. 2026. [Interpretable Memory Models for Spaced Repetition](https://doi.org/10.5281/zenodo.22727105).
> Preprint, Zenodo. doi:10.5281/zenodo.22727106.

## Results on Test Set (Including Same-Day Reviews)

| Model           |  Fine-Tuned  |   Parameters |   State Dimensions |    BCE |   RMSE(bins) |    AUC |
|:----------------|:------------:|-------------:|-------------------:|-------:|-------------:|-------:|
| FSRS v7 3-state |      x       |           34 |                  3 | 0.3237 |       0.0577 | 0.7517 |
| SBD             |      x       |            7 |                  3 | 0.3279 |       0.0593 | 0.7441 |
| SBD Binary      |      x       |            7 |                  3 | 0.3296 |       0.0598 | 0.7329 |
| FSRS v7 3-state |              |           34 |                  3 | 0.3454 |       0.0900 | 0.7282 |
| SBD             |              |            7 |                  3 | 0.3469 |       0.0878 | 0.7224 |
| SBD Binary      |              |            7 |                  3 | 0.3483 |       0.0887 | 0.7057 |
