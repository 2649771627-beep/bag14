# Timing identifiability in opposite-direction LiDAR–radar calibration

This repository accompanies the study of when opposite-direction pairing helps estimate LiDAR–radar time offset and when a direction-dependent observation-centre shift can be mistaken for a timing offset. The replay compares the paired and framewise estimators on matched observations, known-offset simulations and controlled perturbations of recorded data.

## Files

- [`07_Reviewer_Reproduction.zip`](07_Reviewer_Reproduction.zip) contains the standalone replay, pinned dependencies, derived inputs, baseline outputs, supplements and publication index. Its SHA-256 is `c6cb2b3e3c2e03c39a97ed8831cad82692b69778411e232596062118d32339e7`. The archive preserves the checksums and provenance of the reviewed release, including notes from earlier local replays.
- [`source/`](source) provides a browsable copy of the Python source and dependency specifications with the archive's directory layout. Use the complete ZIP to run replays that require the frozen numerical inputs and expected results.

Entry points include the [full replay](source/run_all.py), [Table 6 profile replay](source/run_table6.py), [AIRY point-timing replay](source/run_point_timing.py), [public-dataset evidence replay](source/run_public_evidence.py) and [numerical verification](source/verify_results.py).

## Reproduce the results

Use Python 3.12. From the repository folder, extract the ZIP and install its pinned dependencies in your Python environment.

```sh
python -m zipfile -e 07_Reviewer_Reproduction.zip replay
cd replay
python -m pip install -r requirements.txt
python -B run_all.py --table6-only
```

The Table 6 command validates the input ledger and writes the eight refitted profiles to `generated/table6.csv`. For the full replay, run `python -B run_all.py --workers 6`. To reuse the simulation trials and earlier replay outputs while recomputing the current numerical analyses, use `python -B run_all.py --reuse-simulation --skip-legacy`. The extracted `README.md` and `PUBLICATION_INDEX.csv` locate the evidence for each table and figure.

## Data scope and terms

The field inputs consist of extracted trajectories, candidate radar returns, timestamps and fixed pair identifiers. The archived field comparisons describe how the fitted offset responds to timing conventions and spatial perturbations. The simulations use planted offsets to evaluate absolute error under the stated simulation conditions. Consult `DATA_SOURCES.md` inside the ZIP for recording provenance, data scope and the distinction between original recordings and numerical exports.

The ZIP includes derived material from the [Radar Ghost Dataset v1.1](https://doi.org/10.5281/zenodo.6676246) by Florian Kraus, Nicolas Scheiner, Werner Ritter and Klaus Dietmayer. The source dataset and retained derivatives are subject to [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The original format is documented in the [Radar Ghost repository](https://github.com/flkraus/ghosts). The author's source-code rights remain with their holder.
