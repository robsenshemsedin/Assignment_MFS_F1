# MFS baseline for WSD

Most Frequent Sense baseline computed from SemCor, evaluated on the
Raganato et al. (2017) test sets and the amended versions from
Maru et al. (2022).

## Data (not included)

- WSD Evaluation Framework: http://lcl.uniroma1.it/wsdeval/
  Unzip into the project root.
- hardEN and 42D: https://github.com/SapienzaNLP/wsd-hard-benchmark
  Copy those two folders into the project root.

## Running

    pip install nltk
    python -c "import nltk; nltk.download('wordnet')"
    python src/compare_baselines.py --root .
    python src/compare_baselines.py --root . --backoff none

Results are in `results/`. Write-up in `MFS_baseline_report.md`.