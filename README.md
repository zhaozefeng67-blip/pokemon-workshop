# Pokémon Analytics Dashboard

Streamlit dashboard for the course Pokémon dataset: dual-type analysis, type profiles, capture difficulty, and a logistic-style legendary classifier (SGD).

## Requirements

- Python 3.9 or newer
- Dependencies: `pip install -r requirements.txt`

Third-party libraries: [Streamlit](https://streamlit.io/), [pandas](https://pandas.pydata.org/), [Plotly](https://plotly.com/python/), [scikit-learn](https://scikit-learn.org/).

## Data

Do **not** ship the original `pokemon.csv` in the submission zip (course rule). For local runs, put `pokemon.csv` in the project root (same folder as `app.py`). The file is the raw Kaggle-style table used in class.

Cleaning (type2 normalization, capture rate parsing, BST tiers) is in `dashboard/data/cleaning.py` and runs when the app loads or trains the model.

## Run the app

```bash
cd workshop
pip install -r requirements.txt
streamlit run app.py
```

Open the URL printed in the terminal (usually `http://localhost:8501`). Run the command from the project root so `import dashboard` works.

If an old Streamlit process is still running, stop it (Ctrl+C in that terminal) before starting again.

## Machine learning artifact

The ML page loads `artifacts/ml_results_30.pkl` if present (70/30 stratified split, `random_state=123`). To rebuild:

```bash
python train_ml_artifacts.py
```

Training logic lives in `dashboard/analysis/ml.py`; the script above only calls it and writes the pickle. Include both the script and the `.pkl` in the zip if size allows; otherwise upload the pickle elsewhere and put the download link in this README.

## Project layout

```
app.py                 # entry: streamlit run app.py
train_ml_artifacts.py  # optional: regenerate ml_results_30.pkl
dashboard/
  config.py            # paths, constants
  data/                # load CSV, cleaning helpers
  analysis/            # aggregations + ML training
  plots/               # Plotly figures
  ui/                  # pages, widgets, CSS
combined_project3.0.ipynb   # notebook version of the four analysis parts
```

## Submission notes

- Zip: code + report. Omit `pokemon.csv`.
- Report (separate file): describe each member’s contribution; not duplicated here.
- Optional: include `artifacts/ml_results_30.pkl` (~210 KB) so markers can open the ML tab without training.
