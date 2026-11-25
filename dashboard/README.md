# 📈 Results Dashboard

Interactive Streamlit dashboard for the MSc thesis
**"Few-Shot Learning Paradigms for Bird Species Classification under Data Scarcity"**.

## Run locally

From the repository root:

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

The app opens at <http://localhost:8501>. Use the ⚙️ button (top right) to switch between
Birds-20, CUB-200 or both.

## Pages

| Page | Description |
|------|-------------|
| **Home** | Headline metrics, datasets and core findings |
| **Method Ranking** | Grouped bar chart of all methods at a chosen K and split |
| **K-Shot Efficiency** | Accuracy vs K (1, 3, 5) and the marginal gain per extra shot |
| **Backbone Comparison** | ResNet-50 vs DeiT-S |
| **Distance Ablation** | Euclidean vs Manhattan; the gap collapses for DeiT-S |
| **Per-Class Heatmap** | Novel-species difficulty per method |
| **Cross-Benchmark** | Ranking transfer from Birds-20 B15/N5 to CUB-200 B150/N50 |
| **Literature** | CUB B100/N50 results next to published few-shot methods |

## Data sources

Everything is read from the repository's `results/` folder:

- `results/all_results_master.csv`: one row per experiment (140+ rows)
- `results/**/summary.json`: scanned recursively for per-species accuracy

When an experiment is re-run and its `summary.json` regenerated, every chart picks up the
new numbers on the next launch.

## Layout

```
dashboard/
├── app.py              # navigation router (entry point)
├── pages/              # 00_home … 07_literature
├── components/         # charts.py (Plotly), filters.py, cards.py
├── data/loader.py      # cached loading and normalisation of results
└── assets/style.css
```

The theme lives in `.streamlit/config.toml` at the **repository root**, because Streamlit
reads it from the directory the app is launched from (the repo root, both locally and on
Streamlit Community Cloud).

## Deploy on Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub and choose **Create app → Deploy a public app from GitHub**.
2. Repository: `gmMustafa/Few-Shot-Learning-Paradigms-for-Bird-Species-Classification-under-Data-Scarcity`, branch `main`, main file path `dashboard/app.py`.
3. Under **Advanced settings**, pick Python 3.11. No secrets are needed.
4. Deploy. Dependencies are installed from `dashboard/requirements.txt`, the requirements file next to the entry point.

The `results/` folder is part of the repository, so no extra data hosting is needed.
After deployment, every push to `main` redeploys the app automatically.
