# Anatomy of a Housing Market

A notebook-based statistics and machine-learning project using the Ames Housing dataset. It follows the data from cleaning and exploratory analysis through sampling, inference, regression, and classification.

## Status

| Notebook | Topic | Status |
| --- | --- | --- |
| [01 — Exploratory data analysis](notebooks/01_exploratory_data_analysis.ipynb) | Cleaning, univariate and bivariate EDA | Complete |
| [02 — Data and sampling distributions](notebooks/02_data_and_sampling_distributions.ipynb) | Sampling, bootstrap, CLT, and QQ plots | Ongoing |
| 03 — Experiments and testing | Hypothesis testing and A/B testing | Planned |
| 04 — Regression and prediction | Linear and multiple regression | Planned |
| 05 — Classification | Classification models and evaluation | Planned |
| 06 — Statistical ML | Statistical machine-learning methods | Planned |

## Highlights

- `SalePrice` is strongly right-skewed; `log1p(SalePrice)` is much closer to normal for downstream modelling.
- Overall quality (`OverallQual`) is the strongest observed price predictor, followed by living area (`GrLivArea`).
- Simulated sampling distributions show the Central Limit Theorem: sample means become more normal and less variable as sample size grows.
- Bootstrap resampling estimates the median sale price's uncertainty without relying on a closed-form standard error.

See the notebooks for the full methodology, figures, and interpretations.

## Run locally

Tested with Python 3.13.13.

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook 01_exploratory_data_analysis.ipynb
```

Run the first notebook top to bottom to fetch the dataset and create `data/cleaned_data.csv`, then open and run `02_data_and_sampling_distributions.ipynb`. The notebooks save figures under `figures/`.

## Dataset

The [Ames Housing dataset](https://www.openml.org/search?type=data&sort=runs&id=42165) contains 1,460 residential-property sales in Ames, Iowa (2006–2010), with 79 explanatory features and the `SalePrice` target. Notebook 01 loads it through scikit-learn's `fetch_openml`; see [data_description.txt](data_description.txt) for the full data dictionary.

## Topics covered

Data cleaning, exploratory visualization, correlation analysis, transformations and distribution diagnostics, Monte Carlo simulation, bootstrap confidence intervals, and reproducible notebook workflows.
