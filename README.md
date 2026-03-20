---
title: Air Quality Kloten
emoji: 🌬️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Air Quality Kloten

Interactive dashboard exploring air pollution patterns near Zürich-Kloten airport,
using four years of 10-minute interval monitoring data (2020–2023).

🔗 **[Live dashboard](https://huggingface.co/spaces/mdujella/air-quality-kloten)**

![Dashboard](Screenshot.png)

---

## Project overview

The monitoring station "Kloten - Feld" is located approximately 1 km east of
Zürich-Kloten airport's runway, in a residential neighbourhood exposed to both
airport and road traffic. The dataset covers the full Covid-19 pandemic period,
making it possible to observe the effect of reduced air and road traffic on
pollutant concentrations.

The project covers the full data science pipeline: data loading and preprocessing,
missing data analysis and interpolation, exploratory analysis, predictive modelling,
and an interactive dashboard to communicate findings.

---

## Data

Source: [Ostluft](https://www.ostluft.ch) / AWEL Canton Zürich  
Dataset: [Canton Zürich Open Data Catalogue](https://www.zh.ch/de/politik-staat/statistik-daten/datenkatalog.html#/datasets/2383@awel-kanton-zuerich)  
Period: 05.03.2020 – 31.12.2023  
Resolution: 10-minute intervals (~201,000 rows)

**Pollutants:** PM2.5, NOx, NO2, NO, CO2, SO2, eBC2.5, PN[5-100nm]  
**Meteorological parameters:** temperature (T), relative humidity (Hr), global
radiation (StrGlo), rain duration (RainDur), wind direction (WD), wind velocity
(WVv), and inlet measurements (T_Trockner, Hr_Trockner)

---

## Methodology

### Preprocessing and missing data

Missing data varied substantially across parameters: SO2 was missing for ~52% of
the dataset and excluded from modelling; PN[5-100nm] was missing ~18% and CO2
~11%. For the remaining parameters, a time-based interpolation strategy was applied
with variable limits depending on how stable each parameter is (e.g. up to 2 days
for PM2.5, 2 hours for PN[5-100nm]). Days where key pollutants were entirely absent
were removed. A sensor malfunction causing relative humidity values above 100% in
December 2021 was detected and corrected via interpolation. Interpolation flags were
retained as features for downstream modelling.

### Feature engineering

Wind direction (in degrees) was decomposed into x/y components and combined with
wind speed to produce directional wind vectors. Rolling 1-hour and 24-hour
aggregates (mean, median, max) were computed for all meteorological features. Binary
flags were added for weekends and nighttime hours, the latter motivated by Zürich
airport's strict night curfew (23:30–06:00). Lagged features (10 min, 1 hour, 2
hours) were also explored as predictors.

### Predictive modelling

Two targets were modelled separately: NOx and PM2.5. Two model types were compared:
Random Forest and HistGradientBoosting (scikit-learn). Models were trained on a
chronological 80/20 train-test split (no shuffling, to respect the time series
structure). Feature importance was assessed using permutation importance on the test
set.

**Results — weather features only:**

| Target | Model | R² | RMSE | MAE |
|--------|-------|----|------|-----|
| NOx | Random Forest | 0.453 | 8.07 | 4.46 |
| NOx | HistGradientBoosting | 0.561 | 7.23 | 3.95 |
| PM2.5 | Random Forest | 0.298 | 3.37 | 2.51 |
| PM2.5 | HistGradientBoosting | 0.337 | 3.27 | 2.38 |

**Results — with lagged features added:**

| Target | Model | R² | RMSE | MAE |
|--------|-------|----|------|-----|
| NOx | HistGradientBoosting | 0.877 | 3.82 | 1.80 |
| PM2.5 | HistGradientBoosting | 0.979 | 0.58 | 0.28 |

NOx is better explained by weather variables than PM2.5, which has a much stronger
autocorrelation structure. Adding lagged values dramatically improves both, with
PM2.5 in particular being almost entirely predictable from its own recent history.

### EDA highlights

- Large PM2.5 spikes are visible each New Year's Eve (fireworks), with a notably
  smaller spike in 2020/2021 due to the Omicron wave reducing celebrations
- PM2.5 concentrations are lower when wind blows from the S/SW/SE, consistent with
  the airport and motorway lying to the west and northwest
- NOx shows a clear weekday/weekend pattern and strong seasonal variation, with
  higher concentrations in winter
- During the Covid-19 lockdown in spring 2020, NOx and PN[5-100nm] concentrations
  were visibly reduced compared to the same period in other years

---

## Dashboard

The dashboard is built with Dash and Plotly and includes five tabs:

- **Initial EDA** — time series and distributions for all pollutants and
  meteorological parameters
- **Missing data** — interpolation overview per variable, showing observed,
  interpolated, and gap-too-long points
- **EDA and correlations** — average pollutant concentrations by hour, day of week,
  season, and day of year
- **Predictive modelling** — predicted vs actual plots and permutation importance
  charts for both models and both targets
- **Covid lockdown analysis** — comparison of pollutant concentrations during the
  2020 lockdown versus the same period in other years

---

## Running locally
```bash
# Clone the repo
git clone https://github.com/mdujella/air-quality-kloten.git
cd air-quality-kloten

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python particle_pollution_dashboard.py
```

Then open `http://127.0.0.1:7860` in your browser.

The raw CSV data files are not included in this repository. The dashboard runs on
preprocessed parquet files in `app_data/`. To reproduce the full pipeline, download
the raw data from the source above and run the notebooks in order.

---

## Repository structure
```
├── app_data/               # Preprocessed data used by the dashboard
├── assets/                 # Static images (permutation importance plots)
├── results/                # Model metrics and predictions (parquet + JSON)
├── particle_pollution_dashboard.py   # Main dashboard
├── Dockerfile
└── requirements.txt
```

---

## Tech stack

Python, Dash, Plotly, pandas, scikit-learn, scipy, joblib