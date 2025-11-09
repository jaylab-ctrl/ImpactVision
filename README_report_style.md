<p align="center">
  <img src="assets/hero.jpg" alt="Road safety waveform" width="80%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-informational" />
  <img src="https://img.shields.io/badge/Type%20of%20ML-Classification-blue" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" /></a>
  <a href="https://colab.research.google.com/"><img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
  <img src="https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red" />
</p>

# Road Safety Analytics

This is an end‑to‑end project analyzing and modeling the **Great Britain personal‑injury road accidents (2005–2014)** dataset. It combines **information visualization** (EDA + dashboards) with **machine learning** (severity prediction), and optional **containerized deployment**.

> Based on the final reports: *Info Viz Project Report* and *ML1 Project Report (Final)* (both included under `/reports`).

---

## Authors for this project
- **Jay Sarode**

---

## Table of Contents
- [Business problem](#business-problem)
- [Data source](#data-source)
- [Dataset attributes](#dataset-attributes)
- [Methods](#methods)
- [Tech Stack](#tech-stack)
- [Results (from the reports)](#results-from-the-reports)
- [Lessons learned](#lessons-learned)
- [Limitations & what can be improved](#limitations--what-can-be-improved)
- [Notebooks](#notebooks)
- [Interactive App (Dash)](#interactive-app-dash)
- [Docker & Cloud](#docker--cloud)
- [Contribution](#contribution)
- [License](#license)

---

## Business problem
Build visual analytics and ML models to **understand patterns and predict accident/casualty severity** so stakeholders can prioritize interventions (e.g., road design, speed policy, policing).

---

## Data source
- Great Britain road accidents (2005–2014), joined from `Accident`, `Vehicle`, `Casualty`, and `Lookup` files.
- Join key: `Accident_Index`; categorical codes decoded through lookup (fuzzy match used in places in the report).
- Place original CSVs under `data/` (kept out of git).

---

## Dataset attributes
High‑value attributes used across EDA and models include:
- **Temporal:** Year, Month, Day-of-week, Hour.
- **Location/Context:** Urban/Rural, Speed limit, Road type, Junction detail, Carriageway hazards.
- **Environment:** Light conditions, Weather conditions, Road surface.
- **Actors:** Vehicle type, Point of impact, Casualty class/age/sex.
- **Targets:** *Accident Severity* and *Casualty Severity.*

> Cleaned and merged dataset referenced in the reports has multi‑million rows (≈4.3M) and ~67 columns.

---

## Methods
- **Information Visualization:** static plots, correlation/cluster maps, pair plots, distribution plots (hist/KDE/Q‑Q), 3D & contour; interactive dashboard using **Dash**.
- **Preprocessing:** missing‑value handling, special‑character cleanup, **IQR outlier** mitigation, **SMOTE** for class imbalance.
- **Dimensionality reduction & signals:** **PCA** (~90% cumulative variance within a few dozen components), random‑forest feature importance, SVD diagnostics.
- **Models:** Decision Tree, Logistic Regression, K‑Nearest Neighbors, Gaussian NB, **SVM** (OVR/OVO), **MLP**, **Random Forest**, and ensembles (bagging/stacking/boosting).
- **Evaluation:** accuracy, precision, recall, F1, ROC curves, confusion matrices.

---

## Tech Stack
- **Python**: 3.10+
- **EDA/ML**: numpy, pandas, matplotlib, plotly, scikit‑learn, imbalanced‑learn
- **Dashboard**: Dash (core + html components)
- **Ops**: Docker (optional), Google Cloud (optional)

---

## Results (from the reports)

> The bullets and figures below are **copied/condensed from the two final reports**. Add the corresponding images (exported from `/reports`) into `assets/` and they will render here in the README.

### Key insights from Information Visualization
1. **Accidents & casualties decreased** steadily from 2005→2014.  
2. **Urban areas** see substantially more accidents than rural areas.  
3. **Two‑vehicle incidents dominate**; **cars** are the most involved vehicle type.  
4. Most accidents occur in **daylight** and on **single carriageways**; **30 mph** zones have the highest counts.  
5. Weather/lighting effects are visible but secondary compared to **speed limit** and **road type**.

<p>
  <img src="assets/trend_accidents.png" width="49%" alt="Accidents per year"/>
  <img src="assets/urban_rural.png" width="49%" alt="Urban vs Rural"/>
</p>

### Modeling outcomes (classification of severity)
- **Imbalance handling:** SMOTE improved minority‑class recall without a large drop in overall accuracy.  
- **Dimensionality:** PCA reached ~90% variance retention in ~30–40 components (as reported).  
- **Top performers:** **Random Forest** and **SVM** variants were strongest; **MLP** competitive.  
- **Diagnostics:** ROC curves (OVR/OVO) and confusion matrices show better discrimination post‑SMOTE.  

> If you prefer a compact table, paste the model comparison exported from the report here:

| Model | Notes from report |
|---|---|
| Decision Tree | Baseline, interpretable |
| Logistic Regression | Competitive on linear signal |
| KNN | Sensitive to scaling/k |
| Naïve Bayes | Fast baseline |
| **SVM (OVR/OVO)** | Strong classifier |
| **Random Forest** | Best overall in the report |
| **MLP** | Competitive after tuning |
| Ensembles | Bagging/stacking/boosting evaluated |

<p>
  <img src="assets/pca_variance.png" width="32%" alt="PCA variance" />
  <img src="assets/rf_importance.png" width="32%" alt="RF feature importance" />
  <img src="assets/roc_multi.png" width="32%" alt="ROC OVR/OVO" />
</p>

> **Tip:** Export figures from the PDFs into the `assets/` folder with the above filenames to auto‑render in this README.

---

## Lessons learned
- Large geospatial/temporal accident datasets demand **chunked IO** and careful **data typing**.  
- **Balancing** (SMOTE) is crucial for realistic severity modeling.  
- **Feature engineering** around speed limits, road types, and temporal patterns provides sizable signal.  
- Dash greatly simplifies **interactive explainability** for stakeholders.

---

## Limitations & what can be improved
- Limited external context (e.g., **traffic volume**, **weather intensity**, **road geometry**) restricts causal interpretation.  
- Extend to **multi‑year backtesting** and **cost‑sensitive** training.  
- Explore **XGBoost/LightGBM** and **calibrated probabilities** for decision support.  
- Deploy with telemetry to track **data drift** and re‑training triggers.

---

## Notebooks
- EDA & visualization: `notebooks/01_eda_visualization.ipynb`
- Modeling: `notebooks/02_modeling.ipynb`

## Interactive App (Dash)
```bash
cd app
python app.py
```
The app starts locally (default `:8050`).

## Docker & Cloud
```bash
docker build -t road-safety-dash .
docker run -p 8050:8050 road-safety-dash
```
Optional: push to GCP (Cloud Run / GKE).

---

## Contribution
Pull requests and issues are welcome. Please open a discussion for larger changes.

## License
MIT — see [LICENSE](LICENSE).
