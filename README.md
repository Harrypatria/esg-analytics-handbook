# ESG Analytics Handbook — Companion Notebooks

Companion code for **Data Analytics and AI in ESG, Climate Change and Sustainability: A Practitioner's Handbook**  
Patria, H., Anindika Yudhy, M.R., & Zulkifli, M. (2026). Convergeni by Patria & Co. · PT Artificial Intelligent Indonesia  
DOI: 10.5281/zenodo.23399742

---

## Contents

| Notebook | Chapter | Topic |
|---|---|---|
| `ch04_carbon_intensity.ipynb` | Chapter 4 | GHG accounting, carbon intensity, UCI Air Quality dataset |
| `ch07_machine_learning.ipynb` | Chapter 7 | ML workflow: regression, XGBoost, SHAP on UCI Air Quality |
| `ch08_nlp_esg_disclosure.ipynb` | Chapter 8 | NLP pipeline: ESG text classification, ClimateBERT |
| `ch09_geospatial_analytics.ipynb` | Chapter 9 | Geospatial analytics: leafmap, NDVI, land-cover change |

Python scripts (`.py`) are also provided for non-notebook environments.

---

## Setup

```bash
# Clone the repository
git clone https://github.com/Harrypatria/esg-analytics-handbook.git
cd esg-analytics-handbook/notebooks

# Install dependencies
pip install -r requirements.txt

# Launch JupyterLab
jupyter lab
```

Tested on Python 3.10+ (Ubuntu 22.04 and Windows 11).

---

## Dataset

All notebooks use the **UCI Air Quality dataset** accessed via cloud API — no CSV download needed:

```python
from ucimlrepo import fetch_ucirepo
import pandas as pd

air_quality = fetch_ucirepo(id=360)
X = air_quality.data.features
y = air_quality.data.targets
df = pd.concat([X, y], axis=1)
```

Dataset: De Vito et al. (2008). Sensor array for air quality monitoring. UCI ML Repository, id=360.

---

## Contact

For enquiries: business@patriaco.co.uk  
Open-source practice: https://github.com/Harrypatria  
Organisation: https://www.convergeni.com
