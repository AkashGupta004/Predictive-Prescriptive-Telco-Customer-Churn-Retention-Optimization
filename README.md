# Predictive-Prescriptive Telco Customer Churn & Retention Optimization

An end-to-end Machine Learning and Prescriptive Analytics system that connects calibrated customer churn prediction, deep autoencoder customer segmentation, SHAP explainability, and Mixed Integer Linear Programming (MILP) budget allocation for telecom customer retention.

---

## 1. System Architecture

```
                          IBM Telco Churn Data (7,043 Customers)
                                            │
                                            ▼
                      Leakage-Safe Preprocessing Pipeline
                      - Imputation & One-Hot Encoding (46 features)
                      - SMOTE Class Balancing (Train set only)
                                            │
                   ┌────────────────────────┴────────────────────────┐
                   ▼                                                 ▼
     Predictive Churn Pipeline                        Unsupervised Segmentation
     - 7 Classifiers + Soft Voting                    - 46-to-16 Deep Autoencoder
     - Model Selection via PR-AUC                     - Latent K-Means Clustering (k=3)
     - Isotonic Calibration (Brier Score)                            │
     - Cost-Sensitive Thresholding (5:1 FN:FP)                       │
                   │                                                 │
                   └────────────────────────┬────────────────────────┘
                                            ▼
                         Customer Value Proxy & Offer Matrix
                         - Customer Value Proxy = MonthlyCharges × 0.40 × 12 × Horizon
                         - Retention Offers: Basic (₹200), Standard (₹500), Premium (₹1,000)
                         - Expected Net Benefit = (p_i × Effectiveness_k × Value_i) - Cost_k
                                            │
                                            ▼
                         Prescriptive MILP Optimization (PuLP / CBC)
                         - Maximize Total Expected Net Benefit (ROI)
                         - Subject to: Total Budget, Mutually Exclusive Offers, Capacities
                         - Baselines: No Campaign, Random, High Risk, High Value, Greedy
                                            │
                                            ▼
                                  Outputs & Analytics
                         - outputs/retention_policy.csv
                         - outputs/optimization_summary.json
                         - outputs/sensitivity_analysis.csv
                         - outputs/sensitivity_summary.json
```

---

## 2. Mathematical Formulation

### 2.1 Predictive Layer
* **SMOTE Balancing:** Applied exclusively to the training partition (60%) within `imblearn.pipeline.Pipeline` to prevent target contamination.
* **Isotonic Probability Calibration:** Calibrates raw classifier output scores to true empirical probabilities using non-parametric isotonic regression via 3-fold cross-validation.
* **Cost-Sensitive Threshold Tuning:**
  $$\theta^* = \arg\min_{\theta \in [0, 1]} \left( 5.0 \cdot \text{FN}(\theta) + 1.0 \cdot \text{FP}(\theta) \right)$$
  Yields optimal operating threshold $\theta^* \approx 0.2341$ (recall: 84.22%).

### 2.2 Prescriptive MILP Formulation
For customer $i \in \{1, \dots, N\}$ and retention offer $k \in \{1, \dots, K\}$:
* **Binary Decision Variables:** $x_{i,k} \in \{0, 1\}$ ($1$ if offer $k$ is allocated to customer $i$, $0$ otherwise).
* **Objective Function:**
  $$\max \sum_{i=1}^N \sum_{k=1}^K \text{Net Benefit}_{i,k} \cdot x_{i,k}$$
  where:
  $$\text{Net Benefit}_{i,k} = \left( p_i \cdot \Delta_{k} \cdot \text{Value Proxy}_i \right) - \text{Cost}_k$$
* **Constraints:**
  1. **Mutually Exclusive Offers (Single Treatment):**
     $$\sum_{k=1}^K x_{i,k} \le 1 \quad \forall i \in \{1, \dots, N\}$$
  2. **Total Campaign Budget:**
     $$\sum_{i=1}^N \sum_{k=1}^K \text{Cost}_k \cdot x_{i,k} \le \text{Budget}$$
  3. **Offer-Specific Capacity Constraints:**
     $$\sum_{i=1}^N x_{i,k} \le \text{Capacity}_k \quad \forall k$$
  4. **Strict Capital Protection (Non-Negative Net Benefit):**
     $$x_{i,k} = 0 \quad \text{if } \text{Net Benefit}_{i,k} \le 0 \quad (\text{when } \texttt{allow\_negative\_net\_benefit} = \text{False})$$

---

## 3. Windows Installation & Environment Setup

### Prerequisites
* Windows 10/11 (64-bit)
* Python 3.10 – 3.14

### Installation Steps

1. **Open PowerShell** and navigate to the project root:
   ```powershell
   cd ML-project
   ```

2. **Create and Activate a Virtual Environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(If script execution is disabled, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

3. **Install Requirements:**
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

---

## 4. Execution Guide

### 4.1 Running Automated Unit Tests
To verify all 10 unit tests across preprocessing, cost-sensitive thresholding, MILP optimization, baselines, and sensitivity analysis:
```powershell
pytest
```
*(Or `python -m pytest -v`)*

### 4.2 Standard Predictive Run (Existing ML Pipeline)
```powershell
python main.py
```
* Fits/loads classifiers, calibrates probabilities, calculates test metrics, and runs autoencoder segmentation.

### 4.3 Running Prescriptive Retention Optimization (MILP)
Run the optimization pipeline using the pre-trained calibrated model and segmenter:
```powershell
python main.py --optimize --budget 50000.0 --horizon-years 5.0
```

### 4.4 Running Full Budget Sensitivity Grid
Evaluate all 6 baseline strategies across ₹10k, ₹25k, ₹50k, ₹75k, and ₹100k:
```powershell
python main.py --optimize --run-sensitivity --horizon-years 5.0
```

### 4.5 Interactive Streamlit Web Dashboard
Launch the interactive 3-tab viva presentation and demonstration dashboard:
```powershell
streamlit run app.py
```
* **Tab 1:** Executive KPI cards, Plotly budget sensitivity curve highlighting the ₹11,400 capital protection plateau, and baseline strategy bar charts.
* **Tab 2:** SHAP global feature attributions and calibrated test metrics (PR-AUC: 0.6085, Cost-Sensitive Threshold: 0.2341, Recall: 84.22%).
* **Tab 3:** Autoencoder + K-Means ($k=3$) customer cluster donut chart and an interactive, filterable data table of allocated retention policies.

### 4.6 Optional Execution Flags
* `--tune --trials 20`: Runs Optuna Bayesian hyperparameter search on all 7 classifiers.
* `--explain`: Generates SHAP feature importance plot (`outputs/shap_summary.png`).
* `--skip-segmentation`: Skips deep autoencoder clustering.
* `--allow-negative-net-benefit`: Allows negative ROI assignments for budget exhaustion simulations.

---

## 5. Project Directory & Output Artifacts

```
ML-project/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # IBM Telco dataset (7,043 records)
├── outputs/
│   ├── churn_model.joblib                    # Serialized calibrated classifier & threshold
│   ├── segmenter.joblib                      # Serialized autoencoder + KMeans pipeline
│   ├── model_comparison.csv                  # PR-AUC validation comparison
│   ├── calibration_comparison.csv            # Brier score validation comparison
│   ├── test_metrics.csv                      # Held-out test set classification metrics
│   ├── test_predictions.csv                  # Predictions and probabilities on test set
│   ├── customer_segments.csv                 # 16-dim latent vectors and cluster IDs
│   ├── retention_policy.csv                  # Prescriptive customer-level offer allocations
│   ├── optimization_summary.json             # Aggregate strategy comparison & metrics
│   ├── sensitivity_analysis.csv              # Full budget grid sensitivity table (6 strategies)
│   └── sensitivity_summary.json             # Marginal ROI & diminishing returns trend
├── src/
│   ├── preprocessing.py                      # Data cleaning & ColumnTransformer pipeline
│   ├── models.py                             # 7 classifiers, SMOTE, and Optuna tuning
│   ├── ensemble.py                           # Equal-weight soft-voting ensemble
│   ├── calibration.py                        # Isotonic and Sigmoid CalibratedClassifierCV
│   ├── evaluation.py                         # Cost-sensitive threshold tuning & metrics
│   ├── explainability.py                     # TreeExplainer & KernelExplainer SHAP plots
│   ├── segmentation.py                       # 46->16 deep autoencoder + KMeans clustering
│   └── retention_optimization.py             # MILP engine, baselines, and sensitivity analysis
├── tests/
│   ├── test_preprocessing.py                 # Feature shape & column integrity tests
│   ├── test_evaluation.py                    # Threshold optimization & metric tests
│   └── test_retention_optimization.py        # Hand-calculated 3-customer 2-offer MILP & baselines
├── main.py                                   # Central CLI execution controller
├── pytest.ini                                # Pytest path configuration
└── requirements.txt                          # Project dependency specifications
```
