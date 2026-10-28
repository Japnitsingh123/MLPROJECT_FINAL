# 🚦 Bangalore Traffic Congestion Predictor

An end-to-end Machine Learning web application that predicts road traffic congestion in Bangalore based on temporal data, location, weather conditions, and environmental impact indicators. Built with **FastAPI**, **Scikit-Learn**, **XGBoost**, and an intelligent contextual analysis engine (supporting **Google Gemini**).

---

## 🚀 Features

- **Machine Learning Stacking Ensemble**: Base regressor (Random Forest with 300 estimators) stacked into a meta-regressor (XGBoost with 300 estimators) achieving **$R^2 > 0.995$** and **$98.3\%$ accuracy**.
- **Accurate Environmental Feature Calibration**: Maps qualitative environmental levels (*Very Clean*, *Clean*, *Slightly Polluted*, *Polluted*, *Highly Polluted*) to realistic Bangalore dataset distributions, preventing negative traffic volumes and enabling the full dynamic prediction range.
- **Model Evaluation & FP/FN Diagnosis**: Provides actual vs. predicted congestion score comparisons, quantifying absolute error and classifying predictions into:
  - 🎯 **Perfect Prediction** (within $\pm 1.5$ points)
  - ⚠️ **FP-like (Over-Prediction)** (Predicted > Actual)
  - ⚡ **FN-like (Under-Prediction)** (Predicted < Actual)
- **Hybrid AI Analysis**: Automatically uses Google Gemini 1.5 Flash if an API key is provided, or seamlessly falls back to a built-in rule-based traffic reasoning engine when offline or unconfigured.
- **Instant Server Startup**: Precomputed benchmark metrics cache eliminates startup latency.
- **Modern Responsive Web UI**: Tailwind CSS interface with auto-filled date pickers, dynamic road filtering per area, and interactive diagnosis cards.

---

## 📁 Project Structure

```
MLPROJECT_FINAL/
├── balanced_bangalore_traffic_dataset.csv     # 19,845 balanced Bangalore traffic records
├── traffic_model_balanced_stacking_rf_xgb.pkl  # Trained Stacking Regressor model (RF -> XGBoost)
├── metrics.json                               # Precomputed benchmark validation metrics
├── main.py                                    # FastAPI application backend and prediction API
├── traffic_predictor.html                     # Modern frontend user interface
├── test_system.py                             # Automated test suite (8 tests)
├── stack.py                                   # Training script for the Stacking Regressor
├── train_model_balanced1.py                   # Alternative training pipeline
├── gridsearch_randomforest.py                 # Hyperparameter tuning for Random Forest
├── gridsearch_xgboost.py                      # Hyperparameter tuning for XGBoost
├── requirements.txt                           # Project dependencies
├── .env.example                               # Sample environment variables
└── README.md                                  # Documentation
```

---

## 🛠️ Quickstart Installation

### 1. Clone the repository
```bash
git clone https://github.com/Japnitsingh123/MLPROJECT_FINAL.git
cd MLPROJECT_FINAL
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure Gemini AI
Copy `.env.example` to `.env` and set your Google Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Note: If no API key is set, the application will use its built-in traffic intelligence engine seamlessly).*

---

## 🧪 Run Automated Tests

Execute the test suite to verify data integrity, endpoints, and prediction ranges:
```bash
python test_system.py
```

Expected output:
```
Ran 8 tests in 0.759s
OK
[PASS] /init_data verified with 8 Bangalore areas.
[PASS] /performance_data verified (RMSE=2.0313, R2=0.9953, Accuracy=98.32%).
[PASS] Environmental categories correctly calibrated to positive, increasing traffic volumes.
[PASS] /predict baseline succeeded: score=57.27, category='Medium Congestion'
[PASS] /predict actual comparison succeeded: error=27.27, diagnosis='FP-like (Over-Prediction)'
[PASS] Verified full dynamic range: Both Low and High Congestion are reachable.
[PASS] Input validations for date and environmental category passed.
[PASS] Frontend HTML endpoint loaded successfully.
```

---

## 🌐 Launch the Web Application

Run the server directly:
```bash
python main.py
```
Or using Uvicorn:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 📊 Benchmark Metrics

| Metric | Holdout Test Score | Description |
| :--- | :---: | :--- |
| **RMSE** | `2.0313` | Root Mean Squared Error on holdout 20% test split |
| **$R^2$ Score** | `0.9953` | Explains 99.53% of variance in traffic congestion |
| **MAPE** | `1.68%` | Mean Absolute Percentage Error |
| **Accuracy** | `98.32%` | $100\% - \text{MAPE}$ |
