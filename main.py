import os
import json
import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Base directory for resolving file paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "balanced_bangalore_traffic_dataset.csv"
MODEL_PATH = BASE_DIR / "traffic_model_balanced_stacking_rf_xgb.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"
HTML_PATH = BASE_DIR / "traffic_predictor.html"
ENV_PATH = BASE_DIR / ".env"

# Load environment variables
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()


class PredictionInput(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    area: str = Field(..., description="Area name in Bangalore")
    road: str = Field(..., description="Road or intersection name")
    weather: str = Field(..., description="Weather condition")
    env_category: str = Field(..., description="Environmental impact category")
    actual: Optional[float] = Field(None, description="Actual congestion level (0-100)")


def categorize_congestion(level: float) -> str:
    """Categorize congestion score into Low, Medium, or High."""
    if level <= 33.33:
        return "Low Congestion"
    elif level <= 66.66:
        return "Medium Congestion"
    else:
        return "High Congestion"


def convert_env_category(cat: str) -> float:
    """
    Map qualitative environmental categories to realistic dataset distributions.
    Dataset Environmental Impact statistics:
      - Min: 58.5, 10th%: 64.5, 25th%: 69.2, Median: 80.7, 75th%: 101.7, 90th%: 128.9, Max: 194.1
    These mapped values yield positive, realistic traffic volume estimations (approx 7,500 to 40,000 vehicles).
    """
    cat = cat.lower().strip()
    mapping = {
        "very clean": 65.0,
        "clean": 75.0,
        "slightly polluted": 85.0,
        "polluted": 105.0,
        "highly polluted": 130.0,
    }
    if cat not in mapping:
        raise ValueError(f"Invalid environmental category '{cat}'. Valid choices: {list(mapping.keys())}")
    return mapping[cat]


def generate_local_traffic_insight(
    area: str,
    road: str,
    weather: str,
    pred: float,
    category: str,
    traffic_volume: float,
    env_impact: float,
    is_weekend: bool,
    actual_value: Optional[float],
    error_value: Optional[float],
    fp_fn_label: Optional[str]
) -> str:
    """Generate intelligent natural-language traffic analysis when Gemini is unavailable."""
    paragraphs = []

    # 1. Error / FP-FN Analysis if actual is provided
    if actual_value is not None and fp_fn_label is not None and error_value is not None:
        if "Over-Prediction" in fp_fn_label:
            fp_fn_desc = (
                f"{fp_fn_label}: The model predicted a congestion score of {pred:.2f}, "
                f"which is {abs(error_value):.2f} points higher than the recorded actual congestion of {actual_value:.2f}. "
                "This false-positive inclination may stem from higher estimated traffic volume along this arterial corridor."
            )
        elif "Under-Prediction" in fp_fn_label:
            fp_fn_desc = (
                f"{fp_fn_label}: The model predicted a congestion score of {pred:.2f}, "
                f"which is {abs(error_value):.2f} points lower than the recorded actual congestion of {actual_value:.2f}. "
                "This under-prediction likely reflects unforeseen localized bottlenecks, roadwork, or rapid transit congestion."
            )
        else:
            fp_fn_desc = (
                f"{fp_fn_label}: The predicted congestion of {pred:.2f} closely aligns with the actual value of {actual_value:.2f}."
            )
        paragraphs.append(fp_fn_desc)

    # 2. Congestion context & environmental factors
    day_type = "weekend" if is_weekend else "weekday"
    weather_note = ""
    w_lower = weather.lower()
    if "rain" in w_lower:
        weather_note = "Rainfall and wet road surfaces typically slow vehicular speeds and exacerbate bottlenecks."
    elif "fog" in w_lower:
        weather_note = "Reduced visibility due to fog is contributing to cautious commuter driving speeds."
    elif "wind" in w_lower or "overcast" in w_lower:
        weather_note = f"Prevailing {weather.lower()} weather maintains steady traffic flow with moderate atmospheric effects."
    else:
        weather_note = "Clear atmospheric conditions support unhindered visibility across key junctions."

    context_desc = (
        f"{category} was predicted for {road} in {area} on a {day_type}. "
        f"With an estimated traffic volume of approximately {traffic_volume:,.0f} vehicles and an environmental impact "
        f"index of {env_impact:.1f}, vehicular density is operating within typical seasonal thresholds. {weather_note}"
    )
    paragraphs.append(context_desc)

    # 3. Actionable summary
    if pred > 66.66:
        action_note = "Commuters are advised to utilize bypass connectors or schedule travel outside peak commuter hours to avoid heavy gridlock."
    elif pred > 33.33:
        action_note = "Traffic movement remains steady with manageable delay times across major intersections."
    else:
        action_note = "Optimal travel conditions are expected with free-flowing traffic across the sector."
    paragraphs.append(action_note)

    return "\n\n".join(paragraphs)


# Initialize FastAPI app
app = FastAPI(
    title="Bangalore Traffic Predictor API",
    description="Predicts traffic congestion based on date, area, road, weather, and environmental impact.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini AI configuration
gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
model_gemini = None
gemini_model_name = "gemini-1.5-flash"

if gemini_api_key:
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)
        model_gemini = genai.GenerativeModel(gemini_model_name)
        print(f"Gemini configured successfully with model {gemini_model_name}.")
    except Exception as e:
        print(f"Gemini configuration notice: {e}. Falling back to internal intelligence engine.")
        model_gemini = None
else:
    print("No GEMINI_API_KEY configured in environment or .env. Using built-in traffic intelligence engine.")

# Global state variables
model = None
data = None
areas = []
area_to_roads = {}
performance_data_cache = {}
a_tv = 500.0
b_tv = -25000.0


def compute_or_load_performance(model_obj, df):
    """Load cached benchmark performance metrics or calculate them on the test split."""
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
                if "rmse" in metrics and "r2" in metrics and "accuracy" in metrics:
                    return metrics
        except Exception as e:
            print(f"Notice reading cached metrics: {e}")

    # Fallback to computing metrics on 20% test split
    print("Evaluating model performance on test set (this runs once)...")
    tmp = df.dropna(subset=[
        "Date", "Area Name", "Road/Intersection Name",
        "Traffic Volume", "Environmental Impact", "Congestion Level", "Weather Conditions"
    ]).copy()

    tmp["Date"] = pd.to_datetime(tmp["Date"])
    tmp["Year"] = tmp["Date"].dt.year
    tmp["Month"] = tmp["Date"].dt.month
    tmp["Day"] = tmp["Date"].dt.day
    tmp["Weekday"] = tmp["Date"].dt.weekday
    tmp["Is_Weekend"] = (tmp["Weekday"] >= 5).astype(int)

    X_full = tmp[[
        "Area Name", "Road/Intersection Name", "Weather Conditions",
        "Traffic Volume", "Environmental Impact",
        "Year", "Month", "Day", "Weekday", "Is_Weekend"
    ]]
    y_full = tmp["Congestion Level"]

    _, X_test_b, _, y_test_b = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

    preds = model_obj.predict(X_test_b)
    rmse = float(math.sqrt(mean_squared_error(y_test_b, preds)))
    r2 = float(r2_score(y_test_b, preds))
    mape = float(np.mean(np.abs((y_test_b - preds) / y_test_b)) * 100)
    accuracy = float(100.0 - mape)

    metrics = {
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "accuracy": round(accuracy, 2),
        "mape": round(mape, 2)
    }

    try:
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass

    return metrics


# Load Model, Dataset, and Linear Regression Parameters on Startup
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")
    data = pd.read_csv(DATASET_PATH)

    data["Area Name"] = data["Area Name"].astype(str).str.strip()
    data["Road/Intersection Name"] = data["Road/Intersection Name"].astype(str).str.strip()

    mapping = data.groupby(["Area Name", "Road/Intersection Name"]).size().reset_index(name="count")
    areas = sorted(mapping["Area Name"].unique().tolist())

    area_to_roads = {
        a: mapping[mapping["Area Name"] == a]
        .sort_values("count", ascending=False)["Road/Intersection Name"]
        .tolist()
        for a in areas
    }

    # Fit Environmental Impact -> Traffic Volume linear estimator
    valid_tv_rows = data.dropna(subset=["Environmental Impact", "Traffic Volume"])
    if len(valid_tv_rows) > 0:
        X_tv = valid_tv_rows["Environmental Impact"].values.reshape(-1, 1)
        y_tv = valid_tv_rows["Traffic Volume"].values.reshape(-1, 1)
        tv_reg = LinearRegression()
        tv_reg.fit(X_tv, y_tv)
        a_tv = float(tv_reg.coef_[0][0])
        b_tv = float(tv_reg.intercept_[0])

    performance_data_cache = compute_or_load_performance(model, data)
    print("Model, dataset, and performance metrics initialized successfully.")
    print(f"Metrics: RMSE={performance_data_cache.get('rmse')}, R2={performance_data_cache.get('r2')}, Accuracy={performance_data_cache.get('accuracy')}%")

except Exception as e:
    print(f"Initialization Error: {e}")
    model = None
    data = None
    areas = ["Error"]
    area_to_roads = {"Error": [f"Initialization failed: {e}"]}


@app.get("/init_data")
def get_init_data():
    """Return available areas and corresponding road options."""
    if data is None or not areas:
        raise HTTPException(status_code=500, detail="Server data not loaded")
    return {
        "areas": areas,
        "area_to_roads": area_to_roads
    }


@app.get("/performance_data")
def get_performance():
    """Return model performance metrics (RMSE, R2, Accuracy)."""
    if not performance_data_cache:
        raise HTTPException(status_code=500, detail="Performance metrics unavailable")
    return performance_data_cache


@app.post("/predict")
def predict(inputs: PredictionInput):
    """Predict traffic congestion level and return analysis."""
    if model is None:
        raise HTTPException(status_code=500, detail="Server model not loaded")

    # Date parsing and validation
    try:
        date_obj = datetime.strptime(inputs.date.strip(), "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date '{inputs.date}'. Format must be YYYY-MM-DD."
        )

    area = inputs.area.strip()
    road = inputs.road.strip()
    weather = inputs.weather.strip()

    if not area:
        raise HTTPException(status_code=400, detail="Area name cannot be empty.")
    if not road:
        raise HTTPException(status_code=400, detail="Road name cannot be empty.")

    try:
        env_impact = convert_env_category(inputs.env_category)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    # Calculate traffic volume and clamp to positive values
    raw_traffic_volume = (a_tv * env_impact) + b_tv
    traffic_volume = max(1000.0, float(raw_traffic_volume))

    is_weekend = 1 if date_obj.weekday() >= 5 else 0

    input_df = pd.DataFrame([{
        "Area Name": area,
        "Road/Intersection Name": road,
        "Weather Conditions": weather,
        "Traffic Volume": float(traffic_volume),
        "Environmental Impact": float(env_impact),
        "Year": date_obj.year,
        "Month": date_obj.month,
        "Day": date_obj.day,
        "Weekday": date_obj.weekday(),
        "Is_Weekend": is_weekend
    }])

    try:
        pred = float(model.predict(input_df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # Clamp prediction score within sensible range [0, 100]
    pred = max(0.0, min(100.0, pred))
    category = categorize_congestion(pred)

    actual_value = inputs.actual
    fp_fn_label = None
    error_value = None

    if actual_value is not None:
        error_value = float(pred - actual_value)
        if abs(error_value) < 1.5:
            fp_fn_label = "Perfect Prediction"
        elif error_value > 0:
            fp_fn_label = "FP-like (Over-Prediction)"
        else:
            fp_fn_label = "FN-like (Under-Prediction)"

    # AI Insight generation
    analysis_text = ""
    provider = "Built-in Traffic Intelligence Engine"

    if model_gemini:
        prompt = f"""
Provide a clear traffic insight based on the following details:

Area: {area}
Road: {road}
Weather: {weather}
Predicted Congestion Level: {pred:.2f}
Category: {category}
Actual Value (if given): {actual_value}
Prediction Error: {error_value}
Error Type: {fp_fn_label}
Show FP,FN explicitly in beginning if actual value is given.
Explain briefly why this level of congestion likely occurred.
Write in smooth, natural language with short paragraphs only.
Do not use hashtags, bullet points, or markdown headers.
"""
        try:
            ai_response = model_gemini.generate_content(prompt)
            if ai_response and ai_response.text:
                analysis_text = ai_response.text.strip()
                provider = "Google Gemini 1.5 Flash"
        except Exception as e:
            print(f"Gemini call failed ({e}); falling back to local traffic intelligence.")
            analysis_text = ""

    if not analysis_text:
        analysis_text = generate_local_traffic_insight(
            area=area,
            road=road,
            weather=weather,
            pred=pred,
            category=category,
            traffic_volume=traffic_volume,
            env_impact=env_impact,
            is_weekend=bool(is_weekend),
            actual_value=actual_value,
            error_value=error_value,
            fp_fn_label=fp_fn_label
        )

    return {
        "prediction": round(pred, 2),
        "category": category,
        "fp_fn_result": fp_fn_label,
        "error_value": round(error_value, 2) if error_value is not None else None,
        "actual_value": round(actual_value, 2) if actual_value is not None else None,
        "internal_values": {
            "traffic_volume": round(traffic_volume, 2),
            "env_impact": round(env_impact, 2)
        },
        "gemini_analysis": analysis_text,
        "analysis_provider": provider
    }


@app.get("/", response_class=HTMLResponse)
def frontend():
    """Serve the traffic prediction web user interface."""
    if not HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="Frontend HTML file not found")
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"Starting Bangalore Traffic Predictor on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
