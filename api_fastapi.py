import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import math
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class PredictionInput(BaseModel):
    date: str
    area: str
    road: str
    weather: str
    env_category: str
    actual: float | None = None


def categorize_congestion(level: float) -> str:
    if level <= 33:
        return "Low Congestion"
    elif level <= 66:
        return "Medium Congestion"
    else:
        return "High Congestion"


def convert_env_category(cat: str) -> float:
    cat = cat.lower().strip()

    if cat == "very clean":
        return 10        
    elif cat == "clean":
        return 30        
    elif cat == "slightly polluted":
        return 45        
    elif cat == "polluted":
        return 60        
    elif cat == "highly polluted":
        return 85        
    else:
        raise ValueError("Invalid environmental category")


app = FastAPI(
    title="Bangalore Traffic Predictor API",
    description="Predicts traffic congestion based on date, area, road, weather, and environmental impact.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


api_key = "AIzaSyA1aRri2-gE11S6DHHAdprZ4Zodyj8s7gE"

try:
    genai.configure(api_key=api_key)
    model_gemini = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
except:
    model_gemini = None
    print("Gemini not configured")

performance_data_cache = {}

try:
    model = joblib.load("traffic_model_balanced_stacking_rf_xgb.pkl")
    data = pd.read_csv("balanced_bangalore_traffic_dataset.csv")

    data["Area Name"] = data["Area Name"].astype(str).str.strip()
    data["Road/Intersection Name"] = data["Road/Intersection Name"].astype(str).str.strip()

    mapping = data.groupby(["Area Name", "Road/Intersection Name"]).size().reset_index(name="count")
    areas = mapping["Area Name"].sort_values().unique().tolist()

    area_to_roads = {
        a: mapping[mapping["Area Name"] == a]
        .sort_values("count", ascending=False)["Road/Intersection Name"]
        .tolist()
        for a in areas
    }

    

    print("Model, data, and mappings loaded successfully.")

    
    tmp = data.dropna(subset=[
        "Date", "Area Name", "Road/Intersection Name",
        "Traffic Volume", "Environmental Impact", "Congestion Level", "Weather Conditions"
    ])

    tmp["Date"] = pd.to_datetime(tmp["Date"])
    tmp["Year"] = tmp["Date"].dt.year
    tmp["Month"] = tmp["Date"].dt.month
    tmp["Day"] = tmp["Date"].dt.day
    tmp["Weekday"] = tmp["Date"].dt.weekday
    tmp["Is_Weekend"] = (tmp["Weekday"] >= 5).astype(int)
    tmp = tmp.drop(columns=["Date"])

    
    X_full = tmp[[ 
    "Area Name", "Road/Intersection Name", "Weather Conditions",
    "Traffic Volume", "Environmental Impact",
    "Year", "Month", "Day", "Weekday", "Is_Weekend"
]]
    y_full = tmp["Congestion Level"]

    
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

   
    preds = model.predict(X_test_b)

    rmse = math.sqrt(mean_squared_error(y_test_b, preds))
    r2   = r2_score(y_test_b, preds)
    mape = np.mean(np.abs((y_test_b - preds) / y_test_b)) * 100
    accuracy = 100 - mape


   
    from sklearn.linear_model import LinearRegression

    tv_reg = LinearRegression()

    valid_tv_rows = data.dropna(subset=["Environmental Impact", "Traffic Volume"])

    X_tv = valid_tv_rows["Environmental Impact"].values.reshape(-1,1)
    y_tv = valid_tv_rows["Traffic Volume"].values.reshape(-1,1)

    tv_reg.fit(X_tv, y_tv)

    a_tv = float(tv_reg.coef_[0][0])     
    b_tv = float(tv_reg.intercept_[0])    

    


    performance_data_cache = {
        "rmse": rmse,
        "r2": r2,
        "accuracy": accuracy
    }

    print(f"Performance metrics: RMSE={rmse:.3f}, R²={r2:.3f}")

except Exception as e:
    print("ERROR:", e)
    model = None
    data = None
    areas = ["Error"]
    area_to_roads = {"Error": ["Could not load dataset"]}


@app.get('/init_data')
def get_init_data():
    if data is None:
        raise HTTPException(status_code=500, detail="Server data not loaded")
    return {
        "areas": areas,
        "area_to_roads": area_to_roads
    }


@app.get("/performance_data")
def get_performance():
    if not performance_data_cache:
        raise HTTPException(status_code=500, detail="Performance metrics unavailable")
    return performance_data_cache


@app.post('/predict')
def predict(inputs: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Server model not loaded")

    try:
        
        date = datetime.strptime(inputs.date, '%Y-%m-%d')
        row_df = pd.DataFrame([{
            "Year": date.year,
            "Month": date.month,
            "Day": date.day,
            "Weekday": date.weekday(),
            "Is_Weekend": 1 if date.weekday() >= 5 else 0
        }])

        area = inputs.area
        road = inputs.road
        weather = inputs.weather

       

        
        env_impact = convert_env_category(inputs.env_category)

      
        traffic_volume = a_tv * env_impact + b_tv


        final_df = pd.DataFrame([{
            "Area Name": area,
            "Road/Intersection Name": road,
            "Weather Conditions": weather,
            "Traffic Volume": float(traffic_volume),
            "Environmental Impact": float(env_impact),
            "Year": row_df["Year"][0],
            "Month": row_df["Month"][0],
            "Day": row_df["Day"][0],
            "Weekday": row_df["Weekday"][0],
            "Is_Weekend": row_df["Is_Weekend"][0]
        }])

        pred = model.predict(final_df)[0]
        category = categorize_congestion(pred)

        actual_value = inputs.actual
        fp_fn_label = None
        error_value = None

        if actual_value is not None:
            error_value = pred - actual_value
            if error_value > 0:
                fp_fn_label = "FP-like (Over-Prediction)"
            elif error_value < 0:
                fp_fn_label = "FN-like (Under-Prediction)"
            else:
                fp_fn_label = "Perfect Prediction"

        gemini_text = ""
        if model_gemini:
            prompt = f"""
Provide a clear traffic insight based on the following details:

Predicted Congestion Level: {pred:.2f}
Category: {category}
Actual Value (if given): {actual_value}
Prediction Error: {error_value}
Error Type: {fp_fn_label}
Show FP,FN explicitly in beggining.
Explain briefly why this level of congestion likely occurred. 
if actual value is given,then give the FP,FN and  make anaylsis on it.
Write in smooth, natural language with short paragraphs only. 
Do not use hashtags, bullet points, or lists.
""" 

            try:
                ai_response = model_gemini.generate_content(prompt)
                gemini_text = ai_response.text
            except:
                gemini_text = "Gemini analysis unavailable."

        return {
            "prediction": float(pred),
            "category": category,
            "fp_fn_result": fp_fn_label,
            "error_value": float(error_value) if error_value is not None else None,
            "internal_values": {
                "traffic_volume": float(traffic_volume),
                "env_impact": float(env_impact)
            },
            "gemini_analysis": gemini_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
