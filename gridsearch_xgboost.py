from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import joblib


df = pd.read_csv("balanced_bangalore_traffic_dataset.csv")

df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Weekday"] = df["Date"].dt.weekday
df["Is_Weekend"] = df["Weekday"].apply(lambda x: 1 if x >= 5 else 0)

df = df.dropna()

categorical_cols = ["Area Name", "Road/Intersection Name", "Weather Conditions"]
numeric_cols = [
    "Traffic Volume", "Environmental Impact",
    "Year", "Month", "Day", "Weekday", "Is_Weekend"
]

X = df[categorical_cols + numeric_cols]
y = df["Congestion Level"]


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)


rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(objective="reg:squarederror", random_state=42)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", xgb)  
])


param_grid = {
    "model__n_estimators": [150, 250, 350],
    "model__max_depth": [4, 6, 8],
    "model__learning_rate": [0.05, 0.1]
}


grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    verbose=2,
    n_jobs=-1
)

grid.fit(X, y)

print("\nBest Params:", grid.best_params_)
print("Best Score:", grid.best_score_)


joblib.dump(grid.best_estimator_, "traffic_model_best.pkl")
print("\nSaved tuned model as traffic_model_best.pkl")
