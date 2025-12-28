# ============================================
# train_model_balanced_stacking_rf_xgb.py
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# 1️⃣ Load balanced dataset
df = pd.read_csv("balanced_bangalore_traffic_dataset.csv")

# 2️⃣ Target + features
target = "Congestion Level"

features = [
    "Date",
    "Area Name",
    "Road/Intersection Name",
    "Traffic Volume",
    "Environmental Impact",
    "Weather Conditions"        # ✅ NEW
]

df = df[features + [target]].dropna()

# 3️⃣ Date → components
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Weekday"] = df["Date"].dt.weekday
df["Is_Weekend"] = df["Weekday"].apply(lambda x: 1 if x >= 5 else 0)
df = df.drop(columns=["Date"])

# 4️⃣ Categorical + numeric features
categorical_cols = ["Area Name", "Road/Intersection Name", "Weather Conditions"]  # ✅ added weather
numeric_cols = [
    "Traffic Volume",
    "Environmental Impact",
    "Year",
    "Month",
    "Day",
    "Weekday",
    "Is_Weekend"
]

# 5️⃣ Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

# 6️⃣ STACKING MODEL (Base: RF, Meta: XGB)
base_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
    min_samples_split=4,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)

meta_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

stack_model = StackingRegressor(
    estimators=[
        ("rf", base_model)
    ],
    final_estimator=meta_model,
    passthrough=True
)

# Pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", stack_model)
])

# 7️⃣ Train-test split
X = df[categorical_cols + numeric_cols]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8️⃣ Train model
model.fit(X_train, y_train)

# -----------------------------
# 🔥 9️⃣ TRAINING METRICS
# -----------------------------
train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
train_r2 = r2_score(y_train, train_pred)

# -----------------------------
# 🔥 🔥 TESTING METRICS
# -----------------------------
test_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)

print("\n===============================================")
print(" 🚀 MODEL PERFORMANCE REPORT (Stacking: RF → XGB)")
print("===============================================")
print("📘 TRAINING METRICS")
print(f"   🔹 Training RMSE: {train_rmse:.4f}")
print(f"   🔹 Training R²:   {train_r2:.4f}")

print("\n📗 TESTING METRICS")
print(f"   🔹 Testing RMSE: {test_rmse:.4f}")
print(f"   🔹 Testing R²:   {test_r2:.4f}")

# 🔟 Cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print("\n📊 CROSS-VALIDATION")
print(f"   🔹 5-Fold R² Mean: {scores.mean():.4f}")

# 💾 Save model
joblib.dump(model, "traffic_model_balanced_stacking_rf_xgb.pkl")
print("\n💾 Model saved as traffic_model_balanced_stacking_rf_xgb.pkl")
print("===============================================\n")
