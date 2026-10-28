"""
Automated System Test for Bangalore Traffic Predictor
Verifies dataset loading, stacking model loading, FastAPI endpoints,
environmental categories calibration, and FP/FN actual comparison.
"""

import sys
import unittest
from fastapi.testclient import TestClient
from main import app, convert_env_category, categorize_congestion, a_tv, b_tv

client = TestClient(app)


class TestTrafficPredictor(unittest.TestCase):

    def test_01_init_data(self):
        """Test /init_data returns Bangalore areas and roads."""
        response = client.get("/init_data")
        self.assertEqual(response.status_code, 200, f"Expected 200, got {response.status_code}")
        data = response.json()
        self.assertIn("areas", data)
        self.assertIn("area_to_roads", data)
        self.assertGreater(len(data["areas"]), 5)
        self.assertIn("Indiranagar", data["areas"])
        self.assertIn("100 Feet Road", data["area_to_roads"]["Indiranagar"])
        print(f"[PASS] /init_data verified with {len(data['areas'])} Bangalore areas.")

    def test_02_performance_data(self):
        """Test /performance_data returns valid metrics."""
        response = client.get("/performance_data")
        self.assertEqual(response.status_code, 200)
        perf = response.json()
        self.assertIn("rmse", perf)
        self.assertIn("r2", perf)
        self.assertIn("accuracy", perf)
        self.assertLess(perf["rmse"], 3.0)
        self.assertGreater(perf["r2"], 0.95)
        self.assertGreater(perf["accuracy"], 90.0)
        print(f"[PASS] /performance_data verified (RMSE={perf['rmse']}, R2={perf['r2']}, Accuracy={perf['accuracy']}%).")

    def test_03_environmental_mapping_calibration(self):
        """Verify environmental category mapping does not generate negative traffic volume."""
        categories = ["Very Clean", "Clean", "Slightly Polluted", "Polluted", "Highly Polluted"]
        prev_tv = 0
        for cat in categories:
            env_val = convert_env_category(cat)
            tv = (a_tv * env_val) + b_tv
            self.assertGreater(tv, 0, f"Traffic volume for {cat} must be positive, got {tv}")
            self.assertGreater(tv, prev_tv, f"Traffic volume must increase with pollution: {cat}")
            prev_tv = tv
        print("[PASS] Environmental categories correctly calibrated to positive, increasing traffic volumes.")

    def test_04_prediction_baseline(self):
        """Test /predict returns valid prediction and categories."""
        payload = {
            "date": "2026-10-28",
            "area": "Indiranagar",
            "road": "100 Feet Road",
            "weather": "Clear",
            "env_category": "Slightly Polluted"
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        res = response.json()
        self.assertIn("prediction", res)
        self.assertIn("category", res)
        self.assertIn("gemini_analysis", res)
        self.assertIn("analysis_provider", res)
        self.assertGreaterEqual(res["prediction"], 0.0)
        self.assertLessEqual(res["prediction"], 100.0)
        self.assertGreater(len(res["gemini_analysis"]), 20)
        print(f"[PASS] /predict baseline succeeded: score={res['prediction']}, category='{res['category']}'")

    def test_05_prediction_actual_evaluation(self):
        """Test /predict with actual value produces FP/FN evaluation."""
        payload = {
            "date": "2026-10-28",
            "area": "Indiranagar",
            "road": "100 Feet Road",
            "weather": "Clear",
            "env_category": "Slightly Polluted",
            "actual": 30.0  # Deliberately lower to test FP-like over-prediction
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertIsNotNone(res["error_value"])
        self.assertIsNotNone(res["fp_fn_result"])
        self.assertEqual(res["actual_value"], 30.0)
        self.assertIn("Over-Prediction", res["fp_fn_result"])
        print(f"[PASS] /predict actual comparison succeeded: error={res['error_value']}, diagnosis='{res['fp_fn_result']}'")

    def test_06_environmental_sweep_predictions(self):
        """Test full range of predictions across environmental categories."""
        scores = []
        for cat in ["Very Clean", "Clean", "Slightly Polluted", "Polluted", "Highly Polluted"]:
            payload = {
                "date": "2026-10-28",
                "area": "Koramangala",
                "road": "Intermediate Ring Road",
                "weather": "Clear",
                "env_category": cat
            }
            res = client.post("/predict", json=payload).json()
            scores.append((cat, res["prediction"], res["category"]))

        print("\n--- Environmental Category Prediction Sweep ---")
        for cat, score, category in scores:
            print(f"  [{cat:17s}] -> Score: {score:5.2f} | Category: {category}")

        # Ensure Low, Medium, and High are all reachable
        categories_reached = {c[2] for c in scores}
        self.assertIn("Low Congestion", categories_reached)
        self.assertIn("High Congestion", categories_reached)
        print("[PASS] Verified full dynamic range: Both Low and High Congestion are reachable.")

    def test_07_invalid_input_handling(self):
        """Test error handling on invalid date or invalid category."""
        bad_date = client.post("/predict", json={
            "date": "invalid-date",
            "area": "Indiranagar",
            "road": "100 Feet Road",
            "weather": "Clear",
            "env_category": "Clean"
        })
        self.assertEqual(bad_date.status_code, 400)

        bad_cat = client.post("/predict", json={
            "date": "2026-10-28",
            "area": "Indiranagar",
            "road": "100 Feet Road",
            "weather": "Clear",
            "env_category": "NonExistentCategory"
        })
        self.assertEqual(bad_cat.status_code, 400)
        print("[PASS] Input validations for date and environmental category passed.")

    def test_08_frontend_html(self):
        """Test frontend HTML endpoint."""
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Bangalore Traffic Predictor", response.text)
        self.assertIn("predictionValue", response.text)
        print("[PASS] Frontend HTML endpoint loaded successfully.")


if __name__ == "__main__":
    unittest.main()
