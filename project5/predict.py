"""
predict.py
==========
Inference script for the Cirrhosis Mortality Risk Prediction model.

Model: XGBoost Classifier (tuned) with native NaN handling.
Target: Predict whether a patient will die (target=1) vs. survive/transplant-censored
        (target=0), based on baseline clinical features from the Mayo Clinic PBC
        (Primary Biliary Cirrhosis) study.

IMPORTANT — Design decisions replicated from training (see README.md for full rationale):
- N_Days, ID, Status are NEVER used as features (Status is the source of the target,
  N_Days is temporal leakage — it is only known AFTER the outcome occurs).
- Missing values in Ascites, Hepatomegaly, Spiders, Alk_Phos, SGOT are LEFT AS NaN
  and passed natively to XGBoost (do NOT impute them) — this preserves the fact that
  missingness itself is informative (it flags patients who were not part of the
  randomized trial arm of the original study).
- Edema is ORDINAL encoded (N=0, S=1, Y=2), not one-hot, because it represents a
  clinical severity gradient.
- Drug is ONE-HOT encoded with dummy_na=True so that "Drug missing" becomes an
  explicit feature (this captures the same signal as trial participation, without
  needing a separate redundant flag column).

Usage:
    from predict import CirrhosisRiskPredictor

    predictor = CirrhosisRiskPredictor(model_path="xgb_cirrhosis_model.pkl")
    result = predictor.predict(raw_patient_dict)
    print(result)
"""

import joblib
import pandas as pd
import numpy as np


class CirrhosisRiskPredictor:
    """
    Wraps the trained XGBoost model and replicates the exact preprocessing
    pipeline used during training, so that raw patient records can be scored
    directly without the caller needing to know internal encoding details.
    """

    # Columns expected in the raw input (mirrors the original UCI Cirrhosis dataset,
    # minus ID/Status/N_Days/Stage which are handled separately below).
    RAW_FEATURE_COLUMNS = [
        "Age", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema",
        "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos",
        "SGOT", "Tryglicerides", "Platelets", "Prothrombin", "Stage", "Drug",
    ]

    # Final column order the model was trained on. Must match exactly —
    # XGBoost native-NaN models are sensitive to column order/names via booster.
    MODEL_FEATURE_ORDER = [
        "Age", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema",
        "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos",
        "SGOT", "Tryglicerides", "Platelets", "Prothrombin", "Stage",
        "Drug_D-penicillamine", "Drug_Placebo", "Drug_nan",
    ]

    def __init__(self, model_path: str = "xgb_cirrhosis_model.pkl"):
        self.model = joblib.load(model_path)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replicates training-time preprocessing exactly. Accepts a dataframe with
        raw column values (e.g. Sex='F'/'M', Ascites='Y'/'N'/NaN, Edema='N'/'S'/'Y',
        Drug='D-penicillamine'/'Placebo'/NaN, Age in DAYS as in the original dataset).
        """
        data = df.copy()

        missing_cols = set(self.RAW_FEATURE_COLUMNS) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required input columns: {missing_cols}")

        # Binary encodings
        data["Sex"] = data["Sex"].map({"F": 0, "M": 1})
        for col in ["Ascites", "Hepatomegaly", "Spiders"]:
            data[col] = data[col].map({"N": 0, "Y": 1})

        # Ordinal encoding (severity gradient: N < S < Y)
        data["Edema"] = data["Edema"].map({"N": 0, "S": 1, "Y": 2})

        # One-hot encode Drug, keeping NaN as an explicit category
        drug_dummies = pd.get_dummies(data["Drug"], prefix="Drug", dummy_na=True)
        data = pd.concat([data.drop(columns=["Drug"]), drug_dummies], axis=1)

        # Ensure all expected dummy columns exist (in case a category is absent
        # from a small inference batch)
        for dummy_col in ["Drug_D-penicillamine", "Drug_Placebo", "Drug_nan"]:
            if dummy_col not in data.columns:
                data[dummy_col] = False

        # Reorder columns to match training exactly. NaNs in Ascites/Hepatomegaly/
        # Spiders/Alk_Phos/SGOT/Cholesterol/Copper/Tryglicerides/Platelets/
        # Prothrombin/Stage are intentionally left as-is for XGBoost's native
        # missing-value handling.
        data = data[self.MODEL_FEATURE_ORDER]

        return data

    def predict(self, patient: dict, threshold: float = 0.5) -> dict:
        """
        Predict mortality risk for a single patient.

        Parameters
        ----------
        patient : dict
            Raw feature values, e.g.:
            {
                "Age": 21464, "Sex": "F", "Ascites": "N", "Hepatomegaly": "Y",
                "Spiders": "N", "Edema": "N", "Bilirubin": 1.4, "Cholesterol": 261,
                "Albumin": 3.7, "Copper": 76, "Alk_Phos": 1718, "SGOT": 137.95,
                "Tryglicerides": 172, "Platelets": 190, "Prothrombin": 10.6,
                "Stage": 3.0, "Drug": "D-penicillamine"
            }
        threshold : float
            Decision threshold on predicted probability (default 0.5).

        Returns
        -------
        dict with keys: risk_probability, predicted_label, label_name
        """
        df = pd.DataFrame([patient])
        X = self._preprocess(df)

        proba = float(self.model.predict_proba(X)[0, 1])
        label = int(proba >= threshold)

        return {
            "risk_probability": round(proba, 4),
            "predicted_label": label,
            "label_name": "high_risk_death" if label == 1 else "lower_risk_survive",
        }

    def predict_batch(self, patients_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Predict for multiple patients at once. Returns original df + prediction columns."""
        X = self._preprocess(patients_df)
        probas = self.model.predict_proba(X)[:, 1]
        labels = (probas >= threshold).astype(int)

        result = patients_df.copy()
        result["risk_probability"] = np.round(probas, 4)
        result["predicted_label"] = labels
        return result


if __name__ == "__main__":
    # Example usage
    predictor = CirrhosisRiskPredictor(model_path="xgb_cirrhosis_model.pkl")

    example_patient = {
        "Age": 21464,  # in days (~58.8 years)
        "Sex": "F",
        "Ascites": "N",
        "Hepatomegaly": "Y",
        "Spiders": "N",
        "Edema": "N",
        "Bilirubin": 1.4,
        "Cholesterol": 261,
        "Albumin": 3.7,
        "Copper": 76,
        "Alk_Phos": 1718,
        "SGOT": 137.95,
        "Tryglicerides": 172,
        "Platelets": 190,
        "Prothrombin": 10.6,
        "Stage": 3.0,
        "Drug": "D-penicillamine",
    }

    result = predictor.predict(example_patient)
    print(result)