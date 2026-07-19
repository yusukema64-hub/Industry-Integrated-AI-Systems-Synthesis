"""
readmission_pipeline.py
=======================
Reusable module for the Healthcare Readmission Risk Platform.
Encapsulates feature engineering, model ensemble, patient summary
generation, and the agentic triage loop so they can be imported
into the main notebook or any downstream script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from typing import List, Dict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURES = [
    "age", "num_diagnoses", "los_days", "num_meds", "num_prior_admits",
    "has_diabetes", "has_chf", "has_copd", "insurance_enc", "gender_enc",
    "risk_index",
]
TARGET = "readmitted_30d"

TIER_THRESHOLDS = {"HIGH": 0.55, "MODERATE": 0.35}


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add encoded and composite features to a raw EHR DataFrame."""
    out = df.copy()
    le = LabelEncoder()
    out["insurance_enc"] = le.fit_transform(out["insurance"])
    out["gender_enc"]    = (out["gender"] == "M").astype(int)
    out["risk_index"] = (
        out["num_prior_admits"] * 2
        + out["has_chf"] * 1.5
        + out["has_diabetes"]
        + out["has_copd"]
        + (out["num_diagnoses"] > 5).astype(int)
    )
    return out


# ---------------------------------------------------------------------------
# Model layer
# ---------------------------------------------------------------------------

class ReadmissionEnsemble:
    """
    Weighted ensemble: best sklearn ML model (60%) + sklearn MLP (40%).
    Mirrors the Layers 2–3 approach in the main notebook.
    """

    def __init__(
        self,
        ml_weight: float = 0.60,
        dl_weight:  float = 0.40,
        threshold:  float = 0.45,
    ):
        self.ml_weight = ml_weight
        self.dl_weight  = dl_weight
        self.threshold  = threshold

        self.scaler = StandardScaler()
        self.rf     = RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
        )
        self.lr     = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced")
        self.mlp    = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16), max_iter=300,
            random_state=42, early_stopping=True,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ReadmissionEnsemble":
        Xs = self.scaler.fit_transform(X)
        self.rf.fit(Xs, y)
        self.lr.fit(Xs, y)
        self.mlp.fit(Xs, y)
        # Choose best ML model by training AUC (for reproducibility)
        rf_auc = roc_auc_score(y, self.rf.predict_proba(Xs)[:, 1])
        lr_auc = roc_auc_score(y, self.lr.predict_proba(Xs)[:, 1])
        self._best_ml = self.rf if rf_auc >= lr_auc else self.lr
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() before predict_proba()"
        Xs  = self.scaler.transform(X)
        ml_p = self._best_ml.predict_proba(Xs)[:, 1]
        dl_p = self.mlp.predict_proba(Xs)[:, 1]
        return self.ml_weight * ml_p + self.dl_weight * dl_p

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)


# ---------------------------------------------------------------------------
# Generative AI layer (template-based stub)
# ---------------------------------------------------------------------------

def generate_patient_summary(patient: dict | pd.Series, risk_prob: float) -> str:
    """
    Plain-language patient risk explanation.
    Replace the body with an LLM API call for production use.
    """
    if isinstance(patient, dict):
        patient = pd.Series(patient)

    tier = assign_tier(risk_prob)
    icon = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}[tier]

    factors: List[str] = []
    if patient.get("num_prior_admits", 0) >= 2:
        factors.append(f"{int(patient['num_prior_admits'])} prior admissions")
    if patient.get("has_chf"):
        factors.append("active CHF")
    if patient.get("has_diabetes"):
        factors.append("diabetes mellitus")
    if patient.get("has_copd"):
        factors.append("COPD")
    if patient.get("los_days", 0) > 7:
        factors.append(f"LOS {patient['los_days']:.0f} days")
    if not factors:
        factors.append("multi-morbidity profile")

    actions = {
        "HIGH"    : "48-hr call + home health visit within 72 hrs",
        "MODERATE": "PCP follow-up within 7 days",
        "LOW"     : "Standard discharge pathway",
    }

    return (
        f"{icon} Risk Level: {tier} ({risk_prob:.1%})\n"
        f"   Factors : {', '.join(factors[:3])}\n"
        f"   Action  : {actions[tier]}\n"
        f"   ⚠ Review required by licensed clinician before care decision."
    )


# ---------------------------------------------------------------------------
# Agentic triage layer
# ---------------------------------------------------------------------------

def assign_tier(prob: float) -> str:
    if prob >= TIER_THRESHOLDS["HIGH"]:
        return "HIGH"
    if prob >= TIER_THRESHOLDS["MODERATE"]:
        return "MODERATE"
    return "LOW"


def dispatch_action(tier: str, patient_id: int | str) -> str:
    messages = {
        "HIGH"    : f"[DISPATCH] Patient {patient_id}: Care manager review + 48-hr call scheduled",
        "MODERATE": f"[DISPATCH] Patient {patient_id}: 7-day PCP follow-up booked",
        "LOW"     : f"[DISPATCH] Patient {patient_id}: Standard discharge",
    }
    return messages[tier]


def run_triage_agent(
    patient_ids: List[int],
    df: pd.DataFrame,
    model: ReadmissionEnsemble,
) -> pd.DataFrame:
    """
    Agentic orchestration loop (Project 6 integration).

    For each patient ID in the discharge queue:
      1. Retrieve EHR features
      2. Score readmission risk via ensemble model
      3. Assign risk tier
      4. Generate plain-language summary
      5. Dispatch care team action
    """
    records = []
    for pid in patient_ids:
        row    = df.iloc[pid % len(df)]
        x      = row[FEATURES].values.reshape(1, -1)
        prob   = float(model.predict_proba(x)[0])
        tier   = assign_tier(prob)
        summary= generate_patient_summary(row, prob)
        action = dispatch_action(tier, pid)
        records.append(
            {
                "patient_id": pid,
                "risk_prob" : round(prob, 4),
                "tier"      : tier,
                "summary"   : summary,
                "action"    : action,
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    np.random.seed(0)
    N = 200
    data = pd.DataFrame(
        {
            "age"             : np.random.normal(65, 12, N).clip(18, 90),
            "num_diagnoses"   : np.random.poisson(4, N).clip(1, 15),
            "los_days"        : np.random.exponential(5, N).clip(1, 30),
            "num_meds"        : np.random.poisson(8, N).clip(0, 30),
            "num_prior_admits": np.random.poisson(1.5, N).clip(0, 10),
            "has_diabetes"    : np.random.binomial(1, 0.35, N),
            "has_chf"         : np.random.binomial(1, 0.20, N),
            "has_copd"        : np.random.binomial(1, 0.18, N),
            "insurance"       : np.random.choice(["Medicare","Medicaid","Private","Self-pay"], N),
            "gender"          : np.random.choice(["F","M"], N),
            "readmitted_30d"  : np.random.binomial(1, 0.28, N),
        }
    )
    data = engineer_features(data)

    X = data[FEATURES].values
    y = data[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = ReadmissionEnsemble()
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test))
    print(f"Smoke-test AUC: {auc:.4f}")

    triage = run_triage_agent(list(range(5)), data, model)
    print(triage[["patient_id","risk_prob","tier","action"]].to_string(index=False))
    print("✓ Module smoke-test passed")
