import pickle
import numpy as np
import pandas as pd

from src.clustering import FEATURE_COLS


def load_model(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_thresholds(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def add_derived_features(point: dict) -> dict:
    """Compute cyclic time and interaction features for a raw feature dict."""
    point = point.copy()
    h = point['hour']
    m = point['month']
    point['hour_sin']  = np.sin(2 * np.pi * h / 24)
    point['hour_cos']  = np.cos(2 * np.pi * h / 24)
    point['month_sin'] = np.sin(2 * np.pi * m / 12)
    point['month_cos'] = np.cos(2 * np.pi * m / 12)
    vis = point.get('Visibility(mi)', 10.0)
    point['rush_rain'] = point.get('is_rush_hour', 0) * point.get('is_raining', 0)
    point['night_vis'] = point.get('is_night', 0) * (1 / (vis + 0.1))
    point['rain_wind'] = point.get('rain_intensity', 0) * point.get('Wind_Speed(mph)', 0.0)
    return point


def predict_hotspot(model, point: dict, threshold: float = 0.5) -> dict:
    """Predict whether a single location/time point is a hotspot.

    Args:
        model: Trained sklearn-compatible model with predict_proba.
        point: Dict with keys matching FEATURE_COLS (and possibly extra UI keys).
        threshold: Decision threshold (default 0.5).

    Returns:
        Dict with 'probability', 'prediction', and 'alert_level'.
    """
    # Lọc ĐÚNG các cột cần thiết cho Machine Learning, bỏ qua các cột hiển thị UI (Location_State...)
    df = pd.DataFrame([point])[FEATURE_COLS]
    
    proba = float(model.predict_proba(df)[0][1])
    prediction = int(proba >= threshold)

    # Logic Ngưỡng Động: Bám sát vào mức Threshold tối ưu của mô hình
    if proba >= (threshold + 0.1):
        alert = "HIGH RISK"
    elif proba >= (threshold - 0.1):
        alert = "CAUTION"
    else:
        alert = "SAFE"

    return {'probability': proba, 'prediction': prediction, 'alert_level': alert}


def batch_predict(model, cases: list[dict], base: dict, threshold: float = 0.5) -> list[dict]:
    """Run predictions on multiple scenario cases merged with a base config."""
    results = []
    for case in cases:
        desc = case.get('desc', '')
        point = {**base, **{k: v for k, v in case.items() if k != 'desc'}}
        point = add_derived_features(point)
        result = predict_hotspot(model, point, threshold)
        result['desc'] = desc
        results.append(result)
        print(f"{desc:<45} {result['probability']:>9.2%}  {result['alert_level']}")
    return results