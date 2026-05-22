"""
Accidents Hotspot Prediction System (AHPS)
Full pipeline: load -> preprocess -> cluster -> train -> (optional tune) -> predict demo
Real-time weather prediction: --predict --lat <lat> --lng <lng> --api-key <key>
"""
import argparse
import os
import pickle
from datetime import datetime

import requests

from src.data_loader import load_data
from src.preprocessing import run_preprocessing, _group_weather
from src.clustering import run_dbscan_labeling, build_labeled_dataset
from src.modeling import split_data, train_all_models, save_models
from src.predict import load_thresholds, batch_predict, add_derived_features, load_model, predict_hotspot
from src.osm_lookup import get_infra_features


DEFAULT_DATA_PATH = 'data/US_Accidents_March23_sampled_500k.csv'
MODELS_DIR = 'models'
ENCODERS_FILE = 'encoders.pkl'


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(data_path: str, tune: bool = False, n_trials: int = 100) -> None:
    print("=" * 60)
    print("Step 1: Load data")
    print("=" * 60)
    df = load_data(data_path)

    print("\n" + "=" * 60)
    print("Step 2: Preprocessing")
    print("=" * 60)
    df, encoders = run_preprocessing(df)

    print("\n" + "=" * 60)
    print("Step 3: DBSCAN labeling")
    print("=" * 60)
    df = run_dbscan_labeling(df)
    df_final = build_labeled_dataset(df)
    os.makedirs('data', exist_ok=True)
    df_final.to_csv('data/US_Accidents_Labeled_Dataset.csv', index=False)
    print("Labeled dataset saved to data/US_Accidents_Labeled_Dataset.csv")

    print("\n" + "=" * 60)
    print("Step 4: Train/val/test split")
    print("=" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_final)

    print("\n" + "=" * 60)
    print("Step 5: Train models")
    print("=" * 60)
    results = train_all_models(X_train, X_val, X_test, y_train, y_val, y_test)
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_models(results, output_dir=MODELS_DIR)

    # Save label encoders so the real-time predict path can encode weather/wind strings
    enc_path = os.path.join(MODELS_DIR, ENCODERS_FILE)
    with open(enc_path, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"Saved encoders: {enc_path}")

    if tune:
        print("\n" + "=" * 60)
        print("Step 6: Hyperparameter tuning (Optuna)")
        print("=" * 60)
        from src.tuning import tune_lightgbm, tune_xgboost, save_tuned_models
        lgb_model, lgb_th, _ = tune_lightgbm(
            X_train, X_val, X_test, y_train, y_val, y_test, n_trials=n_trials)
        xgb_model, xgb_th, _ = tune_xgboost(
            X_train, X_val, X_test, y_train, y_val, y_test, n_trials=n_trials)
        save_tuned_models(lgb_model, xgb_model, lgb_th, xgb_th, output_dir=MODELS_DIR)

    print("\n" + "=" * 60)
    print("Step 7: Prediction demo")
    print("=" * 60)
    _run_demo(results)


def _run_demo(results: dict) -> None:
    model = results['XGBoost']['model']

    thresh_path = os.path.join(MODELS_DIR, 'best_thresholds.pkl')
    thresholds = load_thresholds(thresh_path) if os.path.exists(thresh_path) else {}
    threshold = thresholds.get('XGBoost', results['XGBoost'].get('threshold', 0.5))
    print(f"Using XGBoost threshold: {threshold:.3f}")

    base = {
        'month': 3, 'Temperature(F)': 65.0, 'Humidity(%)': 70.0,
        'Pressure(in)': 29.8, 'Visibility(mi)': 10.0, 'Wind_Speed(mph)': 8.0,
        'rain_intensity': 0, 'Amenity': 0, 'Crossing': 1, 'Give_Way': 0,
        'No_Exit': 0, 'Railway': 0, 'Station': 0, 'Stop': 0,
        'WindDir_enc': 5, 'Weather_enc': 4, 'State_enc': 3, 'County_enc': 12,
    }

    test_cases = [
        {"desc": "LA, 2am weekend",
         "hour": 0, "day_of_week": 5, "is_night": 1, "is_rush_hour": 0,
         "is_raining": 0, "Junction": 1, "Traffic_Signal": 1},
        {"desc": "LA, 3am weekday",
         "hour": 3, "day_of_week": 2, "is_night": 1, "is_rush_hour": 0,
         "is_raining": 0, "Junction": 1, "Traffic_Signal": 1},
        {"desc": "LA, 8am rush hour weekday",
         "hour": 8, "day_of_week": 0, "is_night": 0, "is_rush_hour": 1,
         "is_raining": 0, "Junction": 1, "Traffic_Signal": 1},
        {"desc": "LA, 8am + heavy rain",
         "hour": 8, "day_of_week": 0, "is_night": 0, "is_rush_hour": 1,
         "is_raining": 1, "Junction": 1, "Traffic_Signal": 1},
        {"desc": "LA, 2am + no Junction/Signal",
         "hour": 0, "day_of_week": 5, "is_night": 1, "is_rush_hour": 0,
         "is_raining": 0, "Junction": 0, "Traffic_Signal": 0},
    ]

    print(f"\n{'Scenario':<45} {'Probability':>11} {'Alert'}")
    print("-" * 75)
    batch_predict(model, test_cases, base, threshold=threshold)


# ---------------------------------------------------------------------------
# Real-time weather prediction
# ---------------------------------------------------------------------------

def _deg_to_compass(deg: float) -> str:
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    return dirs[round(deg / 45) % 8]


def _encode_safe(encoder, value: str, fallback_idx: int = 0) -> int:
    try:
        return int(encoder.transform([value])[0])
    except ValueError:
        return fallback_idx


def fetch_weather(lat: float, lng: float, api_key: str) -> dict:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lng, "appid": api_key, "units": "imperial"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def run_weather_predict(lat: float, lng: float, api_key: str,
                        state: str = "CA", models_dir: str = MODELS_DIR) -> None:
    model_path = os.path.join(models_dir, "model_xgboost.pkl")
    thresh_path = os.path.join(models_dir, "best_thresholds.pkl")
    enc_path    = os.path.join(models_dir, ENCODERS_FILE)

    if not os.path.exists(model_path):
        print("Model not found. Run the training pipeline first.")
        return
    if not os.path.exists(enc_path):
        print("Encoders not found. Re-run the training pipeline.")
        return

    model      = load_model(model_path)
    thresholds = load_thresholds(thresh_path) if os.path.exists(thresh_path) else {}
    threshold  = thresholds.get("XGBoost", 0.5)
    with open(enc_path, "rb") as f:
        encoders = pickle.load(f)

    print(f"Fetching weather for ({lat}, {lng}) ...")
    data = fetch_weather(lat, lng, api_key)

    temperature  = data["main"]["temp"]
    humidity     = data["main"]["humidity"]
    pressure     = data["main"]["pressure"] * 0.02953
    visibility   = data.get("visibility", 16093) / 1609.34
    wind_speed   = data["wind"]["speed"]
    wind_deg     = data["wind"].get("deg", 0)
    weather_desc = data["weather"][0]["description"]
    rain_1h_mm   = data.get("rain", {}).get("1h", 0.0)
    sunrise_ts   = data["sys"]["sunrise"]
    sunset_ts    = data["sys"]["sunset"]

    rain_1h_in = rain_1h_mm / 25.4
    is_raining = int(rain_1h_in > 0)
    if rain_1h_in == 0:
        rain_intensity = 0
    elif rain_1h_in <= 0.10:
        rain_intensity = 1
    elif rain_1h_in <= 0.30:
        rain_intensity = 2
    else:
        rain_intensity = 3

    compass       = _deg_to_compass(wind_deg)
    weather_group = _group_weather(weather_desc)
    weather_enc   = _encode_safe(encoders["weather"], weather_group)
    wind_enc      = _encode_safe(encoders["wind"], compass)
    state_enc     = _encode_safe(encoders["state"], state.upper())
    county_enc    = _encode_safe(encoders["county"], "Other")

    now         = datetime.now()
    hour        = now.hour
    month       = now.month
    day_of_week = now.weekday()
    is_rush_hour = int(hour in [7, 8, 9, 16, 17, 18, 19])
    is_night    = int(not (sunrise_ts < now.timestamp() < sunset_ts))

    print("Fetching road infrastructure from OSM ...")
    infra = get_infra_features(lat, lng, db_path=os.path.join(models_dir, 'infra_lookup.db'))

    point = {
        "hour": hour, "month": month, "day_of_week": day_of_week,
        "is_rush_hour": is_rush_hour, "is_night": is_night,
        "Temperature(F)": temperature, "Humidity(%)": humidity,
        "Pressure(in)": pressure, "Visibility(mi)": visibility,
        "Wind_Speed(mph)": wind_speed,
        "is_raining": is_raining, "rain_intensity": rain_intensity,
        **infra,
        "Weather_enc": weather_enc, "WindDir_enc": wind_enc,
        "State_enc": state_enc, "County_enc": county_enc,
    }
    point  = add_derived_features(point)
    result = predict_hotspot(model, point, threshold)

    sep = "-" * 62
    print(f"\n{sep}")
    print(f"  Location     : ({lat:.4f}, {lng:.4f})  State: {state.upper()}")
    print(f"  Time         : {now.strftime('%Y-%m-%d %H:%M')}  "
          f"{'Night' if is_night else 'Day'}"
          f"{'  [Rush hour]' if is_rush_hour else ''}")
    print(f"{sep}")
    print(f"  Weather      : {weather_desc}  ->  {weather_group}")
    print(f"  Temperature  : {temperature:.1f} °F")
    print(f"  Humidity     : {humidity} %")
    print(f"  Pressure     : {pressure:.2f} inHg")
    print(f"  Visibility   : {visibility:.2f} mi")
    print(f"  Wind         : {wind_speed:.1f} mph  ({compass})")
    print(f"  Precipitation: {rain_1h_in:.3f} in/h  (intensity={rain_intensity})")
    print(f"{sep}")
    infra_flags = [k for k, v in infra.items() if v == 1]
    print(f"  Road infra   : {', '.join(infra_flags) if infra_flags else 'none detected'}")
    print(f"{sep}")
    print(f"  Hotspot probability : {result['probability']:.2%}")
    print(f"  Alert level         : {result['alert_level']}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AHPS Pipeline / Real-time Predict')

    parser.add_argument('--data', default=DEFAULT_DATA_PATH)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--lat', type=float, default=34.05)
    parser.add_argument('--lng', type=float, default=-118.24)
    parser.add_argument('--api-key', type=str, default='')
    parser.add_argument('--state', type=str, default='CA')
    parser.add_argument('--models', type=str, default=MODELS_DIR)

    args = parser.parse_args()

    if args.predict:
        if not args.api_key:
            parser.error("--api-key is required when using --predict")
        run_weather_predict(
            lat=args.lat, lng=args.lng, api_key=args.api_key,
            state=args.state, models_dir=args.models,
        )
    else:
        run_pipeline(data_path=args.data, tune=args.tune, n_trials=args.n_trials)