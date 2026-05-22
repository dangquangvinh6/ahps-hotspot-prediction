from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import pickle
from datetime import datetime, timezone, timedelta
import requests
from dotenv import load_dotenv

from app.schemas import Coordinates, PredictionRequest
from src.preprocessing import _group_weather
from src.predict import load_thresholds, add_derived_features, load_model, predict_hotspot
from src.osm_lookup import get_infra_features

load_dotenv()
OWM_API_KEY = os.getenv("OWM_API_KEY")
MODELS_DIR = "models"

ml_components = {}

STATE_MAP = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'District of Columbia': 'DC'
}


def load_ml_assets():
    model_path = os.path.join(MODELS_DIR, "model_xgboost.pkl")
    enc_path = os.path.join(MODELS_DIR, "encoders.pkl")
    thresh_path = os.path.join(MODELS_DIR, "best_thresholds.pkl")

    if not os.path.exists(model_path) or not os.path.exists(enc_path):
        print("CẢNH BÁO: Không tìm thấy model hoặc encoders. Kiểm tra lại thư mục models/")
        return

    ml_components["model"] = load_model(model_path)
    with open(enc_path, "rb") as f:
        ml_components["encoders"] = pickle.load(f)
        
    thresholds = load_thresholds(thresh_path) if os.path.exists(thresh_path) else {}
    ml_components["threshold"] = thresholds.get("XGBoost", 0.5)
    print("Đã load XGBoost model (No-SMOTE) và encoders vào RAM thành công!")


app = FastAPI(title="AHPS Web Demo", on_startup=[load_ml_assets])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


def _deg_to_compass(deg: float) -> str:
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    return dirs[round(deg / 45) % 8]


def _encode_safe(encoder, value: str, fallback_value: str) -> int:
    try:
        return int(encoder.transform([value])[0])
    except Exception:
        try:
            return int(encoder.transform([fallback_value])[0])
        except Exception:
            return 0


def get_state_county(lat: float, lng: float) -> tuple[str, str, str, str]:
    """Trả về 4 thông số: Mã Bang, Mã Hạt, Tên Bang Thật, Tên Hạt Thật để hiện lên UI"""
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lng}&format=json&addressdetails=1&accept-language=en"
    headers = {"User-Agent": "AHPS-Hotspot-Prediction-Demo/1.0 (contact: student@uit.edu.vn)"}
    
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            addr = res.json().get("address", {})
            
            # Lấy tên hiển thị
            state_full = addr.get("state", "Unknown")
            county_raw = addr.get("county", "Unknown")
            county_clean = county_raw.replace(" County", "") 
            
            # Lấy mã chuẩn để AI đọc
            state_abbr = STATE_MAP.get(state_full, "CA") 
            county_str = f"{state_abbr}_{county_clean}" if county_clean else "Other"
            
            return state_abbr, county_str, state_full, county_clean
    except Exception:
        pass
        
    return "CA", "Other", "Unknown State", "Unknown County"


@app.get("/")
def read_index():
    return FileResponse("static/index.html")


@app.post("/extract_features")
def extract_features(coords: Coordinates):
    if not OWM_API_KEY:
        raise HTTPException(status_code=500, detail="Thiếu OWM_API_KEY trong file .env")
    if "encoders" not in ml_components:
        raise HTTPException(status_code=500, detail="Model chưa được load.")

    encoders = ml_components["encoders"]
    
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": coords.lat, "lon": coords.lng, "appid": OWM_API_KEY, "units": "imperial"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Lỗi kết nối OpenWeatherMap: {str(e)}")
    
    main_data = data.get("main") or {}
    wind_data = data.get("wind") or {}
    weather_list = data.get("weather") or [{}]
    sys_data = data.get("sys") or {}
    rain_data = data.get("rain") or {}

    temperature  = main_data.get("temp", 65.0)
    humidity     = main_data.get("humidity", 50)
    pressure     = main_data.get("pressure", 1012) * 0.02953
    
    # [FIX 1] Xử lý lỗi tầm nhìn API OWM (đỉnh ở 10,000m ~ 6.21 miles)
    visibility_raw = data.get("visibility", 16093) / 1609.34
    visibility = 10.0 if visibility_raw >= 6.2 else round(visibility_raw, 2)

    wind_speed   = wind_data.get("speed", 0.0)
    wind_deg     = wind_data.get("deg", 0)
    weather_desc = weather_list[0].get("description", "clear sky")
    rain_1h_mm   = rain_data.get("1h", 0.0)
    
    now_ts = datetime.now(timezone.utc).timestamp()
    sunrise_ts   = sys_data.get("sunrise", now_ts - 43200)
    sunset_ts    = sys_data.get("sunset", now_ts + 43200)

    rain_1h_in = rain_1h_mm / 25.4
    is_raining = int(rain_1h_in > 0)
    if rain_1h_in == 0: rain_intensity = 0
    elif rain_1h_in <= 0.10: rain_intensity = 1
    elif rain_1h_in <= 0.30: rain_intensity = 2
    else: rain_intensity = 3

    # [FIX 2] Xử lý gió đứng im
    if wind_speed == 0:
        compass = "CALM"
    else:
        compass = _deg_to_compass(wind_deg)
        
    weather_group = _group_weather(weather_desc)
    
    tz_offset = data.get("timezone", 0) 
    local_now = datetime.now(timezone.utc) + timedelta(seconds=tz_offset)
    
    hour = local_now.hour
    is_rush_hour = int(hour in [7, 8, 9, 16, 17, 18, 19])
    is_night = int(not (sunrise_ts < now_ts < sunset_ts))

    infra = get_infra_features(coords.lat, coords.lng, db_path=os.path.join(MODELS_DIR, 'infra_lookup.db'))
    
    # Lấy thông tin Tên Tỉnh/Hạt
    state_abbr, county_str, state_name, county_name = get_state_county(coords.lat, coords.lng)

    point = {
        "Location_State": state_name,      # Dành cho UI hiển thị
        "Location_County": county_name,    # Dành cho UI hiển thị
        "hour": hour, "month": local_now.month, "day_of_week": local_now.weekday(),
        "is_rush_hour": is_rush_hour, "is_night": is_night,
        "Temperature(F)": round(temperature, 1), "Humidity(%)": humidity,
        "Pressure(in)": round(pressure, 2), "Visibility(mi)": visibility,
        "Wind_Speed(mph)": round(wind_speed, 1),
        "is_raining": is_raining, "rain_intensity": rain_intensity,
        **infra,
        "Weather_enc": _encode_safe(encoders["weather"], weather_group, fallback_value="Clear"),
        "WindDir_enc": _encode_safe(encoders["wind"], compass, fallback_value="CALM"),
        "State_enc": _encode_safe(encoders["state"], state_abbr, fallback_value="CA"),
        "County_enc": _encode_safe(encoders["county"], county_str, fallback_value="Other"),
    }
    
    return point


@app.post("/predict")
def predict(req: PredictionRequest):
    if "model" not in ml_components:
        raise HTTPException(status_code=500, detail="Model chưa được load.")
        
    point = req.features
    point = add_derived_features(point)
    result = predict_hotspot(ml_components["model"], point, ml_components["threshold"])
    
    return {
        "prediction": result['alert_level'],
        "probability": f"{result['probability']:.2%}"
    }