"""OSM infrastructure lookup for the 9 binary road features used in the model."""
import sqlite3
import requests
import time
from pathlib import Path

INFRA_FEATURES = [
    'Amenity', 'Crossing', 'Give_Way', 'Junction',
    'No_Exit', 'Railway', 'Station', 'Stop', 'Traffic_Signal',
]

DEFAULT_INFRA = {k: 0 for k in INFRA_FEATURES}

# Kích thước lưới ~111m x ~111m
_BIN_SIZE   = 0.001  
# Bán kính 100m để bao phủ toàn bộ sai số GPS và ngã tư lớn
_QUERY_DIST = 100      

def _to_bin(lat: float, lng: float) -> tuple:
    """Trả về chỉ mục lưới (bin) và TỌA ĐỘ TÂM của ô lưới đó."""
    lat_bin = round(lat / _BIN_SIZE)
    lng_bin = round(lng / _BIN_SIZE)
    
    # Tính toán tọa độ tâm xác định của ô lưới
    lat_center = lat_bin * _BIN_SIZE
    lng_center = lng_bin * _BIN_SIZE
    
    return lat_bin, lng_bin, lat_center, lng_center


def _init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS infra (
            lat_bin INTEGER, lng_bin INTEGER,
            Amenity INTEGER, Crossing INTEGER, Give_Way INTEGER,
            Junction INTEGER, No_Exit INTEGER, Railway INTEGER,
            Station INTEGER, Stop INTEGER, Traffic_Signal INTEGER,
            PRIMARY KEY (lat_bin, lng_bin)
        )
    """)
    conn.commit()
    return conn


def _read_cache(conn: sqlite3.Connection, lat_bin: int, lng_bin: int):
    cur = conn.execute(
        "SELECT Amenity,Crossing,Give_Way,Junction,No_Exit,Railway,"
        "Station,Stop,Traffic_Signal "
        "FROM infra WHERE lat_bin=? AND lng_bin=?",
        (lat_bin, lng_bin),
    )
    row = cur.fetchone()
    return dict(zip(INFRA_FEATURES, row)) if row is not None else None


def _write_cache(conn: sqlite3.Connection, lat_bin: int, lng_bin: int, feats: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO infra VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (lat_bin, lng_bin,
         feats['Amenity'],  feats['Crossing'],  feats['Give_Way'],
         feats['Junction'], feats['No_Exit'],   feats['Railway'],
         feats['Station'],  feats['Stop'],      feats['Traffic_Signal']),
    )
    conn.commit()


def get_infra_features(lat: float, lng: float,
                       db_path: str = 'models/infra_lookup.db') -> dict:
    """Return the 9 binary infra features for a GPS coordinate."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    lat_bin, lng_bin, lat_center, lng_center = _to_bin(lat, lng)

    conn = _init_db(db_path)
    cached = _read_cache(conn, lat_bin, lng_bin)
    if cached is not None:
        conn.close()
        print(f"  [CACHE] Lấy từ bộ nhớ đệm: {cached}")
        return cached

    feats = DEFAULT_INFRA.copy()

    for attempt in range(3):
        try:
            overpass_url = "https://overpass-api.de/api/interpreter"
            overpass_query = f"""
            [out:json][timeout:15];
            (
              node(around:{_QUERY_DIST},{lat_center},{lng_center});
              way(around:{_QUERY_DIST},{lat_center},{lng_center});
            );
            out tags;
            """
            
            # [BẢN VÁ BẢO MẬT] Bổ sung định danh rõ ràng để vượt qua tường lửa của OSM
            headers = {
                "User-Agent": "AHPS-Hotspot-Prediction-UIT-Project/1.0 (contact: 24522007@ms.uit.edu.vn)",
                "Accept": "application/json"
            }
            
            response = requests.post(
                overpass_url, 
                data={'data': overpass_query}, 
                headers=headers, # Truyền định danh vào yêu cầu
                timeout=15
            )
            response.raise_for_status()

            elements = response.json().get('elements', [])
            feats = {k: 0 for k in INFRA_FEATURES}
            
            for el in elements:
                tags = el.get('tags', {})
                hw = tags.get('highway', '').lower()
                jn = tags.get('junction', '').lower()
                rw = tags.get('railway', '').lower()
                am = tags.get('amenity', '').lower()
                pt = tags.get('public_transport', '').lower()
                ne = tags.get('noexit', '').lower()
                
                if 'traffic_signals' in hw: feats['Traffic_Signal'] = 1
                if 'crossing'        in hw: feats['Crossing']       = 1
                if 'stop'            in hw: feats['Stop']           = 1
                if 'give_way'        in hw: feats['Give_Way']       = 1
                if 'motorway_junction' in hw: feats['Junction']     = 1
                if jn and jn not in ('nan', ''): feats['Junction']  = 1
                if rw and rw not in ('nan', ''):
                    feats['Station' if 'station' in rw else 'Railway'] = 1
                if 'station' in pt: feats['Station']  = 1
                if am and am not in ('nan', ''): feats['Amenity']   = 1
                if ne == 'yes': feats['No_Exit'] = 1

            print(f"  Infra (Overpass API) : {feats}")
            _write_cache(conn, lat_bin, lng_bin, feats)
            break

        except Exception as e:
            if attempt < 2:
                sleep_time = 2 ** attempt
                print(f"  [LỖI] Overpass API: Thử lại lần {attempt + 2} sau {sleep_time}s... ({e})")
                time.sleep(sleep_time)
            else:
                print(f"  [THẤT BẠI] Overpass API ngưng phản hồi, dùng giá trị mặc định.")

    conn.close()
    return feats