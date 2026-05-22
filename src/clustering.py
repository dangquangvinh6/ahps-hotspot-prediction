import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


FEATURE_COLS = [
    'hour', 'month', 'day_of_week', 'is_rush_hour', 'is_night',
    'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
    'is_raining',
    'Amenity', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
    'Railway', 'Station', 'Stop', 'Traffic_Signal',
    'Weather_enc', 'WindDir_enc', 'State_enc', 'County_enc', 'rain_intensity',
    # Cyclic time encoding
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    # Interaction features
    'rush_rain', 'night_vis', 'rain_wind',
]

BUCKETS = [
    'early_morning_weekday', 'early_morning_weekend',
    'morning_peak_weekday', 'morning_peak_weekend',
    'midday_weekday', 'midday_weekend',
    'evening_peak_weekday', 'evening_peak_weekend',
    'evening_weekday', 'evening_weekend',
]

EPS_METERS = 300


def _time_bucket(hour: int, dow: int) -> str:
    is_weekend = dow in [5, 6]
    if 0 <= hour < 6:
        slot = 'early_morning'
    elif 6 <= hour < 9:
        slot = 'morning_peak'
    elif 9 <= hour < 16:
        slot = 'midday'
    elif 16 <= hour < 19:
        slot = 'evening_peak'
    else:
        slot = 'evening'
    return f"{slot}_{'weekend' if is_weekend else 'weekday'}"


def run_dbscan_labeling(df: pd.DataFrame) -> pd.DataFrame:
    eps_rad = EPS_METERS / 1000 / 6371

    df['bucket'] = df.apply(lambda r: _time_bucket(r['hour'], r['day_of_week']), axis=1)
    df['cluster'] = -1
    df['is_hotspot'] = 0

    print(f"{'Bucket':<30} {'N rows':>8} {'min_s':>6} {'Clusters':>9} {'Hotspot%':>10} {'Noise%':>8}")
    print("-" * 75)

    cluster_offset = 0
    for bucket in BUCKETS:
        mask = df['bucket'] == bucket
        subset = df[mask].copy()
        n = len(subset)

        if n > 50_000:
            min_samples = 10
        elif n > 20_000:
            min_samples = 7
        else:
            min_samples = 5

        coords_rad = np.radians(subset[['Start_Lat', 'Start_Lng']].values)
        db = DBSCAN(eps=eps_rad, min_samples=min_samples,
                    algorithm='ball_tree', metric='haversine', n_jobs=-1)
        labels = db.fit_predict(coords_rad)

        labels_offset = np.where(labels == -1, -1, labels + cluster_offset)
        if (labels != -1).any():
            cluster_offset += int(labels.max()) + 1

        df.loc[mask, 'cluster'] = labels_offset
        df.loc[mask, 'is_hotspot'] = (labels != -1).astype(int)

        n_clus = len(set(labels)) - (1 if -1 in labels else 0)
        n_hot = (labels != -1).sum()
        print(f"  {bucket:<28} {n:>8,} {min_samples:>6} {n_clus:>9,} "
              f"{n_hot / n * 100:>9.1f}% {(n - n_hot) / n * 100:>7.1f}%")

    print(f"\n  Total clusters : {cluster_offset:,}")
    print(f"  Hotspot        : {df['is_hotspot'].sum():,} ({df['is_hotspot'].mean() * 100:.1f}%)")
    print(f"  Noise          : {(df['is_hotspot'] == 0).sum():,} ({(df['is_hotspot'] == 0).mean() * 100:.1f}%)")
    return df


def build_labeled_dataset(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = FEATURE_COLS + ['label']

    df_positive = df[df['is_hotspot'] == 1].copy().reset_index(drop=True)
    df_positive['label'] = 1

    df_noise = df[df['cluster'] == -1].copy().reset_index(drop=True)
    df_noise['label'] = 0

    df_final = pd.concat([
        df_positive[keep_cols].reset_index(drop=True),
        df_noise[keep_cols].reset_index(drop=True),
    ], ignore_index=True)

    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    n_pos = (df_final['label'] == 1).sum()
    n_neg = (df_final['label'] == 0).sum()
    print(f"Final shape          : {df_final.shape}")
    print(f"Label=1 (hotspot)    : {n_pos:,} ({n_pos / len(df_final) * 100:.1f}%)")
    print(f"Label=0 (non-hotspot): {n_neg:,} ({n_neg / len(df_final) * 100:.1f}%)")
    print(f"Imbalance ratio      : {n_neg / n_pos:.2f}")
    return df_final
