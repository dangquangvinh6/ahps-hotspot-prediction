import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


DROP_COLS = [
    'ID', 'Source', 'End_Time', 'End_Lat', 'End_Lng', 'Distance(mi)', 'Severity',
    'Street', 'Zipcode', 'Country', 'Timezone', 'Airport_Code', 'Weather_Timestamp', 'City',
    'Description',
]

REDUNDANT_COLS = [
    'Wind_Chill(F)',
    'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight',
    'Bump', 'Roundabout', 'Turning_Loop', 'Traffic_Calming',
]

OUTLIER_BOUNDS = {
    'Precipitation(in)': (0, 16),
    'Wind_Speed(mph)': (0, 253),
    'Temperature(F)': (-80, 134),
    'Pressure(in)': (26.35, 31.85),
}

NUMERIC_FILL = ['Precipitation(in)', 'Wind_Speed(mph)', 'Visibility(mi)',
                'Humidity(%)', 'Temperature(F)', 'Pressure(in)']

CAT_FILL = ['Weather_Condition', 'Wind_Direction', 'Sunrise_Sunset']

BOOL_COLS = ['Amenity', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
             'Railway', 'Station', 'Stop', 'Traffic_Signal',
             'is_rush_hour', 'is_night', 'is_raining']

COUNTY_RARE_THRESHOLD = 500


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    print(f"Shape after dropping unnecessary columns: {df.shape}")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    for col, (lo, hi) in OUTLIER_BOUNDS.items():
        if col not in df.columns:
            continue
        mask = df[col].between(lo, hi) | df[col].isnull()
        dropped = (~mask).sum()
        df = df[mask]
        print(f"  {col}: dropped {dropped:,} outliers")
    print(f"Shape after outlier removal: {df.shape}")
    return df


def drop_redundant_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in REDUNDANT_COLS if c in df.columns])
    print(f"Shape after dropping redundant features: {df.shape}")
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUMERIC_FILL:
        if c not in df.columns:
            continue
        median = df[c].median()
        df[c] = df[c].fillna(median)

    for c in CAT_FILL:
        if c not in df.columns:
            continue
        mode = df[c].mode()[0]
        df[c] = df[c].fillna(mode)

    print(f"Missing values remaining: {df.isnull().sum().sum()}")
    return df


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='ISO8601')
    df['hour'] = df['Start_Time'].dt.hour
    df['month'] = df['Start_Time'].dt.month
    df['day_of_week'] = df['Start_Time'].dt.dayofweek.astype(int)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18, 19])
    df['is_night'] = (df['Sunrise_Sunset'] == 'Night')
    df = df.drop(columns=['Start_Time', 'Sunrise_Sunset'])
    return df


def engineer_rain_features(df: pd.DataFrame) -> pd.DataFrame:
    df['is_raining'] = (df['Precipitation(in)'] > 0).astype(int)
    df['rain_intensity'] = pd.cut(
        df['Precipitation(in)'].fillna(0),
        bins=[-1, 0, 0.10, 0.30, 999],
        labels=[0, 1, 2, 3]
    ).astype(float).astype(int)
    df = df.drop(columns=['Precipitation(in)'])
    return df


def engineer_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    df['hour_sin']  = np.sin(2 * np.pi * df['hour']  / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour']  / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df


def engineer_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df['rush_rain'] = df['is_rush_hour'] * df['is_raining']
    df['night_vis'] = df['is_night'] * (1 / (df['Visibility(mi)'] + 0.1))
    df['rain_wind'] = df['rain_intensity'] * df['Wind_Speed(mph)']
    return df


def _group_weather(w: str) -> str:
    w = str(w).lower()
    if any(x in w for x in ['rain', 'drizzle', 'shower']):
        return 'Rain'
    if any(x in w for x in ['snow', 'sleet', 'ice', 'wintry']):
        return 'Snow/Ice'
    if any(x in w for x in ['fog', 'haze', 'mist', 'smoke']):
        return 'Fog/Haze'
    if any(x in w for x in ['cloud', 'overcast']):
        return 'Cloudy'
    if any(x in w for x in ['clear', 'fair']):
        return 'Clear'
    if any(x in w for x in ['thunder', 'storm', 't-storm']):
        return 'Storm'
    if any(x in w for x in ['wind', 'dust', 'sand']):
        return 'Wind/Dust'
    if 'partly' in w:
        return 'Partly Cloudy'
    return 'Other'


def encode_categorical_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode categorical features. Returns (df, encoders dict)."""
    encoders = {}

    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(int)

    df['Wind_Direction'] = df['Wind_Direction'].str.upper()
    le_wind = LabelEncoder()
    df['WindDir_enc'] = le_wind.fit_transform(df['Wind_Direction'])
    df = df.drop(columns=['Wind_Direction'])
    encoders['wind'] = le_wind

    df['weather_group'] = df['Weather_Condition'].apply(_group_weather)
    le_weather = LabelEncoder()
    df['Weather_enc'] = le_weather.fit_transform(df['weather_group'])
    df = df.drop(columns=['Weather_Condition', 'weather_group'])
    encoders['weather'] = le_weather

    le_state = LabelEncoder()
    df['State_enc'] = le_state.fit_transform(df['State'])
    df = df.drop(columns=['State'])
    encoders['state'] = le_state

    df['State_County'] = le_state.inverse_transform(df['State_enc'].values) + '_' + df['County']
    county_counts = df['State_County'].value_counts()
    rare = county_counts[county_counts < COUNTY_RARE_THRESHOLD].index
    df['County_grouped'] = df['State_County'].where(~df['State_County'].isin(rare), other='Other')

    le_county = LabelEncoder()
    df['County_enc'] = le_county.fit_transform(df['County_grouped'])
    df = df.drop(columns=['County', 'State_County', 'County_grouped'])
    encoders['county'] = le_county

    print(f"Encoding done. Shape: {df.shape}")
    return df, encoders


def run_preprocessing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = drop_unnecessary_columns(df)
    df = remove_outliers(df)
    df = drop_redundant_features(df)
    df = fill_missing_values(df)
    df = engineer_time_features(df)
    df = engineer_rain_features(df)
    df = engineer_cyclic_features(df)
    df = engineer_interaction_features(df)
    df, encoders = encode_categorical_features(df)
    return df, encoders
