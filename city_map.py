"""
City-level synthetic data generator for Kolkata EMS simulation:

Generates:
1) zones_kolkata.csv
2) historical_incidents_kolkata.csv
3) risk_profiles_kolkata.csv

If 'hospitals_kolkata.csv' exists, it uses real hospital locations
to compute nearby hospital counts and distances for each zone.
"""

import os
import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------
# CONFIG
# ---------------------------

CITY_NAME = "Kolkata"

# Kolkata approx center & spread
CITY_CENTER_LAT = 22.5726
CITY_CENTER_LNG = 88.3639
LAT_SPREAD = 0.25
LNG_SPREAD = 0.25

NUM_ZONES = 20               # how many zones to create
DAYS_HISTORY = 365           # how many past days of incidents
AVG_INCIDENTS_PER_DAY = 15   # city-wide average

TIME_WINDOWS = [
    ("00:00", "06:00"),
    ("06:00", "12:00"),
    ("12:00", "18:00"),
    ("18:00", "24:00"),
]

SEVERITY_MAPPING = {
    "LOW": 1,
    "MED": 2,
    "HIGH": 3,
    "CRITICAL": 4,
    "FATAL": 5,
}

INCIDENT_TYPES = [
    "ROAD_ACCIDENT",
    "FIRE",
    "CARDIAC_EMERGENCY",
    "RESPIRATORY_DISTRESS",
    "TRAUMA_OTHER",
]

WEATHER_CONDITIONS = ["CLEAR", "RAIN", "FOG", "CLOUDY"]

# Weights for risk score components
RISK_WEIGHTS = {
    "F": 0.25,  # frequency
    "S": 0.35,  # severity
    "R": 0.15,  # response time
    "Z": 0.10,  # static zone risk
    "T": 0.15,  # traffic (avg_traffic_index)
}

random.seed(123)
np.random.seed(123)


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between two lat/lng points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def random_point_in_city():
    lat = CITY_CENTER_LAT + random.uniform(-LAT_SPREAD, LAT_SPREAD)
    lng = CITY_CENTER_LNG + random.uniform(-LNG_SPREAD, LNG_SPREAD)
    return round(lat, 6), round(lng, 6)


def day_type_from_date(dt: datetime) -> str:
    # 3% chance of "festive" day
    if random.random() < 0.03:
        return "FESTIVE"
    if dt.weekday() >= 5:
        return "WEEKEND"
    return "WEEKDAY"


def time_window_for_datetime(dt: datetime) -> str:
    h = dt.hour
    for start, end in TIME_WINDOWS:
        s_h = int(start.split(":")[0])
        e_h = int(end.split(":")[0])
        if s_h <= h < e_h or (s_h == 18 and h >= 18 and e_h == 24):
            return f"{start}-{end}"
    return "00:00-06:00"


def load_hospitals():
    if os.path.exists("hospitals_kolkata.csv"):
        print("Found hospitals_kolkata.csv – using real hospital positions for zone features.")
        return pd.read_csv("hospitals_kolkata.csv")
    print("No hospitals_kolkata.csv found – generating zone features without hospital info.")
    return None


# ---------------------------
# 1) GENERATE ZONES
# ---------------------------

def generate_zones(num_zones: int):
    hospitals_df = load_hospitals()

    zones = []
    for i in range(1, num_zones + 1):
        zone_id = f"ZONE_{i:03d}"
        center_lat, center_lng = random_point_in_city()

        # Static zone features
        population_density = random.uniform(5000, 40000)    # people/km^2
        road_density = random.uniform(2, 20)                # km road / km^2
        major_intersections = random.randint(1, 15)
        black_spots_count = random.randint(0, 8)
        avg_traffic_index = random.uniform(0.3, 1.0)        # 0–1
        commercial_activity_index = random.uniform(0.1, 1.0)

        nearby_hospital_count = 0
        avg_distance_to_nearest_hospital_km = None

        if hospitals_df is not None and len(hospitals_df) > 0:
            distances = []
            for _, row in hospitals_df.iterrows():
                d = haversine(center_lat, center_lng, float(row["lat"]), float(row["lng"]))
                distances.append(d)
            distances = sorted(distances)
            nearby_hospital_count = sum(d <= 5.0 for d in distances)
            if len(distances) > 0:
                avg_distance_to_nearest_hospital_km = distances[0]

        zones.append({
            "zone_id": zone_id,
            "zone_name": f"Kolkata Zone {i}",
            "city": CITY_NAME,
            "center_lat": center_lat,
            "center_lng": center_lng,
            "population_density": round(population_density, 2),
            "road_density": round(road_density, 2),
            "major_intersections": major_intersections,
            "black_spots_count": black_spots_count,
            "avg_traffic_index": round(avg_traffic_index, 2),
            "commercial_activity_index": round(commercial_activity_index, 2),
            "nearby_hospital_count": nearby_hospital_count,
            "avg_distance_to_nearest_hospital_km": round(avg_distance_to_nearest_hospital_km, 2)
                if avg_distance_to_nearest_hospital_km is not None else None,
        })

    zones_df = pd.DataFrame(zones)
    return zones_df


# ---------------------------
# 2) GENERATE HISTORICAL INCIDENTS
# ---------------------------

def generate_historical_incidents(zones_df: pd.DataFrame):
    incidents = []
    zone_ids = zones_df["zone_id"].tolist()

    # base incident rate per zone ~ population_density + traffic
    pop = zones_df["population_density"].values
    traf = zones_df["avg_traffic_index"].values
    base_rates = pop * 0.00002 + traf * 3
    base_rates = base_rates / base_rates.sum()  # normalized

    start_date = datetime.utcnow() - timedelta(days=DAYS_HISTORY)
    incident_id_counter = 1

    for day_offset in range(DAYS_HISTORY):
        date = start_date + timedelta(days=day_offset)
        day_type = day_type_from_date(date)

        # total incidents for the day (Poisson)
        day_incidents = np.random.poisson(AVG_INCIDENTS_PER_DAY)
        for _ in range(day_incidents):
            zone_index = np.random.choice(len(zone_ids), p=base_rates)
            zone_id = zone_ids[zone_index]
            zone_row = zones_df.iloc[zone_index]

            # random time of day
            seconds_in_day = random.randint(0, 24 * 3600 - 1)
            timestamp = date + timedelta(seconds=seconds_in_day)
            time_window = time_window_for_datetime(timestamp)

            # position around zone center
            base_lat, base_lng = zone_row["center_lat"], zone_row["center_lng"]
            lat = base_lat + random.uniform(-0.01, 0.01)
            lng = base_lng + random.uniform(-0.01, 0.01)

            incident_type = random.choice(INCIDENT_TYPES)

            # severity
            sev_category = random.choices(
                ["LOW", "MED", "HIGH", "CRITICAL", "FATAL"],
                weights=[0.25, 0.35, 0.2, 0.15, 0.05],
                k=1
            )[0]
            sev_numeric = SEVERITY_MAPPING[sev_category]

            num_injured = 1
            if sev_category in ["HIGH", "CRITICAL", "FATAL"]:
                num_injured = random.randint(1, 4)
            num_critical = 1 if sev_category in ["CRITICAL", "FATAL"] else 0
            num_fatal = 1 if sev_category == "FATAL" else 0

            # response & transport time (in minutes)
            response_time = random.uniform(5.0, 25.0)
            if sev_category in ["CRITICAL", "FATAL"]:
                response_time += random.uniform(-3.0, 3.0)
            transport_time = random.uniform(5.0, 40.0)

            weather = random.choices(
                WEATHER_CONDITIONS,
                weights=[0.6, 0.25, 0.1, 0.05],
                k=1
            )[0]
            road_condition = random.choice(["DRY", "WET", "DAMAGED"])

            incidents.append({
                "incident_id": f"INC_{incident_id_counter:06d}",
                "timestamp": timestamp.isoformat(),
                "date": timestamp.date().isoformat(),
                "time_window": time_window,
                "day_type": day_type,
                "lat": round(lat, 6),
                "lng": round(lng, 6),
                "zone_id": zone_id,
                "incident_type": incident_type,
                "num_injured": num_injured,
                "num_critical": num_critical,
                "num_fatalities": num_fatal,
                "severity_category": sev_category,
                "severity_numeric": sev_numeric,
                "response_time_minutes": round(response_time, 2),
                "transport_time_minutes": round(transport_time, 2),
                "weather_condition": weather,
                "road_condition": road_condition,
                "was_night_time": timestamp.hour >= 20 or timestamp.hour < 6,
            })

            incident_id_counter += 1

    incidents_df = pd.DataFrame(incidents)
    return incidents_df


# ---------------------------
# 3) COMPUTE ZONE_BASE_RISK FROM STATIC FEATURES
# ---------------------------

def compute_zone_base_risk(zones_df: pd.DataFrame):
    """
    Compute zone_base_risk from static features using normalized
    population_density, road_density, black_spots_count, avg_traffic_index.
    """
    z = zones_df.copy()

    def norm(col):
        vals = z[col].values.astype(float)
        if vals.max() == vals.min():
            return np.ones_like(vals) * 0.5
        return (vals - vals.min()) / (vals.max() - vals.min())

    PD = norm("population_density")
    RD = norm("road_density")
    BS = norm("black_spots_count")
    TI = norm("avg_traffic_index")

    zone_base_risk = 0.3 * PD + 0.2 * RD + 0.3 * BS + 0.2 * TI
    z["zone_base_risk"] = np.round(zone_base_risk, 3)
    return z


# ---------------------------
# 4) BUILD RISK PROFILES (includes TRAFFIC factor)
# ---------------------------

def generate_risk_profiles(zones_df: pd.DataFrame, incidents_df: pd.DataFrame):
    # Attach zone_base_risk and avg_traffic_index
    zones_with_risk = compute_zone_base_risk(zones_df)
    zone_base_risk_map = zones_with_risk.set_index("zone_id")["zone_base_risk"].to_dict()
    zone_traffic_map = zones_with_risk.set_index("zone_id")["avg_traffic_index"].to_dict()

    group_cols = ["zone_id", "time_window", "day_type"]
    grouped = incidents_df.groupby(group_cols)

    rows = []
    for (zone_id, time_window, day_type), g in grouped:
        incident_count = len(g)
        if incident_count == 0:
            continue

        days_observed = DAYS_HISTORY
        avg_incidents_per_day = incident_count / days_observed

        critical_incidents_count = g[g["severity_category"].isin(["CRITICAL", "FATAL"])].shape[0]
        fatalities_count = g["num_fatalities"].sum()
        avg_severity_score = g["severity_numeric"].mean()
        avg_response_time_min = g["response_time_minutes"].mean()
        max_response_time_min = g["response_time_minutes"].max()
        avg_num_injured = g["num_injured"].mean()
        night_incident_ratio = g["was_night_time"].mean()
        bad_weather_ratio = g[g["weather_condition"].isin(["RAIN", "FOG"])] \
            .shape[0] / incident_count

        zone_base_risk = zone_base_risk_map.get(zone_id, 0.5)
        avg_traffic_index = zone_traffic_map.get(zone_id, 0.5)

        rows.append({
            "zone_id": zone_id,
            "time_window": time_window,
            "day_type": day_type,
            "incident_count": incident_count,
            "avg_incidents_per_day": avg_incidents_per_day,
            "critical_incidents_count": critical_incidents_count,
            "fatalities_count": fatalities_count,
            "avg_severity_score": avg_severity_score,
            "avg_response_time_min": avg_response_time_min,
            "max_response_time_min": max_response_time_min,
            "avg_num_injured": avg_num_injured,
            "night_incident_ratio": night_incident_ratio,
            "bad_weather_ratio": bad_weather_ratio,
            "zone_base_risk": zone_base_risk,
            "avg_traffic_index": avg_traffic_index,
        })

    rp = pd.DataFrame(rows)

    if rp.empty:
        return rp

    # ------------- NORMALIZATION FOR RISK SCORE -------------

    def norm_series(s: pd.Series):
        vals = s.values.astype(float)
        if len(vals) == 0 or vals.max() == vals.min():
            return np.ones_like(vals) * 0.5
        return (vals - vals.min()) / (vals.max() - vals.min())

    rp["F_norm"] = norm_series(rp["avg_incidents_per_day"])
    rp["S_norm"] = norm_series(rp["avg_severity_score"])
    rp["R_norm"] = norm_series(rp["avg_response_time_min"])
    rp["Z_norm"] = norm_series(rp["zone_base_risk"])
    rp["T_norm"] = norm_series(rp["avg_traffic_index"])

    w = RISK_WEIGHTS
    rp["risk_score"] = (
        w["F"] * rp["F_norm"] +
        w["S"] * rp["S_norm"] +
        w["R"] * rp["R_norm"] +
        w["Z"] * rp["Z_norm"] +
        w["T"] * rp["T_norm"]
    ).round(3)

    return rp


# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    print("Generating zones...")
    zones_df = generate_zones(NUM_ZONES)
    zones_df.to_csv("zones_kolkata.csv", index=False)
    print(f"Saved zones_kolkata.csv with {len(zones_df)} zones")

    print("Generating historical incidents...")
    incidents_df = generate_historical_incidents(zones_df)
    incidents_df.to_csv("historical_incidents_kolkata.csv", index=False)
    print(f"Saved historical_incidents_kolkata.csv with {len(incidents_df)} incidents")

    print("Generating risk profiles...")
    risk_profiles_df = generate_risk_profiles(zones_df, incidents_df)
    risk_profiles_df.to_csv("risk_profiles_kolkata.csv", index=False)
    print(f"Saved risk_profiles_kolkata.csv with {len(risk_profiles_df)} rows")

    print("Done.")
