"""
Synthetic dataset generator for:
    - Detailed ambulances around Kolkata

If 'hospitals_kolkata.csv' exists in the same folder, it will:
    - Link each ambulance to a base_hospital_id
    - Use hospital coordinates as base_lat/base_lng

Output:
    - ambulances_kolkata.csv
"""

import random
import string
from datetime import datetime, timedelta
import os
import pandas as pd

# ---------------------------
# CONFIG
# ---------------------------

NUM_AMBULANCES = 60          # total ambulances in the city

CITY_NAME = "Kolkata"

# Kolkata approx center
CITY_CENTER_LAT = 22.5726
CITY_CENTER_LNG = 88.3639
LAT_SPREAD = 0.25
LNG_SPREAD = 0.25

AMBULANCE_TYPES = [
    "BLS",         # Basic Life Support
    "ALS",         # Advanced Life Support
    "TRAUMA_CARE",
    "CARDIAC_CARE",
    "NEONATAL_ICU",
    "PATIENT_TRANSPORT"
]

AMBULANCE_STATUSES = [
    "IDLE",
    "EN_ROUTE_TO_SCENE",
    "AT_SCENE",
    "EN_ROUTE_TO_HOSPITAL",
    "AT_HOSPITAL",
    "MAINTENANCE",
    "OFFLINE"
]

FUEL_TYPES = ["DIESEL", "CNG", "ELECTRIC"]
MAKES = ["Force Traveller", "Tata Winger", "Maruti Omni", "Mahindra Supro", "E-Ambulance"]
INFECTION_CONTROL_LEVELS = ["BASIC", "ADVANCED", "HIGH"]
MAINTENANCE_STATUSES = ["OK", "DUE_SOON", "OVERDUE"]

random.seed(123)  # reproducible


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def random_lat_lng():
    """Random (lat, lng) around Kolkata."""
    lat = CITY_CENTER_LAT + random.uniform(-LAT_SPREAD, LAT_SPREAD)
    lng = CITY_CENTER_LNG + random.uniform(-LNG_SPREAD, LNG_SPREAD)
    return round(lat, 6), round(lng, 6)

def generate_ambulance_id(i: int) -> str:
    return f"AMB_{i:03d}"

def generate_registration_number():
    # Simple fake WB registration numbers like WB-12-AB-3456
    rto_code = random.choice(["WB01", "WB02", "WB19", "WB20", "WB24"])
    series = ''.join(random.choices(string.ascii_uppercase, k=2))
    number = random.randint(1000, 9999)
    return f"{rto_code}-{series}-{number}"

def jitter_around(lat, lng, max_delta=0.02):
    """Slight jitter around a base coordinate."""
    return round(lat + random.uniform(-max_delta, max_delta), 6), \
           round(lng + random.uniform(-max_delta, max_delta), 6)

def random_last_maintenance_date():
    days_ago = random.randint(0, 365)
    dt = datetime.utcnow() - timedelta(days=days_ago)
    return dt.date().isoformat()

def load_hospitals_if_available():
    if os.path.exists("hospitals_kolkata.csv"):
        print("Found hospitals_kolkata.csv – using hospitals as bases.")
        return pd.read_csv("hospitals_kolkata.csv")
    else:
        print("No hospitals_kolkata.csv found – generating ambulances with city-base only.")
        return None


# ---------------------------
# GENERATE AMBULANCE DATA
# ---------------------------

def generate_ambulances(num_ambulances: int):
    hospitals_df = load_hospitals_if_available()
    ambulances = []
    now = datetime.utcnow().isoformat()

    for i in range(1, num_ambulances + 1):
        ambulance_id = generate_ambulance_id(i)
        registration_number = generate_registration_number()

        amb_type = random.choice(AMBULANCE_TYPES)
        fuel_type = random.choice(FUEL_TYPES)
        make_model = random.choice(MAKES)
        year_of_manufacture = random.randint(2008, 2024)

        odometer_km = random.randint(20_000, 300_000)
        mileage_kmpl = round(random.uniform(7.0, 13.0), 1)

        # Link to a base hospital if available
        if hospitals_df is not None and len(hospitals_df) > 0:
            base_row = hospitals_df.sample(1).iloc[0]
            base_hospital_id = base_row["hospital_id"]
            base_lat = float(base_row["lat"])
            base_lng = float(base_row["lng"])
        else:
            base_hospital_id = None
            base_lat, base_lng = random_lat_lng()

        # Current location is slightly jittered from base
        current_lat, current_lng = jitter_around(base_lat, base_lng, max_delta=0.03)

        status = random.choices(
            AMBULANCE_STATUSES,
            weights=[0.4, 0.15, 0.05, 0.2, 0.1, 0.05, 0.05],  # mostly active
            k=1
        )[0]

        # ------------------------
        # CREW / STAFFING
        # ------------------------
        crew_paramedics_planned = random.randint(1, 3)
        crew_nurses_planned = random.randint(0, 2)
        crew_doctor_planned = 1 if amb_type in ["ALS", "TRAUMA_CARE", "CARDIAC_CARE", "NEONATAL_ICU"] and random.random() < 0.6 else 0

        # ------------------------
        # ONBOARD EQUIPMENT
        # ------------------------

        # Basic for all
        equipment_oxygen_cylinders = random.randint(1, 6)
        equipment_stretcher_count = random.randint(1, 2)
        equipment_wheelchair_count = random.randint(0, 2)
        equipment_trauma_kit = True
        equipment_suction_machine = random.choice([True, True, False])
        equipment_spine_board = random.choice([True, True, False])

        # Advanced equipment dependent on type
        if amb_type in ["ALS", "TRAUMA_CARE", "CARDIAC_CARE", "NEONATAL_ICU"]:
            equipment_ventilator = random.choice([True, True, True, False])
            equipment_defibrillator = True
            equipment_ecg_monitor = True
            equipment_neo_incubator = True if amb_type == "NEONATAL_ICU" else random.choice([True, False])
        else:
            equipment_ventilator = random.choice([True, False, False])
            equipment_defibrillator = random.choice([True, False])
            equipment_ecg_monitor = random.choice([True, True, False])
            equipment_neo_incubator = False

        infection_control_level = random.choice(INFECTION_CONTROL_LEVELS)
        gps_enabled = True
        realtime_telemetry_enabled = random.choice([True, True, False])
        onboard_camera = random.choice([True, False])

        # Derived capabilities
        can_handle_critical = (
            equipment_ventilator and
            equipment_defibrillator and
            equipment_ecg_monitor and
            crew_paramedics_planned >= 2
        )

        pediatric_capable = (
            amb_type in ["NEONATAL_ICU", "ALS", "TRAUMA_CARE"] or
            equipment_neo_incubator
        )

        # ------------------------
        # PERFORMANCE / HISTORY
        # ------------------------
        trips_per_day_estimate = random.uniform(2.0, 10.0)
        avg_response_time_min = random.uniform(6.0, 20.0)
        avg_turnaround_time_min = random.uniform(45.0, 180.0)
        breakdowns_last_year = random.randint(0, 6)

        maintenance_status = random.choices(
            MAINTENANCE_STATUSES,
            weights=[0.7, 0.2, 0.1],
            k=1
        )[0]
        last_maintenance_date = random_last_maintenance_date()

        utilisation_rate = round(
            min(1.0, max(0.2, trips_per_day_estimate / 12.0)), 2
        )  # crude heuristic

        # ------------------------
        # AMBULANCE QUALITY SCORE
        # ------------------------
        q = 0.25

        # Newer vehicles slightly better
        if year_of_manufacture >= 2018:
            q += 0.15
        elif year_of_manufacture >= 2012:
            q += 0.07

        # Good equipment
        if equipment_ventilator:
            q += 0.1
        if equipment_defibrillator:
            q += 0.08
        if equipment_ecg_monitor:
            q += 0.07
        if pediatric_capable:
            q += 0.05

        # Good telemetry
        if realtime_telemetry_enabled:
            q += 0.07
        if gps_enabled:
            q += 0.05

        # Penalize frequent breakdowns
        if breakdowns_last_year >= 4:
            q -= 0.15
        elif breakdowns_last_year >= 2:
            q -= 0.08

        # Penalize overdue maintenance
        if maintenance_status == "OVERDUE":
            q -= 0.1
        elif maintenance_status == "DUE_SOON":
            q -= 0.05

        ambulance_quality_score = round(max(0.0, min(q, 1.0)), 2)

        record = {
            "ambulance_id": ambulance_id,
            "registration_number": registration_number,
            "city": CITY_NAME,

            "ambulance_type": amb_type,
            "fuel_type": fuel_type,
            "make_model": make_model,
            "year_of_manufacture": year_of_manufacture,
            "odometer_km": odometer_km,
            "mileage_kmpl": mileage_kmpl,

            "base_hospital_id": base_hospital_id,
            "base_lat": base_lat,
            "base_lng": base_lng,
            "current_lat": current_lat,
            "current_lng": current_lng,
            "status": status,

            # Crew
            "crew_paramedics_planned": crew_paramedics_planned,
            "crew_nurses_planned": crew_nurses_planned,
            "crew_doctor_planned": crew_doctor_planned,

            # Equipment
            "equipment_oxygen_cylinders": equipment_oxygen_cylinders,
            "equipment_stretcher_count": equipment_stretcher_count,
            "equipment_wheelchair_count": equipment_wheelchair_count,
            "equipment_trauma_kit": equipment_trauma_kit,
            "equipment_suction_machine": equipment_suction_machine,
            "equipment_spine_board": equipment_spine_board,
            "equipment_ventilator": equipment_ventilator,
            "equipment_defibrillator": equipment_defibrillator,
            "equipment_ecg_monitor": equipment_ecg_monitor,
            "equipment_neo_incubator": equipment_neo_incubator,

            "infection_control_level": infection_control_level,
            "gps_enabled": gps_enabled,
            "realtime_telemetry_enabled": realtime_telemetry_enabled,
            "onboard_camera": onboard_camera,

            # Derived capability flags
            "can_handle_critical": can_handle_critical,
            "pediatric_capable": pediatric_capable,

            # Performance
            "trips_per_day_estimate": round(trips_per_day_estimate, 2),
            "avg_response_time_min": round(avg_response_time_min, 2),
            "avg_turnaround_time_min": round(avg_turnaround_time_min, 2),
            "breakdowns_last_year": breakdowns_last_year,
            "utilisation_rate": utilisation_rate,
            "maintenance_status": maintenance_status,
            "last_maintenance_date": last_maintenance_date,

            # Quality
            "ambulance_quality_score": ambulance_quality_score,

            "last_updated": now,
        }

        ambulances.append(record)

    return ambulances


# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    ambulances = generate_ambulances(NUM_AMBULANCES)
    df = pd.DataFrame(ambulances)
    df.to_csv("ambulances_kolkata.csv", index=False)

    print(f"Generated {len(ambulances)} ambulances → ambulances_kolkata.csv")
