"""
Synthetic dataset generator for:
1) Hospitals around Kolkata (with rich inventory & facilities)
2) Doctors linked to those hospitals (with detailed attributes)

Outputs:
    - hospitals_kolkata.csv
    - doctors_kolkata.csv
"""

import random
import string
from datetime import datetime
import pandas as pd

# ---------------------------
# CONFIG
# ---------------------------

NUM_HOSPITALS = 15

MIN_DOCTORS_PER_HOSPITAL = 15
MAX_DOCTORS_PER_HOSPITAL = 30

CITY_NAME = "Kolkata"

# Kolkata approx center
CITY_CENTER_LAT = 22.5726
CITY_CENTER_LNG = 88.3639
LAT_SPREAD = 0.20   # spread around Kolkata (degrees)
LNG_SPREAD = 0.20

SPECIALTIES = [
    "TRAUMA",
    "NEURO",
    "CARDIAC",
    "ORTHO",
    "PEDIATRIC",
    "GENERAL_SURGERY",
    "EMERGENCY_MEDICINE",
    "PULMONOLOGY",
    "NEPHROLOGY",
]

HOSPITAL_STATUSES = ["ACTIVE", "FULL", "OFFLINE"]
DOCTOR_MAX_CONCURRENT_CASES_RANGE = (1, 4)

BLOOD_GROUPS = ["A_pos", "A_neg", "B_pos", "B_neg", "AB_pos", "AB_neg", "O_pos", "O_neg"]

random.seed(42)  # reproducible


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def random_hospital_name():
    prefixes = ["CityCare", "LifeLine", "MediPoint", "HealthPlus", "CareTrust"]
    suffix = ''.join(random.choices(string.ascii_uppercase, k=2))
    return f"{random.choice(prefixes)} Hospital {suffix}"

def random_person_name():
    first_names = [
        "Aarav", "Vivaan", "Diya", "Ishaan", "Ananya",
        "Kabir", "Riya", "Sara", "Arjun", "Priya",
        "Rahul", "Sneha", "Kunal", "Mira", "Siddharth",
        "Rohit", "Simran", "Aditya", "Nikita", "Rohan", "Juhi"
    ]
    last_names = [
        "Sharma", "Mukherjee", "Singh", "Nair", "Das",
        "Iyer", "Mehta", "Patel", "Reddy", "Chatterjee",
        "Ghosh", "Banerjee", "Sen", "Roy", "Bhattacharya"
    ]
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def random_lat_lng():
    """Generate a random (lat, lng) around Kolkata."""
    lat = CITY_CENTER_LAT + random.uniform(-LAT_SPREAD, LAT_SPREAD)
    lng = CITY_CENTER_LNG + random.uniform(-LNG_SPREAD, LNG_SPREAD)
    return round(lat, 6), round(lng, 6)

def random_specialties():
    """Pick a random subset of specialties for a hospital."""
    k = random.randint(3, min(6, len(SPECIALTIES)))
    return sorted(random.sample(SPECIALTIES, k=k))

def generate_hospital_id(i: int) -> str:
    return f"HOSP_{i:03d}"

def generate_doctor_id(i: int) -> str:
    return f"DOC_{i:04d}"

def random_status():
    # Weighted statuses: mostly ACTIVE, some FULL, few OFFLINE
    return random.choices(
        HOSPITAL_STATUSES,
        weights=[0.7, 0.2, 0.1],
        k=1
    )[0]

def random_gender():
    return random.choice(["M", "F", "OTHER"])

def random_languages():
    base = ["English", "Hindi", "Bengali"]
    extra_pool = ["Urdu", "Odia", "Marathi", "Tamil"]
    k = random.randint(0, 2)
    langs = base + random.sample(extra_pool, k=k)
    return str(sorted(set(langs)))

def sub_specialty_for(specialty: str) -> str:
    mapping = {
        "CARDIAC": ["INTERVENTIONAL_CARDIOLOGY", "CARDIAC_ELECTROPHYSIOLOGY", "HEART_FAILURE"],
        "NEURO": ["STROKE_SPECIALIST", "EPILEPTOLOGIST", "SPINE_SURGERY"],
        "TRAUMA": ["POLYTRAUMA", "ORTHO_TRAUMA", "HEAD_INJURY_FOCUS"],
        "ORTHO": ["SPINE_SURGEON", "JOINT_REPLACEMENT", "SPORTS_INJURY"],
        "PEDIATRIC": ["NEONATOLOGY", "PEDIATRIC_ICU", "GENERAL_PEDIATRICS"],
        "PULMONOLOGY": ["CRITICAL_CARE", "SLEEP_MEDICINE", "INTERVENTIONAL_PULMONOLOGY"],
        "NEPHROLOGY": ["DIALYSIS_SPECIALIST", "TRANSPLANT_NEPRHOLOGY"],
        "GENERAL_SURGERY": ["LAPAROSCOPIC_SURGERY", "ONCO_SURGERY"],
        "EMERGENCY_MEDICINE": ["RESUSCITATION_EXPERT", "TOXICOLOGY"]
    }
    if specialty in mapping:
        return random.choice(mapping[specialty])
    return "GENERAL"

def random_shift_type():
    return random.choice(["DAY", "NIGHT", "ROTATING"])

def random_primary_qualification():
    return random.choice([
        "MBBS",
        "MBBS, MD",
        "MBBS, MS",
        "MBBS, DNB",
    ])

def random_additional_qualification():
    pool = [
        "",
        "FNB Critical Care",
        "Fellowship in Trauma Care",
        "Fellowship in Interventional Cardiology",
        "Fellowship in Neuro Critical Care",
        "Fellowship in Pediatric ICU",
    ]
    return random.choice(pool)


# ---------------------------
# GENERATE HOSPITAL DATA
# ---------------------------

def generate_hospitals(num_hospitals: int):
    hospitals = []
    now = datetime.utcnow().isoformat()

    for i in range(1, num_hospitals + 1):
        hospital_id = generate_hospital_id(i)
        name = random_hospital_name()
        lat, lng = random_lat_lng()

        # --- Core capacity ---
        has_icu = random.choice([True, True, True, False])  # mostly True
        icu_beds_total = random.randint(10, 60) if has_icu else 0
        icu_beds_free = random.randint(0, icu_beds_total) if has_icu else 0

        ed_capacity = random.randint(20, 100)
        current_ed_load = random.randint(0, ed_capacity)

        specialties = random_specialties()

        rating_priority = round(random.uniform(0.5, 1.0), 2)
        status = random_status()

        # --- Staff ---
        planned_doctors = random.randint(
            MIN_DOCTORS_PER_HOSPITAL,
            MAX_DOCTORS_PER_HOSPITAL
        )
        staff_nurses_count = planned_doctors * random.randint(2, 4)
        staff_paramedics_count = random.randint(15, 50)
        staff_support_count = random.randint(30, 100)

        # --------------------
        #   INVENTORY (RICH)
        # --------------------

        # Ventilators
        inventory_ventilators_icu = random.randint(0, 25) if has_icu else random.randint(0, 5)
        inventory_ventilators_transport = random.randint(0, 10)

        # Defibrillators
        inventory_defibrillators = random.randint(2, 20)

        # ECMO (only in some high-end hospitals)
        inventory_ecmo_machines = random.choice([0, 0, 1, 2])  # mostly 0, sometimes 1–2

        # Multi-parameter monitors in ED/ICU
        inventory_advanced_monitors = random.randint(10, 50) if has_icu else random.randint(5, 25)

        # Point-of-care devices (ABG machines, portable analyzers)
        inventory_poc_devices = random.randint(1, 10)

        # Operating theatres
        inventory_ot_rooms = random.randint(2, 12)

        # Blood bank details
        inventory_blood_bank_available = random.choice([True, True, True, False])

        # Group-wise blood units (RBC-equivalent units)
        blood_group_units = {}
        if inventory_blood_bank_available:
            # Assign each group a stock range; Rh- usually rarer
            for bg in BLOOD_GROUPS:
                if "neg" in bg:
                    blood_group_units[bg] = random.randint(5, 40)
                else:
                    blood_group_units[bg] = random.randint(20, 120)
        else:
            for bg in BLOOD_GROUPS:
                blood_group_units[bg] = 0

        # Total units across all groups
        inventory_blood_units_total = sum(blood_group_units.values())

        # Some plasma & platelets (overall pool, not group-wise for now)
        if inventory_blood_bank_available:
            inventory_blood_units_plasma = random.randint(30, 250)
            inventory_blood_units_platelets = random.randint(20, 150)
        else:
            inventory_blood_units_plasma = 0
            inventory_blood_units_platelets = 0

        inventory_ambulances_linked = random.randint(0, 6)

        # --------------------
        #   FACILITIES
        # --------------------

        facility_trauma_center = "TRAUMA" in specialties or random.choice([True, False])
        facility_cardiac_cath_lab = "CARDIAC" in specialties and random.choice([True, True, False])
        facility_neonatal_icu = random.choice([True, False])
        facility_dialysis_unit = random.choice([True, True, False])
        facility_emergency_24x7 = True  # assume all in the system are 24x7 emergency
        facility_pharmacy_24x7 = random.choice([True, True, False])

        # --------------------
        #   DIAGNOSTICS
        # --------------------

        diagnostics_has_ct = random.choice([True, True, True, False])
        diagnostics_has_mri = random.choice([True, False, False, False])
        diagnostics_has_ultrasound = True
        diagnostics_path_lab_level = random.choice(["BASIC", "ADVANCED", "FULL_SERVICE"])

        # Diagnostics quality score (0–1)
        diag_score = 0.4
        if diagnostics_has_ct:
            diag_score += 0.2
        if diagnostics_has_mri:
            diag_score += 0.15
        if diagnostics_path_lab_level == "ADVANCED":
            diag_score += 0.1
        elif diagnostics_path_lab_level == "FULL_SERVICE":
            diag_score += 0.2

        # Slight bonus if ECMO or cath lab (signals advanced centre)
        if inventory_ecmo_machines > 0 or facility_cardiac_cath_lab:
            diag_score += 0.05

        diagnostics_quality_score = round(min(diag_score, 1.0), 2)

        hospital_record = {
            "hospital_id": hospital_id,
            "hospital_name": name,
            "city": CITY_NAME,
            "lat": lat,
            "lng": lng,

            # Core capacity
            "has_icu": has_icu,
            "icu_beds_total": icu_beds_total,
            "icu_beds_free": icu_beds_free,
            "ed_capacity": ed_capacity,
            "current_ed_load": current_ed_load,

            # Specialties
            "specialties": str(specialties),

            # Staff
            "staff_doctors_planned": planned_doctors,
            "staff_nurses_count": staff_nurses_count,
            "staff_paramedics_count": staff_paramedics_count,
            "staff_support_count": staff_support_count,

            # INVENTORY - equipment
            "inventory_ventilators_icu": inventory_ventilators_icu,
            "inventory_ventilators_transport": inventory_ventilators_transport,
            "inventory_defibrillators": inventory_defibrillators,
            "inventory_ecmo_machines": inventory_ecmo_machines,
            "inventory_advanced_monitors": inventory_advanced_monitors,
            "inventory_poc_devices": inventory_poc_devices,
            "inventory_ot_rooms": inventory_ot_rooms,
            "inventory_ambulances_linked": inventory_ambulances_linked,

            # Blood bank as facility + quantitative stock
            "inventory_blood_bank_available": inventory_blood_bank_available,
            "inventory_blood_units_total": inventory_blood_units_total,
            "inventory_blood_units_plasma": inventory_blood_units_plasma,
            "inventory_blood_units_platelets": inventory_blood_units_platelets,

            # Facilities
            "facility_trauma_center": facility_trauma_center,
            "facility_cardiac_cath_lab": facility_cardiac_cath_lab,
            "facility_neonatal_icu": facility_neonatal_icu,
            "facility_dialysis_unit": facility_dialysis_unit,
            "facility_emergency_24x7": facility_emergency_24x7,
            "facility_pharmacy_24x7": facility_pharmacy_24x7,

            # Diagnostics
            "diagnostics_has_ct": diagnostics_has_ct,
            "diagnostics_has_mri": diagnostics_has_mri,
            "diagnostics_has_ultrasound": diagnostics_has_ultrasound,
            "diagnostics_path_lab_level": diagnostics_path_lab_level,
            "diagnostics_quality_score": diagnostics_quality_score,

            # Overall rating / priority (can be used in hospital agent)
            "rating_priority": rating_priority,
            "status": status,
            "last_updated": now,
        }

        # Add blood-group-wise columns
        for bg in BLOOD_GROUPS:
            col_name = f"inventory_blood_{bg}"
            hospital_record[col_name] = blood_group_units[bg]

        hospitals.append(hospital_record)

    return hospitals

# ---------------------------
# GENERATE DOCTOR DATA (DETAILED)
# ---------------------------

def generate_doctors(hospitals):
    doctors = []
    doc_counter = 1

    for hosp in hospitals:
        hospital_id = hosp["hospital_id"]
        hospital_specialties = eval(hosp["specialties"])  # convert string back to list
        planned_doctors = hosp["staff_doctors_planned"]
        hospital_diag_score = hosp["diagnostics_quality_score"]
        hospital_rating = hosp["rating_priority"]

        num_doctors = planned_doctors  # keep consistent with staff field

        for _ in range(num_doctors):
            doctor_id = generate_doctor_id(doc_counter)
            doc_counter += 1

            name = random_person_name()
            gender = random_gender()
            age = random.randint(28, 65)

            specialty = random.choice(hospital_specialties)
            sub_specialty = sub_specialty_for(specialty)

            primary_qualification = random_primary_qualification()
            additional_qualification = random_additional_qualification()

            on_call = random.choice([True, True, False])  # mostly on call
            shift_type = random_shift_type()

            # Experience & seniority
            years_experience = random.randint(1, 35)
            if years_experience <= 5:
                seniority_level = "JUNIOR_RESIDENT"
            elif years_experience <= 12:
                seniority_level = "CONSULTANT"
            else:
                seniority_level = "SENIOR_CONSULTANT"

            board_certified = random.choice([True, True, True, False])

            # Work pattern / preferences
            prefers_trauma_cases = (specialty in ["TRAUMA", "EMERGENCY_MEDICINE"]) and random.choice([True, True, False])
            telemedicine_enabled = random.choice([True, True, False])
            languages_spoken = random_languages()

            max_concurrent_cases = random.randint(
                DOCTOR_MAX_CONCURRENT_CASES_RANGE[0],
                DOCTOR_MAX_CONCURRENT_CASES_RANGE[1]
            )

            # Behavioral/performance stats (for simulation/agent decisions)
            avg_consultation_time_min = random.randint(10, 30)
            avg_response_time_to_call_min = random.uniform(1.0, 10.0)
            last_appraisal_score = round(random.uniform(3.0, 5.0), 2)  # 3–5 rating

            contact_channel = f"app://doctor/{doctor_id}"

            # Doctor quality score (0–1)
            # Base on years, seniority, board certification, hospital-level quality, appraisal, responsiveness
            q = 0.2
            if years_experience > 5:
                q += 0.2
            if years_experience > 10:
                q += 0.1
            if seniority_level == "SENIOR_CONSULTANT":
                q += 0.1
            if board_certified:
                q += 0.15

            # Good appraisal and decent response time
            q += 0.05 * (last_appraisal_score - 3.0)  # +0 to +0.1
            if avg_response_time_to_call_min <= 5:
                q += 0.05

            # slight influence from hospital rating & diagnostics
            q += 0.07 * hospital_rating
            q += 0.03 * hospital_diag_score

            doctor_quality_score = round(min(q, 1.0), 2)

            # Burnout risk: rough inverse of quality + workload
            base_burnout = random.uniform(0.1, 0.5)
            if shift_type == "ROTATING":
                base_burnout += 0.1
            if max_concurrent_cases >= 3:
                base_burnout += 0.1
            burnout_risk_score = round(min(base_burnout + (1.0 - doctor_quality_score) * 0.3, 1.0), 2)

            doctors.append({
                "doctor_id": doctor_id,
                "hospital_id": hospital_id,
                "name": name,
                "gender": gender,
                "age": age,

                "specialty": specialty,
                "sub_specialty": sub_specialty,

                "primary_qualification": primary_qualification,
                "additional_qualification": additional_qualification,

                "years_experience": years_experience,
                "seniority_level": seniority_level,
                "board_certified": board_certified,

                "on_call": on_call,
                "shift_type": shift_type,
                "prefers_trauma_cases": prefers_trauma_cases,
                "telemedicine_enabled": telemedicine_enabled,
                "languages_spoken": languages_spoken,

                "max_concurrent_cases": max_concurrent_cases,
                "avg_consultation_time_min": avg_consultation_time_min,
                "avg_response_time_to_call_min": round(avg_response_time_to_call_min, 2),
                "last_appraisal_score": last_appraisal_score,

                "doctor_quality_score": doctor_quality_score,
                "burnout_risk_score": burnout_risk_score,

                "contact_channel": contact_channel,
            })

    return doctors

# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    hospitals = generate_hospitals(NUM_HOSPITALS)
    doctors = generate_doctors(hospitals)

    hospitals_df = pd.DataFrame(hospitals)
    doctors_df = pd.DataFrame(doctors)

    hospitals_df.to_csv("hospitals_kolkata.csv", index=False)
    doctors_df.to_csv("doctors_kolkata.csv", index=False)

    print("Generated:")
    print(f"  - {len(hospitals)} hospitals → hospitals_kolkata.csv")
    print(f"  - {len(doctors)} doctors   → doctors_kolkata.csv")
