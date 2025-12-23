"""
Ambulance Agent for SmartAgent-ER

Features:
- Loads ambulances from ambulances_kolkata.csv
- Uses .env (python-dotenv) to load GOOGLE_MAPS_API_KEY
- Modes:
    * fast  -> ETA based on distance + fixed speed (no Google)
    * full  -> for top 5 nearest ambulances, uses Google Distance Matrix
              to refine ETA; final chosen ambulance uses Google Directions
- Two main use-cases:
    1) handle_new_emergency(...)   -> allocate ambulance for a specific case
    2) suggest_reallocation_based_on_risk(...)  -> pre-position ambulances
       to high-risk zones based on risk_profiles_kolkata.csv

Make sure you have:
    - ambulances_kolkata.csv
    - risk_profiles_kolkata.csv
    - zones_kolkata.csv
    - .env with GOOGLE_MAPS_API_KEY=your_key_here
"""

import math
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


# ---------------------------
# UTILS
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


# ---------------------------
# DATA MODELS
# ---------------------------

@dataclass
class EmergencyRequest:
    """
    Represents a new emergency raised by a citizen / system.
    """
    request_id: str
    lat: float
    lng: float
    severity_level: str         # "LOW", "MED", "HIGH", "CRITICAL"
    emergency_type: str         # "TRAUMA", "CARDIAC", "RESPIRATORY", etc.
    is_child: bool              # True if pediatric case
    timestamp: str              # ISO string
    zone_id: Optional[str] = None


@dataclass
class AmbulanceCandidate:
    """
    Wrapper for a scored ambulance option.
    """
    ambulance_id: str
    distance_km: float
    eta_min: float
    score: float
    meta: Dict[str, Any]


# ---------------------------
# AMBULANCE AGENT
# ---------------------------

class AmbulanceAgent:
    def __init__(
        self,
        csv_path: str = "ambulances_kolkata.csv",
        google_api_key: Optional[str] = None,
        mode: str = "fast",            # "fast" or "full"
        max_eta_for_norm: float = 45.0 # mins, for eta normalization
    ):
        """
        mode="fast":
            - uses haversine distance + approximate ETA (no Google calls)
        mode="full":
            - approximate ETA first, then for top 5 ambulances uses
              Google Distance Matrix to get live ETA & recomputes scores.
        """
        self.csv_path = csv_path
        self.mode = mode.lower()
        if self.mode not in ("fast", "full"):
            self.mode = "fast"

        self.max_eta_for_norm = max_eta_for_norm

        self.ambulances_df = pd.read_csv(csv_path)

        self.google_api_key = google_api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.google_api_key:
            print("[WARN] No GOOGLE_MAPS_API_KEY set. Google-based routing/ETA will fall back to haversine.")

        # Ensure required columns exist
        required_cols = [
            "ambulance_id", "current_lat", "current_lng", "status",
            "ambulance_type", "can_handle_critical", "pediatric_capable",
            "ambulance_quality_score"
        ]
        for c in required_cols:
            if c not in self.ambulances_df.columns:
                raise ValueError(f"Missing column '{c}' in {csv_path}")

        # Normalize booleans if read as strings
        bool_cols = ["can_handle_critical", "pediatric_capable"]
        for c in bool_cols:
            if self.ambulances_df[c].dtype == object:
                self.ambulances_df[c] = self.ambulances_df[c].astype(str).str.lower().isin(["true", "1", "yes"])

    # -----------------------
    # SMALL HELPER: SAFE ETA
    # -----------------------

    def _get_eta_for_row(self, row) -> float:
        """
        Return ETA in minutes using live ETA if available and not NaN,
        otherwise fall back to approximate ETA.
        This prevents eta_min from becoming NaN.
        """
        eta_live = row.get("eta_min_live", None)
        if eta_live is not None and not pd.isna(eta_live):
            return float(eta_live)
        return float(row["eta_min_approx"])

    # -----------------------
    # PUBLIC API – CASE 1: EMERGENCY
    # -----------------------

    def handle_new_emergency(self, emergency: EmergencyRequest, top_k: int = 3) -> Optional[Dict[str, Any]]:
        """
        Use-case 1: real-time emergency allocation.

        - Filter + score ambulances (fast/full mode)
        - Simulate acceptance
        - Return chosen ambulance + routing info
        """
        candidates = self._find_best_ambulances(emergency, top_k=top_k)
        if not candidates:
            print("No suitable ambulances found for this emergency.")
            return None

        chosen = self._simulate_ambulance_acceptance(candidates)
        if chosen is None:
            print("No ambulance accepted the request.")
            return None

        route_info = self._compute_route(chosen, emergency)

        result = {
            "case_type": "EMERGENCY_ALLOCATION",
            "request_id": emergency.request_id,
            "assigned_ambulance_id": chosen.ambulance_id,
            "distance_km": round(chosen.distance_km, 2),
            "eta_min": round(chosen.eta_min, 1),
            "score": round(chosen.score, 3),
            "route_info": route_info,
            "ambulance_meta": chosen.meta,
        }
        return result

    # -----------------------
    # PUBLIC API – CASE 2: RISK-BASED REALLOCATION
    # -----------------------

    def suggest_reallocation_based_on_risk(
        self,
        risk_csv_path: str = "risk_profiles_kolkata.csv",
        zones_csv_path: str = "zones_kolkata.csv",
        day_type: Optional[str] = "WEEKDAY",
        time_window: Optional[str] = "18:00-24:00",
        top_zones: int = 3,
        ambulances_per_zone: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Use-case 2: proactive pre-positioning of ambulances based on city risk.

        - Loads risk_profiles_kolkata.csv and zones_kolkata.csv
        - Picks top 'top_zones' zones by risk_score (for given day_type & time_window)
        - For each zone, finds nearest IDLE ambulances
        - Suggests moving them to zone centers
        """
        if not os.path.exists(risk_csv_path) or not os.path.exists(zones_csv_path):
            print("[WARN] Risk or zones CSV not found. Cannot do risk-based reallocation.")
            return []

        risk_df = pd.read_csv(risk_csv_path)
        zones_df = pd.read_csv(zones_csv_path)

        # Filter by day_type and time_window if provided
        df = risk_df.copy()
        if day_type is not None:
            df = df[df["day_type"] == day_type]
        if time_window is not None:
            df = df[df["time_window"] == time_window]

        if df.empty:
            print("[WARN] No risk profile rows match given day_type/time_window.")
            return []

        # Pick top risky zones
        df = df.sort_values(by="risk_score", ascending=False)
        df_top_zones = df.head(top_zones)

        # Map zone -> (lat, lng)
        zone_centers = zones_df.set_index("zone_id")[["center_lat", "center_lng"]].to_dict("index")

        # Work on ambulances: only IDLE, in Kolkata
        amb_df = self.ambulances_df.copy()
        if "city" in amb_df.columns:
            amb_df = amb_df[amb_df["city"] == "Kolkata"]
        amb_df = amb_df[amb_df["status"] == "IDLE"]
        if amb_df.empty:
            print("[WARN] No IDLE ambulances available for reallocation.")
            return []

        suggestions = []
        used_ambulances = set()

        for _, zrow in df_top_zones.iterrows():
            zone_id = zrow["zone_id"]
            zone_risk = zrow["risk_score"]
            center = zone_centers.get(zone_id)
            if center is None:
                continue
            z_lat = float(center["center_lat"])
            z_lng = float(center["center_lng"])

            # For this zone, compute distance from each unused ambulance
            amb_df_zone = amb_df[~amb_df["ambulance_id"].isin(used_ambulances)].copy()
            if amb_df_zone.empty:
                break

            distances = []
            for _, arow in amb_df_zone.iterrows():
                d = haversine(
                    float(arow["current_lat"]),
                    float(arow["current_lng"]),
                    z_lat,
                    z_lng
                )
                distances.append(d)
            amb_df_zone["distance_km"] = distances

            amb_df_zone = amb_df_zone.sort_values(by="distance_km", ascending=True)
            amb_df_zone = amb_df_zone.head(ambulances_per_zone)

            for _, arow in amb_df_zone.iterrows():
                suggestions.append({
                    "case_type": "RISK_BASED_REALLOCATION",
                    "zone_id": zone_id,
                    "zone_risk_score": round(float(zone_risk), 3),
                    "to_lat": z_lat,
                    "to_lng": z_lng,
                    "ambulance_id": arow["ambulance_id"],
                    "from_lat": float(arow["current_lat"]),
                    "from_lng": float(arow["current_lng"]),
                    "distance_km": round(float(arow["distance_km"]), 2),
                })
                used_ambulances.add(arow["ambulance_id"])

        return suggestions

    # -----------------------
    # INTERNAL: FILTER & SCORE FOR EMERGENCY
    # -----------------------

    def _find_best_ambulances(self, emergency: EmergencyRequest, top_k: int = 3) -> List[AmbulanceCandidate]:
        """
        Filter and rank ambulances based on:
        - availability (IDLE or AT_HOSPITAL)
        - distance to incident
        - ETA (approx or live)
        - capability match (critical / pediatric / type)
        - quality score
        """
        df = self.ambulances_df.copy()

        # 1) Filter by city if available
        if "city" in df.columns:
            df = df[df["city"] == "Kolkata"]

        # 2) Filter by status (only ambulances that can be dispatched)
        preferred_status = ["IDLE", "AT_HOSPITAL"]
        df = df[df["status"].isin(preferred_status)]
        if df.empty:
            return []

        # 3) Compute distance to incident
        distances = []
        for _, row in df.iterrows():
            d = haversine(
                float(row["current_lat"]),
                float(row["current_lng"]),
                emergency.lat,
                emergency.lng
            )
            distances.append(d)
        df["distance_km"] = distances

        # 4) Approximate ETA (fast estimate)
        df["eta_min_approx"] = df["distance_km"] / 25.0 * 60.0  # 25 km/h

        # 5) If in full mode and we have Google API → refine ETA using Distance Matrix for top N
        if self.mode == "full" and self.google_api_key:
            try:
                df = self._update_eta_with_distance_matrix(df, emergency, top_n=5)
            except Exception as e:
                print(f"[WARN] Distance Matrix update failed, using approx ETA only. Error: {e}")

        # 6) Capability matching features
        df["critical_match"] = df["can_handle_critical"].astype(int)
        df["pediatric_match"] = df["pediatric_capable"].astype(int)

        df["type_match"] = df.apply(
            lambda r: self._type_match_score(emergency.emergency_type, str(r["ambulance_type"])),
            axis=1
        )

        # 7) Score each ambulance
        df["score"] = df.apply(
            lambda r: self._score_ambulance_row(r, emergency),
            axis=1
        )

        # 8) Sort by score (desc) and ETA (asc), then distance
        df = df.sort_values(by=["score", "eta_min_approx", "distance_km"], ascending=[False, True, True])

        # Keep top_k
        df_top = df.head(top_k)

        candidates: List[AmbulanceCandidate] = []
        for _, row in df_top.iterrows():
            eta_used = self._get_eta_for_row(row)
            meta = row.to_dict()
            candidates.append(
                AmbulanceCandidate(
                    ambulance_id=row["ambulance_id"],
                    distance_km=row["distance_km"],
                    eta_min=eta_used,
                    score=row["score"],
                    meta=meta,
                )
            )
        return candidates

    def _type_match_score(self, emergency_type: str, ambulance_type: str) -> float:
        """
        Simple mapping: boost score when ambulance type fits emergency type.
        """
        emergency_type = emergency_type.upper()
        ambulance_type = ambulance_type.upper()

        # Basic mapping table
        if emergency_type in ["TRAUMA", "ROAD_ACCIDENT"]:
            if "TRAUMA" in ambulance_type:
                return 1.0
            if "ALS" in ambulance_type:
                return 0.8
            return 0.5

        if emergency_type in ["CARDIAC", "CARDIAC_EMERGENCY", "CHEST_PAIN"]:
            if "CARDIAC" in ambulance_type:
                return 1.0
            if "ALS" in ambulance_type:
                return 0.8
            return 0.4

        if emergency_type in ["RESPIRATORY", "RESPIRATORY_DISTRESS"]:
            if "ALS" in ambulance_type:
                return 0.9
            return 0.5

        # Default
        return 0.6

    def _score_ambulance_row(self, row, emergency: EmergencyRequest) -> float:
        """
        Combine aspects into a single score:

        Factors:
        - ETA (smaller is better) → normalized
        - ambulance_quality_score
        - type_match
        - critical match (if severity HIGH/CRITICAL)
        - pediatric match (if is_child)
        """
        eta = self._get_eta_for_row(row)
        eta_clamped = min(eta, self.max_eta_for_norm)
        eta_norm = 1.0 - (eta_clamped / self.max_eta_for_norm)  # 1 = fast, 0 = slow

        quality = float(row.get("ambulance_quality_score", 0.5))
        type_match = float(row["type_match"])
        critical_match = int(row["critical_match"])
        pediatric_match = int(row["pediatric_match"])

        sev = emergency.severity_level.upper()
        needs_critical = sev in ["HIGH", "CRITICAL"]

        score = 0.0

        score += 0.30 * eta_norm
        score += 0.25 * quality
        score += 0.20 * type_match

        if needs_critical:
            score += 0.20 * critical_match
        if emergency.is_child:
            score += 0.05 * pediatric_match

        return max(0.0, min(1.5, score))

    # -----------------------
    # GOOGLE DISTANCE MATRIX FOR TOP-N ETA (FULL MODE)
    # -----------------------

    def _update_eta_with_distance_matrix(self, df: pd.DataFrame, emergency: EmergencyRequest, top_n: int = 5) -> pd.DataFrame:
        """
        Use Google Distance Matrix to get live ETA (with traffic) for top_n
        nearest ambulances.
        - Updates df with columns: 'eta_min_live', 'eta_source'
        """
        if self.google_api_key is None:
            return df

        # pick top_n by distance
        df_sorted = df.sort_values(by="distance_km", ascending=True)
        df_top = df_sorted.head(top_n).copy()
        if df_top.empty:
            return df

        origins = [
            f"{row['current_lat']},{row['current_lng']}" for _, row in df_top.iterrows()
        ]
        destination = f"{emergency.lat},{emergency.lng}"

        url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        params = {
            "origins": "|".join(origins),
            "destinations": destination,
            "departure_time": "now",
            "key": self.google_api_key,
        }

        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()

        if data.get("status") != "OK":
            raise RuntimeError(f"Distance Matrix error: {data.get('status')} - {data.get('error_message')}")

        rows = data.get("rows", [])
        if not rows:
            return df

        eta_list = []
        for row in rows:
            elements = row.get("elements", [])
            if not elements:
                eta_list.append(None)
                continue
            el = elements[0]
            if el.get("status") != "OK":
                eta_list.append(None)
                continue
            duration_traffic = el.get("duration_in_traffic") or el.get("duration")
            if not duration_traffic:
                eta_list.append(None)
                continue
            eta_sec = duration_traffic["value"]
            eta_min = eta_sec / 60.0
            eta_list.append(eta_min)

        df_top["eta_min_live"] = eta_list
        df_top["eta_source"] = "google_distance_matrix"

        # merge back
        df = df.merge(
            df_top[["ambulance_id", "eta_min_live", "eta_source"]],
            on="ambulance_id",
            how="left",
        )

        return df

    # -----------------------
    # ACCEPTANCE & ROUTE
    # -----------------------

    def _simulate_ambulance_acceptance(self, candidates: List[AmbulanceCandidate]) -> Optional[AmbulanceCandidate]:
        """
        For now: assume top candidate usually accepts.
        """
        if not candidates:
            return None

        import random
        for idx, c in enumerate(candidates):
            base_prob = 0.9 if idx == 0 else 0.6 if idx == 1 else 0.4
            if c.meta.get("maintenance_status") == "OVERDUE":
                base_prob -= 0.2
            if base_prob < 0.1:
                base_prob = 0.1

            if random.random() <= base_prob:
                return c

        return None

    def _compute_route(self, chosen: AmbulanceCandidate, emergency: EmergencyRequest) -> Dict[str, Any]:
        """
        Compute routing using Google Directions API if possible.
        Falls back to haversine-based ETA if no API or error.
        """
        row = chosen.meta
        lat1, lng1 = float(row["current_lat"]), float(row["current_lng"])
        lat2, lng2 = emergency.lat, emergency.lng

        if not self.google_api_key:
            return self._compute_route_haversine_fallback(lat1, lng1, lat2, lng2)

        try:
            return self._compute_route_google(lat1, lng1, lat2, lng2)
        except Exception as e:
            print(f"[WARN] Google Directions failed, using haversine fallback. Error: {e}")
            return self._compute_route_haversine_fallback(lat1, lng1, lat2, lng2)

    def _compute_route_haversine_fallback(self, lat1, lng1, lat2, lng2) -> Dict[str, Any]:
        dist_km = haversine(lat1, lng1, lat2, lng2)
        avg_speed_kmph = 25.0
        eta_min = (dist_km / avg_speed_kmph) * 60.0 if avg_speed_kmph > 0 else None

        return {
            "from_lat": lat1,
            "from_lng": lng1,
            "to_lat": lat2,
            "to_lng": lng2,
            "estimated_distance_km": round(dist_km, 2),
            "estimated_eta_min": round(eta_min, 1) if eta_min is not None else None,
            "routing_provider": "haversine_stub",
            "polyline": None,
        }

    def _compute_route_google(self, lat1, lng1, lat2, lng2) -> Dict[str, Any]:
        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            "origin": f"{lat1},{lng1}",
            "destination": f"{lat2},{lng2}",
            "departure_time": "now",
            "key": self.google_api_key,
        }

        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()

        if data.get("status") != "OK":
            raise RuntimeError(f"Google Directions error: {data.get('status')} - {data.get('error_message')}")

        route = data["routes"][0]
        leg = route["legs"][0]

        distance_m = leg["distance"]["value"]
        duration_s = leg["duration"]["value"]
        duration_traffic_s = leg.get("duration_in_traffic", {}).get("value", duration_s)

        dist_km = distance_m / 1000.0
        eta_min = duration_traffic_s / 60.0

        polyline = route.get("overview_polyline", {}).get("points")

        return {
            "from_lat": lat1,
            "from_lng": lng1,
            "to_lat": lat2,
            "to_lng": lng2,
            "estimated_distance_km": round(dist_km, 2),
            "estimated_eta_min": round(eta_min, 1),
            "routing_provider": "google_directions",
            "polyline": polyline,
        }


# ---------------------------
# DEMO – RUN BOTH CASES
# ---------------------------

if __name__ == "__main__":
    import datetime as dt

    agent = AmbulanceAgent(
        csv_path="ambulances_kolkata.csv",
        mode="full",  # "fast" or "full"
        google_api_key=os.getenv("GOOGLE_MAPS_API_KEY")
    )

    # ---------- CASE 1: EMERGENCY ALLOCATION ----------
    emergency = EmergencyRequest(
        request_id=str(uuid.uuid4()),
        lat=22.5726,          # somewhere in central Kolkata
        lng=88.3639,
        severity_level="CRITICAL",
        emergency_type="TRAUMA",
        is_child=False,
        timestamp=dt.datetime.utcnow().isoformat(),
    )

    emergency_result = agent.handle_new_emergency(emergency, top_k=3)
    print("\n=== CASE 1: EMERGENCY ALLOCATION RESULT ===")
    print(emergency_result)

    # ---------- CASE 2: RISK-BASED REALLOCATION ----------
    suggestions = agent.suggest_reallocation_based_on_risk(
        risk_csv_path="risk_profiles_kolkata.csv",
        zones_csv_path="zones_kolkata.csv",
        day_type="WEEKDAY",
        time_window="18:00-24:00",
        top_zones=3,
        ambulances_per_zone=2,
    )
    print("\n=== CASE 2: RISK-BASED REALLOCATION SUGGESTIONS ===")
    for s in suggestions:
        print(s)
