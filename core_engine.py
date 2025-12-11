import os
import uuid
import math
import datetime as dt
import pandas as pd
import numpy as np
from typing import TypedDict, Optional, List, Dict, Any
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================================
# 1. AMBULANCE AGENT CLASS
# ==========================================

@dataclass
class EmergencyRequest:
    request_id: str
    lat: float
    lng: float
    severity_level: str
    emergency_type: str
    is_child: bool
    timestamp: str
    zone_id: Optional[str] = None

@dataclass
class AmbulanceCandidate:
    ambulance_id: str
    distance_km: float
    eta_min: float
    score: float
    meta: Dict[str, Any]

def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between two lat/lng points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class AmbulanceAgent:
    def __init__(self, csv_path: str = "ambulances_kolkata.csv", mode: str = "fast"):
        self.csv_path = csv_path
        self.mode = mode.lower()
        self.google_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        
        # Load Data
        try:
            self.ambulances_df = pd.read_csv(csv_path)
            bool_cols = ["can_handle_critical", "pediatric_capable"]
            for c in bool_cols:
                if c in self.ambulances_df.columns and self.ambulances_df[c].dtype == object:
                    self.ambulances_df[c] = self.ambulances_df[c].astype(str).str.lower().isin(["true", "1", "yes"])
        except FileNotFoundError:
            print(f"❌ Error: {csv_path} not found.")
            self.ambulances_df = pd.DataFrame() 

    def handle_new_emergency(self, emergency: EmergencyRequest, top_k: int = 3) -> Optional[Dict[str, Any]]:
        candidates = self._find_best_ambulances(emergency, top_k=top_k)
        if not candidates:
            return None
        
        chosen = candidates[0]
        route_info = self._compute_route(chosen, emergency)
        
        return {
            "assigned_ambulance_id": chosen.ambulance_id,
            "eta_min": round(chosen.eta_min, 1),
            "route_info": route_info,
            "explanation": self._explain_choice(chosen, candidates)  # NEW!
        }

    def _explain_choice(self, chosen: AmbulanceCandidate, all_candidates: List[AmbulanceCandidate]) -> str:
        """Generate explanation for why this ambulance was chosen."""
        reasons = []
        reasons.append(f"Closest available ambulance at {chosen.distance_km:.1f}km")
        reasons.append(f"ETA: {chosen.eta_min:.1f} minutes")
        
        if len(all_candidates) > 1:
            next_best = all_candidates[1]
            time_saved = next_best.eta_min - chosen.eta_min
            if time_saved > 2:
                reasons.append(f"Saves {time_saved:.1f} minutes vs next option")
        
        return " | ".join(reasons)

    def _find_best_ambulances(self, emergency: EmergencyRequest, top_k: int = 3) -> List[AmbulanceCandidate]:
        if self.ambulances_df.empty: return []
        df = self.ambulances_df.copy()
        
        df = df[df["status"].isin(["IDLE", "AT_HOSPITAL"])]
        if df.empty: return []
        
        df["distance_km"] = df.apply(lambda r: haversine(r["current_lat"], r["current_lng"], emergency.lat, emergency.lng), axis=1)
        df["eta_min_approx"] = (df["distance_km"] / 25.0) * 60.0
        df["score"] = df.apply(lambda r: 1.0 / (r["distance_km"] + 0.1), axis=1) 
        
        df = df.sort_values(by=["score"], ascending=False).head(top_k)
        
        results = []
        for _, row in df.iterrows():
            results.append(AmbulanceCandidate(
                ambulance_id=row["ambulance_id"],
                distance_km=row["distance_km"],
                eta_min=row["eta_min_approx"],
                score=row["score"],
                meta=row.to_dict()
            ))
        return results

    def _compute_route(self, chosen: AmbulanceCandidate, emergency: EmergencyRequest):
        return {
            "from_lat": chosen.meta["current_lat"],
            "from_lng": chosen.meta["current_lng"],
            "to_lat": emergency.lat,
            "to_lng": emergency.lng,
            "estimated_eta_min": chosen.eta_min
        }

# ==========================================
# 2. HOSPITAL SCORING (FROM NOTEBOOK)
# ==========================================

def hospital_ann_score(features):
    """
    Neural Network simulation for hospital scoring.
    Features: [Normalized_Distance, ICU_Free_Norm, Staff_Norm, Trauma_Center_Score]
    """
    weights = np.array([-0.6, 0.3, 0.1, 0.9])  # Prioritizes trauma center & proximity
    bias = 0.1
    score = np.dot(features, weights) + bias
    return max(0, score)

# ==========================================
# 3. STATE & GRAPH LOGIC
# ==========================================

class AgentState(TypedDict):
    accident_location: tuple
    accident_type: str
    assigned_ambulance: str
    ambulance_eta: float
    ambulance_explanation: str  # NEW!
    selected_hospital_id: str
    selected_hospital_name: str
    hospital_explanation: str  # NEW!
    final_plan: str

# --- Load Global Resources ---
print("--- Loading Resources in Core Engine ---")
ambulance_service = AmbulanceAgent(csv_path="ambulances_kolkata.csv", mode="fast")

try:
    df_hospitals = pd.read_csv('hospitals_kolkata.csv')
    print(f"✅ Loaded {len(df_hospitals)} hospitals")
except FileNotFoundError:
    print("❌ Error: hospitals_kolkata.csv not found.")
    df_hospitals = pd.DataFrame()

# Load Vector DB
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en", encode_kwargs={"normalize_embeddings": True})
try:
    vector_db = FAISS.load_local("doctor_faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("✅ Vector DB loaded.")
except:
    print("⚠️ Vector DB not found. Doctor lookup will fail.")
    vector_db = None

# ==========================================
# 4. NODE FUNCTIONS (UPDATED FROM NOTEBOOK)
# ==========================================

def ambulance_node(state: AgentState):
    print(f"--- [Step 1] Ambulance Dispatch: {state['accident_type']} ---")
    req = EmergencyRequest(
        request_id=str(uuid.uuid4()),
        lat=state['accident_location'][0],
        lng=state['accident_location'][1],
        severity_level="CRITICAL",
        emergency_type=state['accident_type'],
        is_child=False,
        timestamp=dt.datetime.now().isoformat()
    )
    result = ambulance_service.handle_new_emergency(req)
    if result:
        return {
            "assigned_ambulance": result['assigned_ambulance_id'], 
            "ambulance_eta": result['eta_min'],
            "ambulance_explanation": result['explanation']  # NEW!
        }
    return {
        "assigned_ambulance": "NONE", 
        "ambulance_eta": 999.0,
        "ambulance_explanation": "No ambulances available"
    }

def hospital_node(state: AgentState):
    """UPDATED: Uses ICU beds, trauma center, and staffing from notebook."""
    print(f"--- [Step 2] Finding Hospital ---")
    if df_hospitals.empty:
        return {
            "selected_hospital_id": "ERR", 
            "selected_hospital_name": "No Data",
            "hospital_explanation": "Hospital database unavailable"
        }

    user_lat, user_lng = state['accident_location']
    scored_hospitals = []
    
    for _, row in df_hospitals.iterrows():
        # 1. Distance
        dist = np.sqrt((row['lat'] - user_lat)**2 + (row['lng'] - user_lng)**2)
        norm_dist = min(dist * 10, 1.0)
        
        # 2. ICU Availability
        icu_free = row['icu_beds_free'] if pd.notna(row['icu_beds_free']) else 0
        norm_icu = min(icu_free / 20, 1.0)
        
        # 3. Staffing
        staff_count = row['staff_doctors_planned'] if pd.notna(row['staff_doctors_planned']) else 0
        norm_staff = min(staff_count / 50, 1.0)
        
        # 4. Trauma Center Match
        is_trauma = 'CRASH' in state['accident_type'].upper() or 'TRAUMA' in state['accident_type'].upper()
        has_trauma_center = 1.0 if row['facility_trauma_center'] == True else 0.0
        trauma_score = has_trauma_center if is_trauma else 0.5
        
        # 5. Score
        features = np.array([norm_dist, norm_icu, norm_staff, trauma_score])
        score = hospital_ann_score(features)
        
        scored_hospitals.append({
            "id": row['hospital_id'],
            "name": row['hospital_name'], 
            "score": score,
            "dist": dist,
            "icu_free": icu_free,
            "has_trauma": has_trauma_center > 0
        })
    
    # Select Best
    best = sorted(scored_hospitals, key=lambda x: x['score'], reverse=True)[0]
    
    # Generate Explanation
    reasons = [f"Score: {best['score']:.2f}"]
    reasons.append(f"{best['dist']:.1f}km away")
    if best['icu_free'] > 5:
        reasons.append(f"{int(best['icu_free'])} ICU beds available")
    if best['has_trauma']:
        reasons.append("Has Trauma Center")
    
    return {
        "selected_hospital_id": best['id'],
        "selected_hospital_name": best['name'],
        "hospital_explanation": " | ".join(reasons)  # NEW!
    }

def doctor_node(state: AgentState):
    print(f"--- [Step 3] Assigning Doctor ---")
    if not vector_db: 
        return {"final_plan": "Error: Doctor DB not loaded."}
    
    hosp_id = state['selected_hospital_id']
    hosp_name = state['selected_hospital_name']
    
    # Retrieval
    retriever = vector_db.as_retriever(
        search_kwargs={
            "k": 3, 
            "filter": {"hospital_id": hosp_id}
        }
    )
    
    query = f"""
    Accident: {state['accident_type']}
    Find doctor with: high seniority, trauma preference, On Call=True
    """
    docs = retriever.invoke(query)
    doc_context = "\n\n".join([d.page_content for d in docs])
    
    # Generation
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API"))
    
    # --- UPDATED PROMPT: STRICT NO-CODE INSTRUCTION ---
    prompt = f"""
    System: You are an emergency dispatch AI. You do not write code. You only output raw JSON data.
    
    Context:
    - Emergency: {state['accident_type']}
    - Hospital: {hosp_name}
    - Ambulance ETA: {state['ambulance_eta']} minutes

    Candidate Doctors Available:
    {doc_context}

    Task: 
    Analyze the candidates and select the ONE best doctor for this specific emergency.

    OUTPUT RULES:
    1. Return ONLY a single valid JSON object.
    2. DO NOT write Python code, markdown blocks (```), or conversational text.
    3. The output must be parseable by json.loads().

    Required JSON Structure:
    {{
      "recommended_hospital": "{hosp_name}",
      "target_doctor": "Full Name",
      "doctor_specialty": "Specialty",
      "status": "ON_CALL",
      "action": "Specific Action (e.g. Prepare Trauma Bay)",
      "reason": "Clear, short reason for selection."
    }}
    """
    
    response = llm.invoke(prompt)
    
    # clean up potential markdown wrappers just in case
    clean_content = response.content.replace("```json", "").replace("```", "").strip()
    
    return {"final_plan": clean_content}

# ==========================================
# 5. COMPILE GRAPH
# ==========================================

def get_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("find_ambulance", ambulance_node)
    workflow.add_node("find_hospital", hospital_node)
    workflow.add_node("find_doctor", doctor_node)
    
    workflow.set_entry_point("find_ambulance")
    workflow.add_edge("find_ambulance", "find_hospital")
    workflow.add_edge("find_hospital", "find_doctor")
    workflow.add_edge("find_doctor", END)
    
    return workflow.compile()