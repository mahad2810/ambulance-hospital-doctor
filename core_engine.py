import os
import uuid
import math
import datetime as dt
import pandas as pd
import numpy as np
from typing import TypedDict, Optional, List, Dict, Any
import json

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Import the enhanced ambulance agent
from ambulance_agent import AmbulanceAgent, EmergencyRequest

# Load environment variables
load_dotenv()

# ==========================================
# 1. UTILITY FUNCTIONS
# ==========================================

def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between two lat/lng points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c
# ==========================================
# 2. HOSPITAL SCORING
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
    request_id: str
    accident_location: tuple
    accident_type: str
    severity_level: str
    is_child: bool
    
    # Ambulance info
    assigned_ambulance: str
    ambulance_eta: float
    ambulance_distance: float
    ambulance_explanation: str
    ambulance_route: dict
    ambulance_meta: dict
    
    # Hospital info
    selected_hospital_id: str
    selected_hospital_name: str
    selected_hospital_lat: float
    selected_hospital_lng: float
    hospital_distance: float
    hospital_explanation: str
    hospital_icu_beds: int
    hospital_specialties: str
    
    # Doctor info
    selected_doctor_name: str
    selected_doctor_specialty: str
    doctor_explanation: str
    
    # Final outputs
    final_plan: str
    complete_summary: dict

# --- Load Global Resources ---
print("--- Loading Resources in Core Engine ---")
ambulance_service = AmbulanceAgent(
    csv_path="ambulances_kolkata.csv", 
    mode=os.getenv("AMBULANCE_MODE", "fast")  # "fast" or "full" from .env
)

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
    print(f"--- [Step 1] Ambulance Dispatch: {state['accident_type']} | Severity: {state['severity_level']} ---")
    
    req = EmergencyRequest(
        request_id=state['request_id'],
        lat=state['accident_location'][0],
        lng=state['accident_location'][1],
        severity_level=state['severity_level'],
        emergency_type=state['accident_type'],
        is_child=state.get('is_child', False),
        timestamp=dt.datetime.utcnow().isoformat()
    )
    
    result = ambulance_service.handle_new_emergency(req, top_k=3)
    
    if result:
        # Build detailed explanation
        amb_meta = result['ambulance_meta']
        amb_type = amb_meta.get('ambulance_type', 'Unknown')
        amb_quality = amb_meta.get('ambulance_quality_score', 0)
        
        explanation = f"{amb_type} ambulance | {result['distance_km']}km away | ETA: {result['eta_min']} min | Quality Score: {amb_quality:.2f}"
        
        if amb_meta.get('can_handle_critical'):
            explanation += " | Critical Care Capable"
        if amb_meta.get('pediatric_capable'):
            explanation += " | Pediatric Capable"
        
        return {
            "assigned_ambulance": result['assigned_ambulance_id'], 
            "ambulance_eta": result['eta_min'],
            "ambulance_distance": result['distance_km'],
            "ambulance_explanation": explanation,
            "ambulance_route": result['route_info'],
            "ambulance_meta": result['ambulance_meta']
        }
    
    return {
        "assigned_ambulance": "NONE", 
        "ambulance_eta": 999.0,
        "ambulance_distance": 999.0,
        "ambulance_explanation": "❌ No ambulances available",
        "ambulance_route": {},
        "ambulance_meta": {}
    }

def hospital_node(state: AgentState):
    """Enhanced hospital selection with proper distance calculation and detailed scoring."""
    print(f"--- [Step 2] Finding Best Hospital ---")
    
    if df_hospitals.empty:
        return {
            "selected_hospital_id": "ERR", 
            "selected_hospital_name": "No Data",
            "selected_hospital_lat": 0.0,
            "selected_hospital_lng": 0.0,
            "hospital_distance": 0.0,
            "hospital_explanation": "❌ Hospital database unavailable",
            "hospital_icu_beds": 0,
            "hospital_specialties": "Unknown"
        }

    user_lat, user_lng = state['accident_location']
    scored_hospitals = []
    
    for _, row in df_hospitals.iterrows():
        # 1. Distance (using haversine for proper geospatial distance)
        dist_km = haversine(row['lat'], row['lng'], user_lat, user_lng)
        norm_dist = 1.0 - min(dist_km / 20.0, 1.0)  # Normalize: closer = better
        
        # 2. ICU Availability
        icu_free = row['icu_beds_free'] if pd.notna(row['icu_beds_free']) else 0
        norm_icu = min(icu_free / 20, 1.0)
        
        # 3. Staffing
        staff_count = row['staff_doctors_planned'] if pd.notna(row['staff_doctors_planned']) else 0
        norm_staff = min(staff_count / 50, 1.0)
        
        # 4. Specialty Match (Trauma, Cardiac, etc.)
        is_trauma = 'CRASH' in state['accident_type'].upper() or 'TRAUMA' in state['accident_type'].upper() or 'ACCIDENT' in state['accident_type'].upper()
        is_cardiac = 'CARDIAC' in state['accident_type'].upper() or 'HEART' in state['accident_type'].upper()
        
        has_trauma_center = 1.0 if row['facility_trauma_center'] == True else 0.0
        has_cardiac_lab = 1.0 if row.get('facility_cardiac_cath_lab', False) == True else 0.0
        
        specialty_score = 0.5
        if is_trauma and has_trauma_center:
            specialty_score = 1.0
        elif is_cardiac and has_cardiac_lab:
            specialty_score = 1.0
        elif has_trauma_center or has_cardiac_lab:
            specialty_score = 0.7
        
        # 5. Emergency Department Load
        ed_capacity = row.get('ed_capacity', 50)
        ed_load = row.get('current_ed_load', 0)
        ed_utilization = ed_load / ed_capacity if ed_capacity > 0 else 1.0
        norm_ed = 1.0 - min(ed_utilization, 1.0)  # Less crowded = better
        
        # 6. Combined Score with weights
        features = np.array([norm_dist, norm_icu, norm_staff, specialty_score, norm_ed])
        weights = np.array([0.30, 0.25, 0.15, 0.20, 0.10])  # Prioritize proximity and specialty
        score = np.dot(features, weights)
        
        scored_hospitals.append({
            "id": row['hospital_id'],
            "name": row['hospital_name'], 
            "lat": row['lat'],
            "lng": row['lng'],
            "score": score,
            "dist_km": dist_km,
            "icu_free": int(icu_free),
            "has_trauma": has_trauma_center > 0,
            "has_cardiac": has_cardiac_lab > 0,
            "ed_load": int(ed_load),
            "ed_capacity": int(ed_capacity),
            "specialties": row.get('specialties', 'General'),
            "staff_doctors": int(staff_count)
        })
    
    # Select Best
    best = sorted(scored_hospitals, key=lambda x: x['score'], reverse=True)[0]
    
    # Generate Detailed Explanation
    reasons = []
    reasons.append(f"{best['dist_km']:.1f}km away")
    if best['icu_free'] > 0:
        reasons.append(f"{best['icu_free']} ICU beds available")
    if best['has_trauma']:
        reasons.append("✓ Trauma Center")
    if best['has_cardiac']:
        reasons.append("✓ Cardiac Cath Lab")
    reasons.append(f"ED Load: {best['ed_load']}/{best['ed_capacity']}")
    reasons.append(f"{best['staff_doctors']} doctors on staff")
    reasons.append(f"Score: {best['score']:.2f}")
    
    return {
        "selected_hospital_id": best['id'],
        "selected_hospital_name": best['name'],
        "selected_hospital_lat": best['lat'],
        "selected_hospital_lng": best['lng'],
        "hospital_distance": best['dist_km'],
        "hospital_explanation": " | ".join(reasons),
        "hospital_icu_beds": best['icu_free'],
        "hospital_specialties": str(best['specialties'])
    }

def doctor_node(state: AgentState):
    print(f"--- [Step 3] Matching Doctor & Creating Handover Plan ---")
    
    if not vector_db: 
        return {
            "selected_doctor_name": "N/A",
            "selected_doctor_specialty": "N/A",
            "doctor_explanation": "❌ Doctor database not available",
            "final_plan": json.dumps({"error": "Doctor DB not loaded"}),
            "complete_summary": {}
        }
    
    hosp_id = state['selected_hospital_id']
    hosp_name = state['selected_hospital_name']
    
    # Retrieval: Find doctors at the selected hospital matching the emergency type
    try:
        retriever = vector_db.as_retriever(
            search_kwargs={
                "k": 5, 
                "filter": {"hospital_id": hosp_id}
            }
        )
        
        query = f"""
        Emergency Type: {state['accident_type']}
        Severity: {state['severity_level']}
        Requirements: Find doctors with relevant specialty, high seniority, on-call status, trauma experience
        """
        docs = retriever.invoke(query)
        
        if not docs:
            # Fallback: no filter
            print("[WARN] No doctors found with filter, retrying without hospital filter...")
            retriever = vector_db.as_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke(query)
        
        doc_context = "\n\n".join([d.page_content for d in docs]) if docs else "No doctors available"
        
    except Exception as e:
        print(f"[ERROR] Doctor retrieval failed: {e}")
        doc_context = "No doctors available"
    
    # Generation using LLM
    groq_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API")
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_key)
    
    prompt = f"""
    System: You are an emergency medical dispatch AI. Generate a structured medical handover plan.
    
    EMERGENCY CONTEXT:
    - Emergency Type: {state['accident_type']}
    - Severity: {state['severity_level']}
    - Location: {state['accident_location']}
    - Pediatric Case: {state.get('is_child', False)}
    
    DISPATCH STATUS:
    - Ambulance: {state['assigned_ambulance']}
    - ETA: {state['ambulance_eta']} minutes
    - Distance: {state['ambulance_distance']} km
    
    DESTINATION HOSPITAL:
    - Name: {hosp_name}
    - Hospital ID: {hosp_id}
    - Distance: {state['hospital_distance']} km
    - ICU Beds Available: {state['hospital_icu_beds']}
    - Specialties: {state['hospital_specialties']}
    
    AVAILABLE DOCTORS:
    {doc_context}
    
    TASK: 
    1. Select the BEST doctor from the available list for this specific emergency
    2. Consider: specialty match, seniority, on-call status, experience
    3. Generate a complete handover plan
    
    OUTPUT REQUIREMENTS:
    - Return ONLY valid JSON (no markdown, no code blocks, no explanations)
    - Must be parseable by json.loads()
    
    Required JSON Structure:
    {{
      "emergency_summary": {{
        "type": "{state['accident_type']}",
        "severity": "{state['severity_level']}",
        "eta_minutes": {state['ambulance_eta']},
        "ambulance_id": "{state['assigned_ambulance']}"
      }},
      "hospital": {{
        "name": "{hosp_name}",
        "id": "{hosp_id}",
        "icu_beds_free": {state['hospital_icu_beds']},
        "distance_km": {state['hospital_distance']}
      }},
      "assigned_doctor": {{
        "name": "Dr. Full Name",
        "specialty": "Specialty",
        "sub_specialty": "Sub-specialty",
        "seniority": "Level",
        "on_call": true,
        "selection_reason": "Brief reason for selection"
      }},
      "immediate_actions": [
        "Action 1: Specific preparation needed",
        "Action 2: Resources to prepare",
        "Action 3: Team to alert"
      ],
      "required_resources": {{
        "bed_type": "ICU/ED/Trauma Bay",
        "equipment": ["Item1", "Item2"],
        "medications": ["Med1", "Med2"],
        "specialists_on_standby": ["Specialist1"]
      }},
      "estimated_timeline": {{
        "ambulance_arrival": "{state['ambulance_eta']} minutes",
        "prep_time_needed": "X minutes",
        "handover_location": "Location"
      }}
    }}
    """
    
    try:
        response = llm.invoke(prompt)
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        
        # Try to parse JSON
        try:
            plan_json = json.loads(clean_content)
        except json.JSONDecodeError:
            # Fallback: extract JSON from text
            import re
            json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', clean_content, re.DOTALL)
            if json_match:
                plan_json = json.loads(json_match.group(0))
            else:
                raise ValueError("No valid JSON found in LLM response")
        
        # Extract doctor info
        doctor_info = plan_json.get('assigned_doctor', {})
        doctor_name = doctor_info.get('name', 'Dr. Unknown')
        doctor_specialty = doctor_info.get('specialty', 'General Medicine')
        doctor_reason = doctor_info.get('selection_reason', 'Best available match')
        
        # Create complete summary
        complete_summary = {
            "request_id": state['request_id'],
            "timestamp": dt.datetime.utcnow().isoformat(),
            "emergency": {
                "type": state['accident_type'],
                "severity": state['severity_level'],
                "location": state['accident_location'],
                "is_child": state.get('is_child', False)
            },
            "ambulance": {
                "id": state['assigned_ambulance'],
                "eta_minutes": state['ambulance_eta'],
                "distance_km": state['ambulance_distance'],
                "route": state.get('ambulance_route', {})
            },
            "hospital": {
                "id": state['selected_hospital_id'],
                "name": state['selected_hospital_name'],
                "distance_km": state['hospital_distance'],
                "icu_beds": state['hospital_icu_beds']
            },
            "doctor": doctor_info,
            "handover_plan": plan_json
        }
        
        return {
            "selected_doctor_name": doctor_name,
            "selected_doctor_specialty": doctor_specialty,
            "doctor_explanation": doctor_reason,
            "final_plan": json.dumps(plan_json, indent=2),
            "complete_summary": complete_summary
        }
        
    except Exception as e:
        print(f"[ERROR] Doctor node LLM generation failed: {e}")
        fallback_plan = {
            "error": str(e),
            "emergency": state['accident_type'],
            "hospital": hosp_name,
            "ambulance": state['assigned_ambulance'],
            "eta": state['ambulance_eta']
        }
        return {
            "selected_doctor_name": "Duty Doctor",
            "selected_doctor_specialty": "Emergency Medicine",
            "doctor_explanation": "Fallback assignment - LLM unavailable",
            "final_plan": json.dumps(fallback_plan, indent=2),
            "complete_summary": fallback_plan
        }

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